import threading
import time
from types import SimpleNamespace

import pytest

from runtime_model import RuntimeDirectory, RuntimeEndpoint, RuntimeSlot, RuntimeUnit
from runtime_orchestrator import RuntimeOperationCancelled, RuntimeRetirementPending

INSTALL_ID = "11111111-1111-4111-8111-111111111111"
REPLACEMENT_INSTALL_ID = "22222222-2222-4222-8222-222222222222"


def _unit(component, port, revision=1):
    slot = RuntimeSlot(component, "cloud-a", "cloud")
    runtime_id = slot.runtime_name(revision, "install-a")
    return RuntimeUnit(
        slot=slot,
        runtime_id=runtime_id,
        runtime_revision=revision,
        spec_hash="a" * 64,
        endpoint=RuntimeEndpoint(
            dns_name=f"{runtime_id}.dayu.svc.cluster.local",
            port=port,
            runtime_service_uid=f"{component}-runtime-uid",
            service_uid=f"{component}-service-uid",
            pod_uid=f"{component}-pod-uid",
        ),
    )


def _directory(revision=1):
    return RuntimeDirectory(
        install_id="install-a",
        revision=revision,
        routes=(_unit("scheduler", 9001, revision), _unit("distributor", 9003, revision)),
    )


class FakeOrchestrator:
    def __init__(self):
        self.directory = _directory()
        self.session = SimpleNamespace(
            phase="active",
            policy_id="policy-a",
            install_id="install-a",
            operation_id="operation-a",
            updated_at="2026-07-16T00:00:00Z",
            active_directory_revision=1,
            active=(),
            pending=(),
            cleanup=(),
            retirement=None,
            last_error="",
        )
        self.install_calls = []
        self.begin_uninstall_calls = 0
        self.uninstall_calls = 0
        self.redeploy_calls = []
        self.reconcile_calls = []
        self.install_error = None
        self.begin_uninstall_error = None
        self.uninstall_error = None
        self.redeploy_error = None
        self.reconcile_error = None
        self.changed = True

    def install(self, **kwargs):
        self.install_calls.append(kwargs)
        if self.install_error:
            raise self.install_error
        install_id = kwargs["install_id"]
        self.directory = RuntimeDirectory(
            install_id=install_id,
            revision=self.directory.revision,
            routes=self.directory.routes,
        )
        self.session = SimpleNamespace(
            phase="active",
            policy_id="policy-a",
            install_id=install_id,
            operation_id="operation-a",
            updated_at="2026-07-16T00:00:00Z",
            active_directory_revision=self.directory.revision,
            active=(),
            pending=(),
            cleanup=(),
            retirement=None,
            last_error="",
        )
        return self.directory

    def uninstall(self, expected_install_id=""):
        self.uninstall_calls += 1
        if self.uninstall_error:
            raise self.uninstall_error
        self.session = None

    def begin_uninstall(self, expected_install_id=""):
        self.begin_uninstall_calls += 1
        if self.begin_uninstall_error:
            raise self.begin_uninstall_error
        if self.session is None:
            return None
        if expected_install_id and self.session.install_id != expected_install_id:
            return None
        self.session.phase = "uninstalling"
        return self.session

    def current_session(self):
        return self.session

    def redeploy(self, policy, cancel_event=None):
        self.redeploy_calls.append(policy)
        if self.redeploy_error:
            raise self.redeploy_error
        return self.changed

    def reconcile_retirement(self, cancel_event=None):
        self.reconcile_calls.append(cancel_event)
        if self.reconcile_error:
            raise self.reconcile_error
        return False

    def recover(self):
        return self.session

    @staticmethod
    def requires_recovery(session):
        return session is not None and session.phase in {
            "activating-scheduler", "activating-runtime", "publishing",
            "activating-rollout", "publishing-rollout",
        }

    def active_directory(self):
        return self.directory


class FakeRuntimeTelemetry:
    def __init__(self):
        self.bound_urls = []
        self.bound_directories = []
        self.started = 0
        self.unbound = 0
        self.closed = 0

    def bind(self, url, directory):
        self.bound_urls.append(url)
        self.bound_directories.append(directory)

    def start(self):
        self.started += 1

    def unbind(self):
        self.unbound += 1

    def close(self):
        self.closed += 1


@pytest.fixture
def backend_core_instance(mounted_runtime):
    from backend_core import BackendCore

    instance = BackendCore()
    instance.runtime_orchestrator = FakeOrchestrator()
    instance.runtime_telemetry = FakeRuntimeTelemetry()
    return instance


def test_install_delegates_transaction_and_starts_runtime_reconcile_worker(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.session = None
    # Retirement reconciliation is required even when policy-driven rollout
    # is disabled, so zero interval must not suppress this worker.
    backend_core_instance.processor_redeployment_interval_s = 0
    started = []

    class FakeThread:
        def __init__(self, **kwargs):
            started.append(kwargs)

        def start(self):
            started[-1]["started"] = True

    import backend_core as backend_core_module

    monkeypatch.setattr(backend_core_module.threading, "Thread", FakeThread)
    policy = {"id": "policy-a"}
    source_deploy = [{"source": {"id": 0}, "dag": {}, "node_set": ["edge-a"]}]

    ok, message = backend_core_instance.parse_and_apply_templates(
        policy, source_deploy, source_label="source-a", install_id=INSTALL_ID
    )

    assert (ok, message) == (True, "Install services successfully")
    assert len(backend_core_instance.runtime_orchestrator.install_calls) == 1
    install_call = backend_core_instance.runtime_orchestrator.install_calls[0]
    cancel_event = install_call.pop("cancel_event")
    assert install_call == {
        "policy": policy,
        "source_deploy": source_deploy,
        "source_label": "source-a",
        "install_id": INSTALL_ID,
    }
    assert isinstance(cancel_event, threading.Event)
    assert cancel_event.is_set() is False
    assert backend_core_instance.resource_url.endswith(":9001/resource")
    assert backend_core_instance.result_url.endswith(":9003/result")
    assert backend_core_instance.runtime_telemetry.bound_urls == [
        backend_core_instance.resource_url
    ]
    assert backend_core_instance.runtime_telemetry.bound_directories[0].install_id == INSTALL_ID
    assert backend_core_instance.runtime_telemetry.started == 1
    assert started[0]["name"] == f"dayu-runtime-reconcile-{INSTALL_ID}"
    assert started[0]["daemon"] is True
    assert started[0]["started"] is True
    assert started[0]["args"][1] == INSTALL_ID
    assert backend_core_instance._query_admission_enabled is True


def test_install_starts_real_reconcile_worker_without_lifecycle_lock_reentry(
        backend_core_instance,
):
    backend_core_instance.runtime_orchestrator.session = None
    backend_core_instance.processor_redeployment_interval_s = 0
    install_result = []
    install_thread = threading.Thread(
        target=lambda: install_result.append(
            backend_core_instance.parse_and_apply_templates(
                {"id": "policy-a"},
                [],
                install_id=INSTALL_ID,
            ),
        ),
    )

    install_thread.start()
    install_thread.join(timeout=1)
    reconcile_thread = backend_core_instance._runtime_reconcile_thread
    try:
        assert install_thread.is_alive() is False
        assert install_result == [(True, "Install services successfully")]
        assert reconcile_thread is not None
        assert reconcile_thread.is_alive() is True
    finally:
        backend_core_instance._stop_runtime_reconcile_loop()
        if reconcile_thread is not None:
            reconcile_thread.join(timeout=1)


def test_install_lazy_snapshot_load_does_not_hold_lifecycle_admission_lock(
        backend_core_instance,
):
    backend_core_instance.runtime_orchestrator.session = None
    load_started = threading.Event()
    release_load = threading.Event()
    install_result = []
    current_session = backend_core_instance.runtime_orchestrator.current_session
    calls = 0

    def blocking_initial_load():
        nonlocal calls
        calls += 1
        if calls == 1:
            load_started.set()
            assert release_load.wait(1)
        return current_session()

    backend_core_instance.runtime_orchestrator.current_session = blocking_initial_load
    install_thread = threading.Thread(
        target=lambda: install_result.append(
            backend_core_instance.parse_and_apply_templates(
                {"id": "policy-a"}, [], install_id=INSTALL_ID,
            ),
        ),
    )
    install_thread.start()
    assert load_started.wait(1)

    acquired = backend_core_instance._lifecycle_control_lock.acquire(timeout=0.2)
    if acquired:
        backend_core_instance._lifecycle_control_lock.release()
    release_load.set()
    install_thread.join(timeout=1)
    reconcile_thread = backend_core_instance._runtime_reconcile_thread
    backend_core_instance._stop_runtime_reconcile_loop()
    if reconcile_thread is not None:
        reconcile_thread.join(timeout=1)

    assert acquired is True
    assert install_thread.is_alive() is False
    assert install_result == [(True, "Install services successfully")]


def test_install_failure_is_reported_without_starting_runtime_worker(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.session = None
    backend_core_instance.runtime_orchestrator.install_error = RuntimeError("activation failed")
    import backend_core as backend_core_module

    monkeypatch.setattr(
        backend_core_module.threading,
        "Thread",
        lambda **kwargs: pytest.fail("runtime worker must not start after failed install"),
    )

    ok, message = backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID)

    assert ok is False
    assert "activation failed" in message
    assert backend_core_instance._runtime_reconcile_stop_event is None


@pytest.mark.parametrize(
    "phase", ["activating-scheduler", "activating-runtime", "publishing"],
)
def test_recoverable_initial_failure_starts_background_recovery_controller(
        backend_core_instance, monkeypatch, phase,
):
    backend_core_instance.runtime_orchestrator.session = None
    recovery_starts = []

    def fail_during_publication(**kwargs):
        backend_core_instance.runtime_orchestrator.session = SimpleNamespace(
            install_id=kwargs["install_id"],
            phase=phase,
        )
        raise RuntimeError("publication response lost")

    backend_core_instance.runtime_orchestrator.install = fail_during_publication
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_recovery_async",
        lambda: recovery_starts.append(True),
    )

    ok, message = backend_core_instance.parse_and_apply_templates(
        {}, [], install_id=INSTALL_ID,
    )

    assert ok is False
    assert "publication response lost" in message
    assert recovery_starts == [True]
    assert backend_core_instance._install_admission is None


def test_ambiguous_initial_session_read_starts_background_recovery_controller(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.session = None
    recovery_starts = []
    reads = 0

    def current_session():
        nonlocal reads
        reads += 1
        # Admission performs one lock-free lazy load and then one memory-only
        # sample inside lifecycle control before the install transaction.
        if reads <= 2:
            return None
        raise RuntimeError("ConfigMap read unavailable")

    backend_core_instance.runtime_orchestrator.current_session = current_session
    backend_core_instance.runtime_orchestrator.install_error = RuntimeError(
        "ConfigMap write response lost",
    )
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_recovery_async",
        lambda: recovery_starts.append(True),
    )

    ok, message = backend_core_instance.parse_and_apply_templates(
        {}, [], install_id=INSTALL_ID,
    )

    assert ok is False
    assert "write response lost" in message
    assert recovery_starts == [True]


def test_local_projection_failure_is_terminal_but_retried_in_background(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.session = None
    recovery_starts = []
    monkeypatch.setattr(
        backend_core_instance,
        "_bind_runtime_urls",
        lambda _directory: (_ for _ in ()).throw(RuntimeError("bind failed")),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_recovery_async",
        lambda: recovery_starts.append(True),
    )

    ok, message = backend_core_instance.parse_and_apply_templates(
        {}, [], install_id=INSTALL_ID,
    )
    session, _, ready, local_error = (
        backend_core_instance.management_lifecycle_snapshot()
    )

    assert ok is False
    assert "bind failed" in message
    assert session.phase == "active"
    assert ready is False
    assert "local runtime activation failed" in local_error
    assert recovery_starts == [True]


def test_uninstall_cancels_inflight_install_before_persisting_stop_intent(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.session = None
    operation_lock = threading.Lock()
    install_started = threading.Event()
    install_results = []
    seen_cancel_events = []

    def blocking_install(**kwargs):
        cancel_event = kwargs["cancel_event"]
        seen_cancel_events.append(cancel_event)
        with operation_lock:
            backend_core_instance.runtime_orchestrator.session = SimpleNamespace(
                phase="activating-runtime",
                policy_id="policy-a",
                install_id=kwargs["install_id"],
            )
            install_started.set()
            assert cancel_event.wait(2)
            raise RuntimeOperationCancelled("cancelled by lifecycle operation")

    def serialized_begin_uninstall(expected_install_id=""):
        with operation_lock:
            assert expected_install_id == INSTALL_ID
            backend_core_instance.runtime_orchestrator.begin_uninstall_calls += 1
            backend_core_instance.runtime_orchestrator.session.phase = "uninstalling"
            return backend_core_instance.runtime_orchestrator.session

    backend_core_instance.runtime_orchestrator.install = blocking_install
    backend_core_instance.runtime_orchestrator.begin_uninstall = serialized_begin_uninstall
    started_workers = []
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: started_workers.append(install_id),
    )

    install_thread = threading.Thread(
        target=lambda: install_results.append(
            backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID),
        ),
    )
    install_thread.start()
    assert install_started.wait(1)

    started_at = time.monotonic()
    uninstall_result = backend_core_instance.parse_and_delete_templates(INSTALL_ID)
    elapsed = time.monotonic() - started_at
    install_thread.join(timeout=1)

    assert install_thread.is_alive() is False
    assert elapsed < 0.5
    assert uninstall_result == (True, "Uninstall services started")
    assert install_results == [(False, "Install cancelled by lifecycle operation")]
    assert len(seen_cancel_events) == 1
    assert seen_cancel_events[0].is_set() is True
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 1
    assert backend_core_instance.runtime_orchestrator.uninstall_calls == 0
    assert started_workers == [INSTALL_ID]
    assert backend_core_instance._install_admission is None
    assert backend_core_instance._stop_admission is None


def test_identity_bound_stop_cancels_install_before_session_persistence(
        backend_core_instance,
):
    backend_core_instance.runtime_orchestrator.session = None
    operation_lock = threading.Lock()
    install_started = threading.Event()
    install_results = []

    def blocking_install(**kwargs):
        with operation_lock:
            install_started.set()
            assert kwargs["cancel_event"].wait(2)
            raise RuntimeOperationCancelled("cancelled before persistence")

    def serialized_begin_uninstall(expected_install_id=""):
        assert expected_install_id == INSTALL_ID
        with operation_lock:
            return None

    backend_core_instance.runtime_orchestrator.install = blocking_install
    backend_core_instance.runtime_orchestrator.begin_uninstall = serialized_begin_uninstall
    install_thread = threading.Thread(
        target=lambda: install_results.append(
            backend_core_instance.parse_and_apply_templates(
                {}, [], install_id=INSTALL_ID,
            )
        )
    )
    install_thread.start()
    assert install_started.wait(1)

    stop_result = backend_core_instance.parse_and_delete_templates(INSTALL_ID)
    install_thread.join(timeout=1)

    assert stop_result == (True, "No managed services are installed")
    assert install_results == [(False, "Install cancelled by lifecycle operation")]
    assert install_thread.is_alive() is False
    assert backend_core_instance.management_lifecycle_snapshot() == (
        None, None, False, "",
    )


def test_install_cancellation_is_observable_until_install_releases_admission(
        backend_core_instance,
):
    backend_core_instance.runtime_orchestrator.session = None
    install_started = threading.Event()
    cancellation_seen = threading.Event()
    release_install = threading.Event()
    install_results = []
    stop_results = []

    def blocking_install(**kwargs):
        install_started.set()
        assert kwargs["cancel_event"].wait(1)
        cancellation_seen.set()
        assert release_install.wait(1)
        raise RuntimeOperationCancelled("cancelled")

    backend_core_instance.runtime_orchestrator.install = blocking_install
    install_thread = threading.Thread(
        target=lambda: install_results.append(
            backend_core_instance.parse_and_apply_templates(
                {}, [], install_id=INSTALL_ID,
            ),
        ),
    )
    install_thread.start()
    assert install_started.wait(1)
    stop_thread = threading.Thread(
        target=lambda: stop_results.append(
            backend_core_instance.parse_and_delete_templates(INSTALL_ID),
        ),
    )
    stop_thread.start()
    assert cancellation_seen.wait(1)

    session, pending, ready, error = (
        backend_core_instance.management_lifecycle_snapshot()
    )
    assert session is None
    assert pending["kind"] == "install"
    assert pending["install_id"] == INSTALL_ID
    assert pending["phase"] == "cancelling-install"
    assert pending["operation_id"]
    assert ready is False
    assert error == ""

    release_install.set()
    install_thread.join(timeout=1)
    stop_thread.join(timeout=1)
    assert install_thread.is_alive() is False
    assert stop_thread.is_alive() is False
    assert install_results == [(False, "Install cancelled by lifecycle operation")]
    assert stop_results == [(True, "No managed services are installed")]


def test_stop_registration_prevents_install_token_race(
        backend_core_instance, monkeypatch,
):
    uninstall_started = threading.Event()
    release_uninstall = threading.Event()
    stop_results = []

    def blocking_begin_uninstall(expected_install_id=""):
        uninstall_started.set()
        assert release_uninstall.wait(2)
        backend_core_instance.runtime_orchestrator.session.phase = "uninstalling"
        return backend_core_instance.runtime_orchestrator.session

    backend_core_instance.runtime_orchestrator.begin_uninstall = blocking_begin_uninstall
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: None,
    )
    stop_thread = threading.Thread(
        target=lambda: stop_results.append(
            backend_core_instance.parse_and_delete_templates(),
        ),
    )
    stop_thread.start()
    assert uninstall_started.wait(1)

    assert backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID) == (
        False,
        "Install cancelled by lifecycle operation",
    )
    assert backend_core_instance.runtime_orchestrator.install_calls == []

    release_uninstall.set()
    stop_thread.join(timeout=1)
    assert stop_thread.is_alive() is False
    assert stop_results == [(True, "Uninstall services started")]


def test_stop_registration_closes_query_admission_before_waiting_for_install(
        backend_core_instance, monkeypatch,
):
    cancel_event = threading.Event()
    install_done = threading.Event()
    backend_core_instance._install_admission = SimpleNamespace(
        install_id="install-a",
        phase="preparing-install",
        operation_id="install-operation",
        cancel_event=cancel_event,
        done_event=install_done,
        cancel=cancel_event.set,
    )
    backend_core_instance.result_url = "http://distributor/result"
    backend_core_instance._query_admission_enabled = True
    query_stop = threading.Event()
    backend_core_instance._query_cancel_event = query_stop
    backend_core_instance.source_open = True
    backend_core_instance.source_label = "source-a"
    backend_core_instance.source_configs = [{
        "source_label": "source-a",
        "source_list": [{"id": 0}],
    }]
    stop_results = []
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda _install_id: None,
    )
    stop_thread = threading.Thread(
        target=lambda: stop_results.append(
            backend_core_instance.parse_and_delete_templates(),
        ),
    )
    stop_thread.start()
    assert cancel_event.wait(1)
    # Waiting for this boundary proves stop registration and its query fence
    # are both visible, while the stop leader is still waiting on install_done.
    with backend_core_instance._lifecycle_control_lock:
        pass

    assert query_stop.is_set() is True
    assert backend_core_instance.source_open is False
    assert backend_core_instance.open_query("source-a") == (
        False,
        "Runtime is not ready for datasource queries",
    )
    assert stop_thread.is_alive() is True

    install_done.set()
    stop_thread.join(timeout=1)
    assert stop_thread.is_alive() is False
    assert stop_results == [(True, "Uninstall services started")]


@pytest.mark.parametrize("failure_point", ["query", "telemetry", "runtime-reconcile"])
def test_stop_registration_is_released_when_local_shutdown_raises(
        backend_core_instance, monkeypatch, failure_point,
):
    error = RuntimeError(f"{failure_point} shutdown failed")
    if failure_point == "query":
        monkeypatch.setattr(
            backend_core_instance,
            "_close_query_locked",
            lambda: (_ for _ in ()).throw(error),
        )
    elif failure_point == "telemetry":
        monkeypatch.setattr(
            backend_core_instance.runtime_telemetry,
            "unbind",
            lambda: (_ for _ in ()).throw(error),
        )
    else:
        monkeypatch.setattr(
            backend_core_instance,
            "_stop_runtime_reconcile_loop",
            lambda: (_ for _ in ()).throw(error),
        )

    ok, message = backend_core_instance.parse_and_delete_templates()

    assert ok is False
    assert str(error) in message
    assert backend_core_instance._stop_admission is None


def test_overlapping_stop_requests_keep_install_admission_closed_until_both_finish(
        backend_core_instance, monkeypatch,
):
    uninstall_started = threading.Event()
    release_uninstalls = threading.Event()
    results = []

    def blocking_begin_uninstall(expected_install_id=""):
        backend_core_instance.runtime_orchestrator.begin_uninstall_calls += 1
        uninstall_started.set()
        assert release_uninstalls.wait(2)
        backend_core_instance.runtime_orchestrator.session.phase = "uninstalling"
        return backend_core_instance.runtime_orchestrator.session

    backend_core_instance.runtime_orchestrator.begin_uninstall = blocking_begin_uninstall
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: None,
    )
    thread = threading.Thread(
        target=lambda: results.append(
            backend_core_instance.parse_and_delete_templates(),
        ),
    )
    thread.start()
    assert uninstall_started.wait(1)

    assert backend_core_instance._stop_admission is not None
    session, pending, ready, error = (
        backend_core_instance.management_lifecycle_snapshot()
    )
    assert session.install_id == "install-a"
    assert pending["kind"] == "stop"
    assert pending["install_id"] == "install-a"
    assert pending["phase"] == "preparing-uninstall"
    assert pending["operation_id"]
    assert ready is False
    assert error == ""
    assert backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID) == (
        False,
        "Install cancelled by lifecycle operation",
    )
    follower_entered = threading.Event()
    follower_results = []
    follower = threading.Thread(
        target=lambda: (
            follower_entered.set(),
            follower_results.append(
                backend_core_instance.parse_and_delete_templates(),
            ),
        ),
    )
    follower.start()
    assert follower_entered.wait(1)
    assert follower_results == []
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 1

    release_uninstalls.set()
    thread.join(timeout=1)
    follower.join(timeout=1)
    assert thread.is_alive() is False
    assert follower.is_alive() is False

    assert results == [(True, "Uninstall services started")]
    assert follower_results == [(True, "Uninstall services started")]
    assert backend_core_instance._stop_admission is None


@pytest.mark.parametrize("phase", ["uninstalling", "finalizing-uninstall"])
def test_repeated_async_uninstall_keeps_existing_worker_and_returns_immediately(
        backend_core_instance, monkeypatch, phase,
):
    backend_core_instance.runtime_orchestrator.session.phase = phase
    worker_stop = threading.Event()
    backend_core_instance._runtime_reconcile_stop_event = worker_stop
    backend_core_instance._runtime_reconcile_thread = object()
    monkeypatch.setattr(
        backend_core_instance,
        "_stop_runtime_reconcile_loop",
        lambda: pytest.fail("an idempotent uninstall must not stop its cleanup worker"),
    )
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: pytest.fail("the running cleanup worker must be reused"),
    )

    started_at = time.monotonic()
    result = backend_core_instance.parse_and_delete_templates()

    assert result == (True, "Uninstall services started")
    assert time.monotonic() - started_at < 0.5
    assert worker_stop.is_set() is False
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 0
    assert backend_core_instance._stop_admission is None


@pytest.mark.parametrize("phase", ["uninstalling", "finalizing-uninstall"])
def test_install_is_rejected_from_cached_uninstall_state_without_entering_transaction(
        backend_core_instance, phase,
):
    backend_core_instance.runtime_orchestrator.session.phase = phase

    started_at = time.monotonic()
    result = backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID)

    assert result == (False, "Uninstall is in progress")
    assert time.monotonic() - started_at < 0.5
    assert backend_core_instance.runtime_orchestrator.install_calls == []
    assert backend_core_instance._install_admission is None


def test_install_is_rejected_from_active_session_without_entering_transaction(
        backend_core_instance,
):
    backend_core_instance.runtime_orchestrator.session.phase = "active"

    result = backend_core_instance.parse_and_apply_templates(
        {}, [], install_id=INSTALL_ID,
    )

    assert result == (
        False,
        "A managed runtime session already exists; uninstall it before installing",
    )
    assert backend_core_instance.runtime_orchestrator.install_calls == []
    assert backend_core_instance._install_admission is None
    assert backend_core_instance.management_lifecycle_snapshot()[1] is None


def test_stale_target_stop_does_not_touch_replacement_session(backend_core_instance):
    replacement = backend_core_instance.runtime_orchestrator.session

    result = backend_core_instance.parse_and_delete_templates(INSTALL_ID)

    assert result == (True, "Target installation is already absent")
    assert backend_core_instance.runtime_orchestrator.session is replacement
    assert replacement.phase == "active"
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 0
    assert backend_core_instance._stop_admission is None


def test_stop_rejects_noncanonical_install_identity(backend_core_instance):
    replacement = backend_core_instance.runtime_orchestrator.session

    result = backend_core_instance.parse_and_delete_templates("not-an-install-id")

    assert result == (False, "install_id must be a canonical UUID")
    assert backend_core_instance.runtime_orchestrator.session is replacement
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 0
    assert backend_core_instance._stop_admission is None


def test_management_snapshot_never_performs_first_session_load_under_admission_lock(
        backend_core_instance, monkeypatch,
):
    lock_states = []

    def current_session():
        acquired = backend_core_instance._lifecycle_control_lock.acquire(
            blocking=False,
        )
        if acquired:
            backend_core_instance._lifecycle_control_lock.release()
        lock_states.append(not acquired)
        return None

    monkeypatch.setattr(
        backend_core_instance.runtime_orchestrator,
        "current_session",
        current_session,
    )

    assert backend_core_instance.management_lifecycle_snapshot() == (
        None, None, False, "",
    )
    assert lock_states == [False, True]


def test_concurrent_second_install_cannot_overwrite_cancel_token(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.session = None
    first_started = threading.Event()
    release_first = threading.Event()
    first_result = []
    tokens = []

    def blocking_install(**kwargs):
        tokens.append(kwargs["cancel_event"])
        first_started.set()
        assert release_first.wait(2)
        backend_core_instance.runtime_orchestrator.directory = RuntimeDirectory(
            install_id=kwargs["install_id"],
            revision=1,
            routes=_directory().routes,
        )
        backend_core_instance.runtime_orchestrator.session = SimpleNamespace(
            phase="active",
            policy_id="policy-a",
            install_id=kwargs["install_id"],
        )
        return backend_core_instance.runtime_orchestrator.directory

    backend_core_instance.runtime_orchestrator.install = blocking_install
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: None,
    )
    first_thread = threading.Thread(
        target=lambda: first_result.append(
            backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID),
        ),
    )
    first_thread.start()
    assert first_started.wait(1)
    assert backend_core_instance.management_lifecycle_snapshot()[1]["install_id"] == INSTALL_ID

    assert backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID) == (
        False,
        "Another install operation is already in progress",
    )
    assert backend_core_instance._install_admission.cancel_event is tokens[0]

    release_first.set()
    first_thread.join(timeout=1)
    assert first_thread.is_alive() is False
    assert first_result == [(True, "Install services successfully")]
    assert len(tokens) == 1
    assert backend_core_instance._install_admission is None
    assert backend_core_instance.management_lifecycle_snapshot()[1] is None


def test_backend_startup_recovery_rebinds_directory_and_restarts_runtime_worker(
    backend_core_instance, monkeypatch
):
    started = []
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: started.append(install_id),
    )

    backend_core_instance._recover_runtime_session()

    assert backend_core_instance.resource_url.endswith(":9001/resource")
    assert backend_core_instance.result_url.endswith(":9003/result")
    assert started == ["install-a"]
    assert backend_core_instance.runtime_telemetry.started == 1
    assert backend_core_instance._query_admission_enabled is True


def test_runtime_recovery_does_not_publish_local_state_after_close(
        backend_core_instance, monkeypatch,
):
    recover_started = threading.Event()
    release_recover = threading.Event()
    stop_event = threading.Event()
    projections = []

    def blocking_recover():
        recover_started.set()
        assert release_recover.wait(1)
        return backend_core_instance.runtime_orchestrator.session

    backend_core_instance.runtime_orchestrator.recover = blocking_recover
    monkeypatch.setattr(
        backend_core_instance,
        "_activate_local_runtime",
        lambda *args, **kwargs: projections.append((args, kwargs)),
    )
    worker = threading.Thread(
        target=backend_core_instance.run_runtime_recovery,
        args=(stop_event,),
    )
    worker.start()
    assert recover_started.wait(1)

    stop_event.set()
    release_recover.set()
    worker.join(timeout=1)

    assert worker.is_alive() is False
    assert projections == []


def test_runtime_recovery_does_not_restart_uninstall_worker_after_close(
        backend_core_instance, monkeypatch,
):
    recover_started = threading.Event()
    release_recover = threading.Event()
    stop_event = threading.Event()
    started_workers = []
    backend_core_instance.runtime_orchestrator.session.phase = "uninstalling"

    def blocking_recover():
        recover_started.set()
        assert release_recover.wait(1)
        return backend_core_instance.runtime_orchestrator.session

    backend_core_instance.runtime_orchestrator.recover = blocking_recover
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: started_workers.append(install_id),
    )
    worker = threading.Thread(
        target=backend_core_instance.run_runtime_recovery,
        args=(stop_event,),
    )
    worker.start()
    assert recover_started.wait(1)

    stop_event.set()
    release_recover.set()
    worker.join(timeout=1)

    assert worker.is_alive() is False
    assert started_workers == []


def test_runtime_recovery_start_is_single_flight_and_close_stops_it(
        backend_core_instance, monkeypatch,
):
    import backend_core as backend_core_module

    workers = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            workers.append(self)

        def start(self):
            return None

    monkeypatch.setenv("DAYU_RUNTIME_CONTROL_PLANE", "true")
    monkeypatch.setattr(backend_core_module.threading, "Thread", FakeThread)

    backend_core_instance.start()
    backend_core_instance.start()

    assert len(workers) == 1
    assert workers[0].kwargs["name"] == "dayu-runtime-recovery"
    assert backend_core_instance._runtime_recovery_stop_event.is_set() is False

    backend_core_instance.close()
    assert backend_core_instance._runtime_recovery_stop_event.is_set() is True


def test_runtime_recovery_trigger_is_not_lost_while_worker_is_finishing(
        backend_core_instance, monkeypatch,
):
    first_attempt_started = threading.Event()
    release_first_attempt = threading.Event()
    second_attempt_finished = threading.Event()
    attempts = []

    def recover_once_more(stop_event=None):
        attempts.append(stop_event)
        if len(attempts) == 1:
            first_attempt_started.set()
            assert release_first_attempt.wait(1)
        else:
            second_attempt_finished.set()
        return True

    monkeypatch.setattr(
        backend_core_instance,
        "_recover_runtime_session",
        recover_once_more,
    )

    backend_core_instance._start_runtime_recovery_async()
    worker = backend_core_instance._runtime_recovery_thread
    assert first_attempt_started.wait(1)
    backend_core_instance._start_runtime_recovery_async()
    release_first_attempt.set()

    assert second_attempt_finished.wait(1)
    worker.join(timeout=1)
    assert worker.is_alive() is False
    assert len(attempts) == 2
    assert backend_core_instance._runtime_recovery_thread is None


def test_close_rejects_late_lifecycle_and_recovery_triggers(
        backend_core_instance, monkeypatch,
):
    import backend_core as backend_core_module

    backend_core_instance.close()
    monkeypatch.setattr(
        backend_core_module.threading,
        "Thread",
        lambda **_kwargs: pytest.fail("closed backend must not start a worker"),
    )

    backend_core_instance._start_runtime_recovery_async()

    assert backend_core_instance._runtime_recovery_thread is None
    assert backend_core_instance._runtime_recovery_stop_event.is_set() is True
    assert backend_core_instance.parse_and_apply_templates(
        {}, [], install_id=INSTALL_ID,
    ) == (False, "Backend lifecycle is closed")
    assert backend_core_instance.parse_and_delete_templates() == (
        False, "Backend lifecycle is closed",
    )


@pytest.mark.parametrize("phase", ["uninstalling", "finalizing-uninstall"])
def test_backend_startup_recovery_resumes_interrupted_uninstall_in_background(
        backend_core_instance, monkeypatch, phase,
):
    backend_core_instance.runtime_orchestrator.session = SimpleNamespace(
        phase=phase,
        policy_id="policy-a",
        install_id="install-a",
        last_error="",
    )

    started = []
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: started.append(install_id),
    )

    backend_core_instance._recover_runtime_session()

    assert started == ["install-a"]
    assert backend_core_instance.runtime_orchestrator.uninstall_calls == 0
    assert backend_core_instance.runtime_telemetry.started == 0
    assert backend_core_instance._query_admission_enabled is False


def test_uninstall_quickly_accepts_and_starts_background_cleanup(
        backend_core_instance, monkeypatch,
):
    backend_core_instance._bind_runtime_urls(_directory())
    stop_event = threading.Event()
    backend_core_instance._runtime_reconcile_stop_event = stop_event
    backend_core_instance._runtime_reconcile_thread = object()

    started = []
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: started.append(install_id),
    )

    started_at = time.monotonic()
    result = backend_core_instance.parse_and_delete_templates()
    elapsed = time.monotonic() - started_at

    assert result == (True, "Uninstall services started")
    assert elapsed < 0.5
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 1
    assert backend_core_instance.runtime_orchestrator.uninstall_calls == 0
    assert backend_core_instance.runtime_orchestrator.session.phase == "uninstalling"
    assert started == ["install-a"]
    assert stop_event.is_set() is True
    assert backend_core_instance._runtime_reconcile_stop_event is None
    assert backend_core_instance._runtime_reconcile_thread is None
    assert backend_core_instance.resource_url is None
    assert backend_core_instance.result_url is None
    assert backend_core_instance.result_file_url is None
    assert backend_core_instance.log_fetch_url is None
    assert backend_core_instance.runtime_telemetry.unbound == 1
    assert backend_core_instance._query_admission_enabled is False


def test_uninstall_intent_failure_is_reported_and_stops_runtime_worker(backend_core_instance):
    backend_core_instance._bind_runtime_urls(_directory())
    backend_core_instance.runtime_orchestrator.begin_uninstall_error = RuntimeError(
        "stop intent failed"
    )
    stop_event = threading.Event()
    backend_core_instance._runtime_reconcile_stop_event = stop_event

    ok, message = backend_core_instance.parse_and_delete_templates()

    assert ok is False
    assert "stop intent failed" in message
    assert stop_event.is_set() is True
    assert backend_core_instance._runtime_reconcile_stop_event is None
    assert backend_core_instance.runtime_telemetry.unbound == 1
    assert backend_core_instance.runtime_telemetry.bound_urls == [
        backend_core_instance.resource_url
    ]
    assert backend_core_instance.runtime_telemetry.started == 0


def test_runtime_reconcile_worker_completes_an_accepted_uninstall(
        backend_core_instance,
):
    backend_core_instance.runtime_orchestrator.session.phase = "uninstalling"

    class OneTickEvent:
        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            return False

        @staticmethod
        def is_set():
            return False

    stop_event = OneTickEvent()
    backend_core_instance._runtime_reconcile_stop_event = stop_event
    backend_core_instance._runtime_reconcile_thread = object()

    backend_core_instance.run_runtime_reconcile(stop_event, "install-a")

    assert stop_event.waits == 1
    assert backend_core_instance.runtime_orchestrator.uninstall_calls == 1
    assert backend_core_instance.runtime_orchestrator.current_session() is None
    assert backend_core_instance._runtime_reconcile_stop_event is None
    assert backend_core_instance._runtime_reconcile_thread is None


def test_backend_close_cancels_query_runtime_worker_and_telemetry(backend_core_instance):
    reconcile_stop = threading.Event()
    query_stop = threading.Event()
    install_stop = threading.Event()
    backend_core_instance._runtime_reconcile_stop_event = reconcile_stop
    backend_core_instance._runtime_reconcile_thread = object()
    backend_core_instance._query_admission_enabled = True
    backend_core_instance.source_open = True
    backend_core_instance.is_get_result = True
    backend_core_instance._query_cancel_event = query_stop
    backend_core_instance._install_admission = SimpleNamespace(
        install_id=INSTALL_ID,
        cancel_event=install_stop,
        cancel=install_stop.set,
    )

    backend_core_instance.close()

    assert reconcile_stop.is_set() is True
    assert query_stop.is_set() is True
    assert install_stop.is_set() is True
    assert backend_core_instance._query_admission_enabled is False
    assert backend_core_instance.source_open is False
    assert backend_core_instance.runtime_telemetry.closed == 1


def test_redeploy_requires_managed_session_and_current_policy(backend_core_instance):
    backend_core_instance.runtime_orchestrator.session = None
    assert backend_core_instance.parse_and_redeploy_services() == (
        False, "no managed runtime session exists"
    )

    backend_core_instance.runtime_orchestrator.session = SimpleNamespace(
        phase="active", policy_id="missing", install_id="install-a"
    )
    backend_core_instance.schedulers = []
    assert backend_core_instance.parse_and_redeploy_services() == (
        False, "scheduler policy 'missing' does not exist"
    )


def test_redeploy_uses_session_policy_and_rebinds_only_after_commit(backend_core_instance):
    policy = {"id": "policy-a"}
    backend_core_instance.schedulers = [policy]
    backend_core_instance.runtime_orchestrator.directory = _directory(revision=2)

    assert backend_core_instance.parse_and_redeploy_services() == (
        True, "Redeployment succeeded"
    )
    assert backend_core_instance.runtime_orchestrator.redeploy_calls == [policy]
    assert "-r2.dayu.svc.cluster.local.:9001" in backend_core_instance.resource_url

    backend_core_instance.runtime_orchestrator.changed = False
    backend_core_instance.resource_url = "keep-current-binding"
    assert backend_core_instance.parse_and_redeploy_services(policy) == (
        True, "Deployment is unchanged"
    )
    assert backend_core_instance.resource_url == "keep-current-binding"


def test_redeploy_failure_does_not_replace_active_urls(backend_core_instance):
    backend_core_instance.resource_url = "active-url"
    backend_core_instance.runtime_orchestrator.redeploy_error = RuntimeError("proposal conflict")

    ok, message = backend_core_instance.parse_and_redeploy_services({"id": "policy-a"})

    assert ok is False
    assert "proposal conflict" in message
    assert backend_core_instance.resource_url == "active-url"


def test_redeploy_pending_retirement_is_a_successful_deferred_request(
        backend_core_instance,
):
    backend_core_instance.resource_url = "active-url"
    backend_core_instance.runtime_orchestrator.redeploy_error = (
        RuntimeRetirementPending("revision 1 is still retiring")
    )

    result = backend_core_instance.parse_and_redeploy_services({"id": "policy-a"})

    assert result == (
        False,
        "Redeployment deferred while the previous revision retires",
    )
    assert backend_core_instance.resource_url == "active-url"


def test_runtime_url_binding_rejects_missing_or_ambiguous_infrastructure_route(
        backend_core_instance,
):
    processor_slot = RuntimeSlot(
        "processor", "edge-a", "edge", logical_service="face-detection"
    )
    processor = RuntimeUnit(
        processor_slot,
        processor_slot.runtime_name(1, "install-a"),
        1,
        "b" * 64,
    )
    bad_directory = RuntimeDirectory("install-a", 1, (processor,))

    with pytest.raises(RuntimeError, match="scheduler"):
        backend_core_instance._bind_runtime_urls(bad_directory)


def test_reinstall_invalidates_old_sleeping_runtime_worker(
        backend_core_instance, monkeypatch,
):
    import backend_core as backend_core_module

    workers = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            workers.append(self)

        def start(self):
            return None

    monkeypatch.setattr(backend_core_module.threading, "Thread", FakeThread)
    backend_core_instance.processor_redeployment_interval_s = 1
    backend_core_instance.runtime_orchestrator.session = None

    assert backend_core_instance.parse_and_apply_templates({}, [], install_id=INSTALL_ID)[0] is True
    old_event, old_install_id = workers[0].kwargs["args"]
    assert old_event.is_set() is False

    backend_core_instance.runtime_orchestrator.session = None
    assert backend_core_instance.parse_and_apply_templates(
        {}, [], install_id=REPLACEMENT_INSTALL_ID,
    )[0] is True
    assert old_event.is_set() is True
    assert backend_core_instance._runtime_reconcile_stop_event is workers[1].kwargs["args"][0]

    redeploy_count = len(backend_core_instance.runtime_orchestrator.redeploy_calls)
    workers[0].kwargs["target"](old_event, old_install_id)
    assert len(backend_core_instance.runtime_orchestrator.redeploy_calls) == redeploy_count


def test_runtime_reconcile_worker_is_interruptible_before_first_tick(backend_core_instance):
    event = threading.Event()
    event.set()
    backend_core_instance.processor_redeployment_interval_s = 5.0

    backend_core_instance.run_runtime_reconcile(event, "install-a")

    assert backend_core_instance.runtime_orchestrator.reconcile_calls == []
    assert backend_core_instance.runtime_orchestrator.redeploy_calls == []


def test_runtime_worker_reconciles_each_tick_but_rolls_out_only_when_due(
        backend_core_instance, monkeypatch,
):
    class TwoTickEvent:
        def __init__(self):
            self.wait_calls = 0
            self.stopped = False

        def wait(self, timeout):
            assert timeout == 1.0
            self.wait_calls += 1
            return self.stopped or self.wait_calls > 2

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

    import backend_core as backend_core_module

    monotonic_values = iter((100.0, 104.0, 105.0, 105.0))
    monkeypatch.setattr(
        backend_core_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )
    event = TwoTickEvent()
    backend_core_instance._runtime_reconcile_stop_event = event
    backend_core_instance.processor_redeployment_interval_s = 5.0
    policy = {"id": "policy-a"}
    backend_core_instance.schedulers = [policy]
    rollout_calls = []

    def record_rollout(selected_policy, cancel_event=None):
        rollout_calls.append((selected_policy, cancel_event))
        return True, "ok"

    backend_core_instance.parse_and_redeploy_services = record_rollout

    backend_core_instance.run_runtime_reconcile(event, "install-a")

    assert backend_core_instance.runtime_orchestrator.reconcile_calls == [event, event]
    assert rollout_calls == [(policy, event)]


def test_runtime_reconcile_does_not_hold_worker_lock_across_control_plane_io(
        backend_core_instance, monkeypatch,
):
    class OneCycleEvent:
        def __init__(self):
            self.stopped = False

        def wait(self, timeout):
            if self.stopped:
                return True
            return False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

    event = OneCycleEvent()
    backend_core_instance._runtime_reconcile_stop_event = event
    backend_core_instance.processor_redeployment_interval_s = 1.0
    lock_available = []

    import backend_core as backend_core_module

    monotonic_values = iter((0.0, 1.0, 2.0))
    monkeypatch.setattr(
        backend_core_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    def redeploy(_policy, cancel_event=None):
        assert cancel_event is event
        acquired = backend_core_instance._runtime_reconcile_lock.acquire(blocking=False)
        lock_available.append(acquired)
        if acquired:
            backend_core_instance._runtime_reconcile_lock.release()
        event.set()
        return True, "ok"

    backend_core_instance.parse_and_redeploy_services = redeploy

    backend_core_instance.run_runtime_reconcile(event, "install-a")

    assert backend_core_instance.runtime_orchestrator.reconcile_calls == [event]
    assert lock_available == [True]


def test_runtime_reconcile_retries_failed_local_projection_with_backoff(
        backend_core_instance, monkeypatch,
):
    class ThreeWaitEvent:
        def __init__(self):
            self.waits = []
            self.stopped = False

        def wait(self, timeout):
            self.waits.append(timeout)
            return self.stopped or len(self.waits) >= 3

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

    event = ThreeWaitEvent()
    backend_core_instance._runtime_reconcile_stop_event = event
    backend_core_instance.processor_redeployment_interval_s = 0
    backend_core_instance._bound_runtime_key = None
    bind_runtime_urls = backend_core_instance._bind_runtime_urls
    bind_attempts = []

    def fail_once(directory):
        bind_attempts.append(directory.revision)
        if len(bind_attempts) == 1:
            raise RuntimeError("local bind failed once")
        bind_runtime_urls(directory)

    monkeypatch.setattr(
        backend_core_instance,
        "_bind_runtime_urls",
        fail_once,
    )

    backend_core_instance.run_runtime_reconcile(event, "install-a")

    assert event.waits == [1.0, 2.0, 1.0]
    assert bind_attempts == [1, 1]
    assert backend_core_instance._bound_runtime_key == ("install-a", 1)
    assert backend_core_instance._local_runtime_error == ""
    assert backend_core_instance.runtime_orchestrator.install_calls == []


def test_runtime_reconcile_backs_off_deferred_exact_uid_cleanup_failures(
        backend_core_instance,
):
    class FourWaitEvent:
        def __init__(self):
            self.waits = []

        def wait(self, timeout):
            self.waits.append(timeout)
            return len(self.waits) >= 4

        @staticmethod
        def is_set():
            return False

    event = FourWaitEvent()
    session = backend_core_instance.runtime_orchestrator.session
    session.cleanup = (SimpleNamespace(runtime_id="cleanup-a"),)
    session.last_error = "transient Kubernetes delete failure"
    backend_core_instance._bound_runtime_key = ("install-a", 1)
    backend_core_instance._runtime_reconcile_stop_event = event
    backend_core_instance.processor_redeployment_interval_s = 0

    backend_core_instance.run_runtime_reconcile(event, "install-a")

    assert event.waits == [1.0, 2.0, 4.0, 8.0]
    assert backend_core_instance.runtime_orchestrator.reconcile_calls == [
        event, event, event,
    ]
