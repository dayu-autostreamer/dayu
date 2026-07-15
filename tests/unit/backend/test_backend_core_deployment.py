import threading
import time
from types import SimpleNamespace

import pytest

from runtime_model import RuntimeDirectory, RuntimeEndpoint, RuntimeSlot, RuntimeUnit
from runtime_orchestrator import RuntimeOperationCancelled, RuntimeRetirementPending


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
            phase="active", policy_id="policy-a", install_id="install-a"
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
        return self.directory

    def uninstall(self):
        self.uninstall_calls += 1
        if self.uninstall_error:
            raise self.uninstall_error
        self.session = None

    def begin_uninstall(self):
        self.begin_uninstall_calls += 1
        if self.begin_uninstall_error:
            raise self.begin_uninstall_error
        if self.session is None:
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
        policy, source_deploy, source_label="source-a"
    )

    assert (ok, message) == (True, "Install services successfully")
    assert len(backend_core_instance.runtime_orchestrator.install_calls) == 1
    install_call = backend_core_instance.runtime_orchestrator.install_calls[0]
    cancel_event = install_call.pop("cancel_event")
    assert install_call == {
        "policy": policy,
        "source_deploy": source_deploy,
        "source_label": "source-a",
    }
    assert isinstance(cancel_event, threading.Event)
    assert cancel_event.is_set() is False
    assert backend_core_instance.resource_url.endswith(":9001/resource")
    assert backend_core_instance.result_url.endswith(":9003/result")
    assert backend_core_instance.runtime_telemetry.bound_urls == [
        backend_core_instance.resource_url
    ]
    assert backend_core_instance.runtime_telemetry.bound_directories == [_directory()]
    assert backend_core_instance.runtime_telemetry.started == 1
    assert started[0]["name"] == "dayu-runtime-reconcile-install-a"
    assert started[0]["daemon"] is True
    assert started[0]["started"] is True
    assert started[0]["args"][1] == "install-a"
    assert backend_core_instance._query_admission_enabled is True


def test_install_failure_is_reported_without_starting_runtime_worker(
        backend_core_instance, monkeypatch,
):
    backend_core_instance.runtime_orchestrator.install_error = RuntimeError("activation failed")
    import backend_core as backend_core_module

    monkeypatch.setattr(
        backend_core_module.threading,
        "Thread",
        lambda **kwargs: pytest.fail("runtime worker must not start after failed install"),
    )

    ok, message = backend_core_instance.parse_and_apply_templates({}, [])

    assert ok is False
    assert "activation failed" in message
    assert backend_core_instance._runtime_reconcile_stop_event is None


def test_uninstall_cancels_inflight_install_before_persisting_stop_intent(
        backend_core_instance, monkeypatch,
):
    operation_lock = threading.Lock()
    install_started = threading.Event()
    install_results = []
    seen_cancel_events = []

    def blocking_install(**kwargs):
        cancel_event = kwargs["cancel_event"]
        seen_cancel_events.append(cancel_event)
        with operation_lock:
            install_started.set()
            assert cancel_event.wait(2)
            raise RuntimeOperationCancelled("cancelled by lifecycle operation")

    def serialized_begin_uninstall():
        with operation_lock:
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
            backend_core_instance.parse_and_apply_templates({}, []),
        ),
    )
    install_thread.start()
    assert install_started.wait(1)

    started_at = time.monotonic()
    uninstall_result = backend_core_instance.parse_and_delete_templates()
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
    assert started_workers == ["install-a"]
    assert backend_core_instance._install_cancel_event is None
    assert backend_core_instance._stop_request_count == 0


def test_stop_registration_prevents_install_token_race(
        backend_core_instance, monkeypatch,
):
    uninstall_started = threading.Event()
    release_uninstall = threading.Event()
    stop_results = []

    def blocking_begin_uninstall():
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

    assert backend_core_instance.parse_and_apply_templates({}, []) == (
        False,
        "Install cancelled by lifecycle operation",
    )
    assert backend_core_instance.runtime_orchestrator.install_calls == []

    release_uninstall.set()
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
    assert backend_core_instance._stop_request_count == 0


def test_overlapping_stop_requests_keep_install_admission_closed_until_both_finish(
        backend_core_instance, monkeypatch,
):
    uninstall_started = threading.Event()
    release_uninstalls = threading.Event()
    results = []

    def blocking_begin_uninstall():
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

    assert backend_core_instance._stop_request_count == 1
    assert backend_core_instance.parse_and_apply_templates({}, []) == (
        False,
        "Install cancelled by lifecycle operation",
    )
    started_at = time.monotonic()
    assert backend_core_instance.parse_and_delete_templates() == (
        True,
        "Uninstall services started",
    )
    assert time.monotonic() - started_at < 0.5
    assert backend_core_instance.runtime_orchestrator.begin_uninstall_calls == 1

    release_uninstalls.set()
    thread.join(timeout=1)
    assert thread.is_alive() is False

    assert results == [(True, "Uninstall services started")]
    assert backend_core_instance._stop_request_count == 0


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
    assert backend_core_instance._stop_request_count == 0


@pytest.mark.parametrize("phase", ["uninstalling", "finalizing-uninstall"])
def test_install_is_rejected_from_cached_uninstall_state_without_entering_transaction(
        backend_core_instance, phase,
):
    backend_core_instance.runtime_orchestrator.session.phase = phase

    started_at = time.monotonic()
    result = backend_core_instance.parse_and_apply_templates({}, [])

    assert result == (False, "Uninstall is in progress")
    assert time.monotonic() - started_at < 0.5
    assert backend_core_instance.runtime_orchestrator.install_calls == []
    assert backend_core_instance._install_cancel_event is None


def test_concurrent_second_install_cannot_overwrite_cancel_token(
        backend_core_instance, monkeypatch,
):
    first_started = threading.Event()
    release_first = threading.Event()
    first_result = []
    tokens = []

    def blocking_install(**kwargs):
        tokens.append(kwargs["cancel_event"])
        first_started.set()
        assert release_first.wait(2)
        return _directory()

    backend_core_instance.runtime_orchestrator.install = blocking_install
    monkeypatch.setattr(
        backend_core_instance,
        "_start_runtime_reconcile_loop",
        lambda install_id: None,
    )
    first_thread = threading.Thread(
        target=lambda: first_result.append(
            backend_core_instance.parse_and_apply_templates({}, []),
        ),
    )
    first_thread.start()
    assert first_started.wait(1)

    assert backend_core_instance.parse_and_apply_templates({}, []) == (
        False,
        "Another install operation is already in progress",
    )
    assert backend_core_instance._install_cancel_event is tokens[0]

    release_first.set()
    first_thread.join(timeout=1)
    assert first_thread.is_alive() is False
    assert first_result == [(True, "Install services successfully")]
    assert len(tokens) == 1
    assert backend_core_instance._install_cancel_event is None


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
    backend_core_instance._install_cancel_event = install_stop

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

    assert backend_core_instance.parse_and_apply_templates({}, [])[0] is True
    old_event, old_install_id = workers[0].kwargs["args"]
    assert old_event.is_set() is False

    assert backend_core_instance.parse_and_apply_templates({}, [])[0] is True
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
