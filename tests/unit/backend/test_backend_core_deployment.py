import threading
from types import SimpleNamespace

import pytest

from runtime_model import RuntimeDirectory, RuntimeEndpoint, RuntimeSlot, RuntimeUnit


def _unit(component, port, revision=1):
    slot = RuntimeSlot(component, "cloud-a", "cloud")
    runtime_id = slot.runtime_name(revision)
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
        self.uninstall_calls = 0
        self.redeploy_calls = []
        self.install_error = None
        self.uninstall_error = None
        self.redeploy_error = None
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

    def current_session(self):
        return self.session

    def redeploy(self, policy):
        self.redeploy_calls.append(policy)
        if self.redeploy_error:
            raise self.redeploy_error
        return self.changed

    def recover(self):
        return self.session

    def active_directory(self):
        return self.directory


@pytest.fixture
def backend_core_instance(mounted_runtime):
    from backend_core import BackendCore

    instance = BackendCore()
    instance.runtime_orchestrator = FakeOrchestrator()
    return instance


def test_install_delegates_transaction_and_binds_directory_urls(backend_core_instance, monkeypatch):
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
    assert backend_core_instance.runtime_orchestrator.install_calls == [{
        "policy": policy,
        "source_deploy": source_deploy,
        "source_label": "source-a",
    }]
    assert backend_core_instance.resource_url.endswith(":9001/resource")
    assert backend_core_instance.result_url.endswith(":9003/result")
    assert started[0]["name"] == "dayu-runtime-redeployment-install-a"
    assert started[0]["daemon"] is True
    assert started[0]["started"] is True
    assert started[0]["args"][1] == "install-a"


def test_install_failure_is_reported_without_starting_rollout_loop(backend_core_instance, monkeypatch):
    backend_core_instance.runtime_orchestrator.install_error = RuntimeError("activation failed")
    import backend_core as backend_core_module

    monkeypatch.setattr(
        backend_core_module.threading,
        "Thread",
        lambda **kwargs: pytest.fail("rollout loop must not start after failed install"),
    )

    ok, message = backend_core_instance.parse_and_apply_templates({}, [])

    assert ok is False
    assert "activation failed" in message
    assert backend_core_instance._redeployment_stop_event is None


def test_backend_startup_recovery_rebinds_active_directory_and_restarts_rollout_loop(
    backend_core_instance, monkeypatch
):
    started = []
    monkeypatch.setattr(
        backend_core_instance,
        "_start_redeployment_loop",
        lambda install_id: started.append(install_id),
    )

    backend_core_instance._recover_runtime_session()

    assert backend_core_instance.resource_url.endswith(":9001/resource")
    assert backend_core_instance.result_url.endswith(":9003/result")
    assert started == ["install-a"]


@pytest.mark.parametrize("phase", ["clearing-directory", "finalizing-uninstall"])
def test_backend_startup_recovery_finishes_interrupted_uninstall(backend_core_instance, phase):
    backend_core_instance.runtime_orchestrator.session = SimpleNamespace(
        phase=phase,
        policy_id="policy-a",
        install_id="install-a",
        last_error="",
    )

    backend_core_instance._recover_runtime_session()

    assert backend_core_instance.runtime_orchestrator.uninstall_calls == 1


def test_uninstall_delegates_drain_and_clears_only_runtime_bindings(backend_core_instance):
    backend_core_instance._bind_runtime_urls(_directory())
    stop_event = threading.Event()
    backend_core_instance._redeployment_stop_event = stop_event
    backend_core_instance._redeployment_thread = object()

    result = backend_core_instance.parse_and_delete_templates()

    assert result == (True, "Uninstall services successfully")
    assert backend_core_instance.runtime_orchestrator.uninstall_calls == 1
    assert stop_event.is_set() is True
    assert backend_core_instance._redeployment_stop_event is None
    assert backend_core_instance._redeployment_thread is None
    assert backend_core_instance.resource_url is None
    assert backend_core_instance.result_url is None
    assert backend_core_instance.result_file_url is None
    assert backend_core_instance.log_fetch_url is None


def test_uninstall_failure_preserves_error_and_stops_redeployment(backend_core_instance):
    backend_core_instance.runtime_orchestrator.uninstall_error = RuntimeError("leases still active")
    stop_event = threading.Event()
    backend_core_instance._redeployment_stop_event = stop_event

    ok, message = backend_core_instance.parse_and_delete_templates()

    assert ok is False
    assert "leases still active" in message
    assert stop_event.is_set() is True
    assert backend_core_instance._redeployment_stop_event is None


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
    assert "-r2.dayu.svc.cluster.local:9001" in backend_core_instance.resource_url

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


def test_runtime_url_binding_rejects_missing_or_ambiguous_infrastructure_route(
        backend_core_instance,
):
    processor_slot = RuntimeSlot(
        "processor", "edge-a", "edge", logical_service="face-detection"
    )
    processor = RuntimeUnit(
        processor_slot,
        processor_slot.runtime_name(1),
        1,
        "b" * 64,
    )
    bad_directory = RuntimeDirectory("install-a", 1, (processor,))

    with pytest.raises(RuntimeError, match="scheduler"):
        backend_core_instance._bind_runtime_urls(bad_directory)


def test_reinstall_invalidates_old_sleeping_redeployment_worker(
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
    assert backend_core_instance._redeployment_stop_event is workers[1].kwargs["args"][0]

    redeploy_count = len(backend_core_instance.runtime_orchestrator.redeploy_calls)
    workers[0].kwargs["target"](old_event, old_install_id)
    assert len(backend_core_instance.runtime_orchestrator.redeploy_calls) == redeploy_count


def test_redeployment_wait_is_event_interruptible(backend_core_instance, monkeypatch):
    event = threading.Event()
    waits = []
    monkeypatch.setattr(event, "wait", lambda timeout: waits.append(timeout) or True)
    monkeypatch.setattr("backend_core.time.monotonic", lambda: 12.0)
    backend_core_instance.processor_redeployment_interval_s = 5.0

    assert backend_core_instance._wait_until_next_redeployment_cycle(event, 10.0) is True
    assert waits == [3.0]
