import importlib
import json

import pytest

from core.lib.runtime import RuntimeContext, RuntimeResolver


def route(component, node, port, service=""):
    slot = {"component": component, "target_node": node}
    if service:
        slot["logical_service"] = service
    return {
        "slot": slot,
        "runtime_id": f"{component}-{service or 'runtime'}-{node}",
        "runtime_revision": 3,
        "endpoint": {
            "dns_name": f"{component}-{service or 'runtime'}-{node}.dayu.svc",
            "port": port,
            "runtime_service_uid": f"rs-{component}-{node}",
            "service_uid": f"svc-{component}-{node}",
            "pod_uid": f"pod-{component}-{node}",
        },
    }


class FakeTask:
    def __init__(self, service="face-detection", node="edgex1", routes=None, file_path="payload.bin"):
        self.service = service
        self.node = node
        self.routes = routes or []
        self.file_path = file_path
        self.transmit_durations = []
        self.execute_durations = []

    def get_source_id(self): return 1
    def get_task_id(self): return 2
    def get_flow_index(self): return self.service
    def get_current_service_info(self): return self.service, {}
    def get_current_stage_device(self): return self.node
    def set_current_stage_device(self, node): self.node = node
    def get_file_path(self): return self.file_path
    def get_root_uuid(self): return "root-task"
    def get_runtime_directory_revision(self): return 1
    def get_task_uuid(self): return "branch-task"
    def get_runtime_routes(self): return self.routes
    def serialize(self): return '{"task":"serialized"}'
    def save_transmit_time(self, value): self.transmit_durations.append(value)
    def save_execute_time(self, value): self.execute_durations.append(value)


@pytest.fixture
def controller_under_test():
    module = importlib.import_module("core.controller.controller")
    controller = object.__new__(module.Controller)
    controller.task_coordinator = None
    controller.is_display = False
    controller.local_device = "edgex1"
    controller.cloud_device = "cloudx1"
    controller.distribute_address = "http://distributor.dayu.svc:9003/distribute"
    controller.runtime_context = RuntimeContext({"local_node": "edgex1", "cloud_node": "cloudx1"})
    controller.runtime_resolver = RuntimeResolver(controller.runtime_context)
    return module, controller


@pytest.mark.unit
def test_controller_health_and_queue_clear_use_explicit_processor_routes(controller_under_test, monkeypatch):
    module, controller = controller_under_test
    routes = [route("processor", "edgex1", 31000, "face"), route("processor", "edgex1", 32000, "gender")]
    responses = iter([{"status": "ok"}, {"status": "ok"}])
    monkeypatch.setattr(module, "http_request", lambda **kwargs: next(responses))
    assert controller.check_processor_health({"runtime_routes": routes}) is True
    assert controller.check_processor_health({}) is False

    calls = []
    monkeypatch.setattr(module, "http_request", lambda **kwargs: calls.append(kwargs) or {
        "ok": True, "cleared_count": 2, "matched_count": 3, "remaining_count": 1,
    })
    result = controller.clear_processor_queues({
        "runtime_routes": routes, "services": "face", "timeout_s": "bad", "dry_run": True,
    })
    assert result["ok"] is True and result["service_count"] == 1
    assert calls[0]["url"] == "http://processor-face-edgex1.dayu.svc:31000/queue_clear"
    assert json.loads(calls[0]["data"]["data"])["dry_run"] is True
    assert controller.clear_processor_queues({})["ok"] is False


@pytest.mark.unit
def test_controller_processor_delivery_requires_exact_task_route(controller_under_test, monkeypatch, tmp_path):
    module, controller = controller_under_test
    task = FakeTask(routes=[route("processor", "edgex1", 31000, "face-detection")])
    monkeypatch.setattr(module.Controller, "record_execute_ts", staticmethod(lambda *args, **kwargs: None))
    payload = tmp_path / "payload.bin"
    monkeypatch.setattr(module.FileOps, "get_task_file_in_temp", staticmethod(lambda current: str(payload)))
    assert controller.send_task_to_service(task, "face-detection") is False

    payload.write_bytes(b"payload")
    calls = []
    monkeypatch.setattr(module, "deliver_task", lambda **kwargs: calls.append(kwargs) or True)
    assert controller.send_task_to_service(task, "face-detection") is True
    assert calls[0]["url"] == "http://processor-face-detection-edgex1.dayu.svc:31000/predict_local"

    missing = FakeTask(routes=[route("processor", "cloudx1", 31000, "face-detection")])
    assert controller.send_task_to_service(missing, "face-detection") is False
    assert missing.get_current_stage_device() == "edgex1"


@pytest.mark.unit
def test_controller_remote_delivery_uses_task_controller_identity(controller_under_test, monkeypatch, tmp_path):
    module, controller = controller_under_test
    task = FakeTask(node="cloudx1", routes=[route("controller", "cloudx1", 9002)])
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    monkeypatch.setattr(module.FileOps, "get_task_file_in_temp", staticmethod(lambda current: str(payload)))
    monkeypatch.setattr(module.Controller, "record_transmit_ts", staticmethod(lambda *args, **kwargs: None))
    calls = []
    monkeypatch.setattr(module, "deliver_task", lambda **kwargs: calls.append(kwargs) or True)
    assert controller.send_task_to_other_device(task, "cloudx1") is True
    assert calls[0]["url"] == "http://controller-runtime-cloudx1.dayu.svc:9002/submit_task"


@pytest.mark.unit
def test_controller_timestamp_helpers_preserve_behavior(controller_under_test, monkeypatch):
    module, _ = controller_under_test
    task = FakeTask()
    monkeypatch.setattr(module.TimeEstimator, "record_dag_ts", staticmethod(
        lambda task, is_end=False, sub_tag=None: 1.25 if sub_tag == "transmit" else 0.5
    ))
    module.Controller.record_transmit_ts(task, is_end=True)
    module.Controller.record_execute_ts(task, is_end=True)
    assert task.transmit_durations == [1.25]
    assert task.execute_durations == [0.5]
