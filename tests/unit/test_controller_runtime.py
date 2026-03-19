import importlib
from pathlib import Path

import pytest


class FakeTask:
    def __init__(self, service_name="face-detection", stage_device="edgex1", file_path="payload.bin"):
        self.service_name = service_name
        self.stage_device = stage_device
        self.file_path = file_path
        self.serialized = {"task": "serialized"}
        self.transmit_durations = []
        self.execute_durations = []

    def get_source_id(self):
        return 1

    def get_task_id(self):
        return 2

    def get_flow_index(self):
        return self.service_name

    def get_current_service_info(self):
        return self.service_name, {}

    def get_current_stage_device(self):
        return self.stage_device

    def set_current_stage_device(self, device):
        self.stage_device = device

    def get_file_path(self):
        return self.file_path

    def serialize(self):
        return self.serialized

    def save_transmit_time(self, duration):
        self.transmit_durations.append(duration)

    def save_execute_time(self, duration):
        self.execute_durations.append(duration)


@pytest.fixture
def controller_under_test(monkeypatch):
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = None
    controller.is_display = False
    controller.controller_port = 30001
    controller.distributor_port = 30002
    controller.distributor_hostname = "cloudx1"
    controller.distribute_address = "http://10.0.0.9:30002/distribute"
    controller.local_device = "edgex1"
    controller.cloud_device = "cloudx1"

    monkeypatch.setattr(
        controller_module.NodeInfo,
        "hostname2ip",
        staticmethod(lambda hostname: {"edgex1": "10.0.0.1", "cloudx1": "10.0.0.9"}.get(hostname, "10.0.0.5")),
    )
    monkeypatch.setattr(
        controller_module,
        "merge_address",
        lambda ip, port=None, path=None, protocol="http": f"{protocol}://{ip}:{port}/{path.strip('/')}",
    )
    return controller_module, controller


@pytest.mark.unit
def test_controller_health_check_reports_success_and_failure(controller_under_test, monkeypatch):
    controller_module, controller = controller_under_test
    monkeypatch.setattr(controller_module.PortInfo, "force_refresh", staticmethod(lambda: None))
    monkeypatch.setattr(
        controller_module.PortInfo,
        "get_service_ports_dict",
        staticmethod(lambda device: {"svc-a": 31000, "svc-b": 32000}),
    )

    responses = iter([{"status": "ok"}, {"status": "ok"}])
    monkeypatch.setattr(controller_module, "http_request", lambda **kwargs: next(responses))
    assert controller.check_processor_health() is True

    responses = iter([{"status": "ok"}, {"status": "fail"}])
    monkeypatch.setattr(controller_module, "http_request", lambda **kwargs: next(responses))
    assert controller.check_processor_health() is False


@pytest.mark.unit
def test_send_task_to_service_executes_locally_when_processor_and_file_exist(controller_under_test, monkeypatch, tmp_path):
    controller_module, controller = controller_under_test
    task = FakeTask()
    temp_file = tmp_path / task.get_file_path()
    temp_file.write_bytes(b"payload")

    execute_record = []
    request_calls = []

    monkeypatch.setattr(
        controller_module.Context,
        "get_temporary_file_path",
        staticmethod(lambda file_path: str(tmp_path / Path(file_path).name)),
    )
    monkeypatch.setattr(
        controller_module.PortInfo,
        "get_service_ports_dict",
        staticmethod(lambda device: {"face-detection": 31000}),
    )
    monkeypatch.setattr(controller_module, "http_request", lambda **kwargs: request_calls.append(kwargs) or {"state": "ok"})
    monkeypatch.setattr(
        controller_module.Controller,
        "record_execute_ts",
        staticmethod(lambda cur_task, is_end=False: execute_record.append((cur_task.get_task_id(), is_end))),
    )

    assert controller.send_task_to_service(task, "face-detection") == "execute"
    assert execute_record == [(2, False)]
    assert request_calls[0]["method"] == controller_module.NetworkAPIMethod.PROCESSOR_PROCESS_LOCAL
    assert request_calls[0]["data"] == {"data": task.serialize()}


@pytest.mark.unit
def test_send_task_to_service_transmits_to_cloud_when_service_only_exists_remotely(controller_under_test, monkeypatch):
    controller_module, controller = controller_under_test
    task = FakeTask()
    submit_calls = []
    erased = []
    force_refresh_calls = []

    monkeypatch.setattr(
        controller_module.PortInfo,
        "get_service_ports_dict",
        staticmethod(lambda device: {}),
    )
    monkeypatch.setattr(controller_module.PortInfo, "force_refresh", staticmethod(lambda: force_refresh_calls.append(True)))
    monkeypatch.setattr(
        controller_module.KubeConfig,
        "get_service_nodes_dict",
        staticmethod(lambda: {"face-detection": ["cloudx1"]}),
    )
    monkeypatch.setattr(controller, "submit_task", lambda cur_task: submit_calls.append(cur_task) or "transmit")
    monkeypatch.setattr(
        controller_module.Controller,
        "erase_execute_ts",
        staticmethod(lambda cur_task: erased.append(("execute", cur_task.get_task_id()))),
    )
    monkeypatch.setattr(
        controller_module.Controller,
        "erase_transmit_ts",
        staticmethod(lambda cur_task: erased.append(("transmit", cur_task.get_task_id()))),
    )

    assert controller.send_task_to_service(task, "face-detection") == "transmit"
    assert force_refresh_calls == [True]
    assert task.get_current_stage_device() == "cloudx1"
    assert submit_calls == [task]
    assert erased == [("execute", 2), ("transmit", 2)]


@pytest.mark.unit
def test_submit_task_routes_end_remote_and_missing_task_branches(controller_under_test, monkeypatch):
    controller_module, controller = controller_under_test
    events = []

    end_task = FakeTask(service_name=controller_module.TaskConstant.END.value)
    remote_task = FakeTask(stage_device="cloudx1")
    local_task = FakeTask()

    monkeypatch.setattr(controller, "send_task_to_distributor", lambda cur_task: events.append(("distribute", cur_task)))
    monkeypatch.setattr(controller, "send_task_to_other_device", lambda cur_task, device: events.append(("remote", device)))
    monkeypatch.setattr(controller, "send_task_to_service", lambda cur_task, service: events.append(("service", service)) or "execute")

    assert controller.submit_task(None) == "error"
    assert controller.submit_task(end_task) == "transmit"
    assert controller.submit_task(remote_task) == "transmit"
    assert controller.submit_task(local_task) == "execute"
    assert events == [
        ("distribute", end_task),
        ("remote", "cloudx1"),
        ("service", "face-detection"),
    ]


@pytest.mark.unit
def test_controller_records_and_erases_stage_timestamps(controller_under_test, monkeypatch):
    controller_module, _ = controller_under_test
    task = FakeTask()

    erase_calls = []
    monkeypatch.setattr(
        controller_module.TimeEstimator,
        "record_dag_ts",
        staticmethod(lambda cur_task, is_end=False, sub_tag=None: 1.25 if sub_tag == "transmit" else 0.5),
    )
    monkeypatch.setattr(
        controller_module.TimeEstimator,
        "erase_dag_ts",
        staticmethod(lambda cur_task, is_end=False, sub_tag=None: erase_calls.append((sub_tag, is_end))),
    )

    controller_module.Controller.record_transmit_ts(task, is_end=True)
    controller_module.Controller.record_execute_ts(task, is_end=True)
    controller_module.Controller.erase_transmit_ts(task)
    controller_module.Controller.erase_execute_ts(task)

    assert task.transmit_durations == [1.25]
    assert task.execute_durations == [0.5]
    assert erase_calls == [("transmit", False), ("transmit", True), ("execute", False)]
