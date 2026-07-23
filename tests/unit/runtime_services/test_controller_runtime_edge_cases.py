import importlib
from pathlib import Path

import pytest

from core.lib.content import Task
from core.lib.runtime import RuntimeContext, RuntimeResolver


def controller_route(node):
    return {
        "slot": {"component": "controller", "target_node": node},
        "runtime_id": f"controller-{node}",
        "runtime_revision": 1,
        "endpoint": {
            "dns_name": f"controller-{node}.dayu.svc", "port": 30001,
            "runtime_service_uid": f"rs-{node}", "service_uid": f"svc-{node}", "pod_uid": f"pod-{node}",
        },
    }


class FakeTask:
    def __init__(self, service_name="face-detection", stage_device="edgex1", file_path="payload.bin"):
        self.service_name = service_name
        self.stage_device = stage_device
        self.file_path = file_path
        self.serialized = {"task": "serialized"}
        self.transmit_durations = []
        self.execute_durations = []
        self.runtime_routes = [controller_route("edgex1"), controller_route("cloudx1")]
        self.runtime_directory_revision = 1
        self.root_uuid = "root-2"

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

    def get_runtime_routes(self):
        return self.runtime_routes

    def get_runtime_directory_revision(self):
        return self.runtime_directory_revision

    def get_root_uuid(self):
        return self.root_uuid

    def get_task_uuid(self):
        return "branch-2"

    def save_transmit_time(self, duration):
        self.transmit_durations.append(duration)

    def save_execute_time(self, duration):
        self.execute_durations.append(duration)


class StartTask(FakeTask):
    def __init__(self, next_tasks):
        super().__init__(service_name="_start", stage_device="edgex1")
        self._next_tasks = next_tasks

    def step_to_next_stage(self):
        return list(self._next_tasks)


def build_parallel_branch_task(branch_name, value, root_uuid="root-task-0"):
    dag = Task.extract_dag_from_dag_deployment(
        {
            "detector-a": {
                "service": {"service_name": "detector-a", "execute_device": "edge-node"},
                "next_nodes": ["join"],
            },
            "detector-b": {
                "service": {"service_name": "detector-b", "execute_device": "edge-node"},
                "next_nodes": ["join"],
            },
            "join": {
                "service": {"service_name": "join", "execute_device": "edge-node"},
                "next_nodes": [],
            },
        }
    )
    task = Task(
        source_id=0,
        task_id=0,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=dag,
        flow_index=branch_name,
        past_flow_index="_start",
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="payload.bin",
        root_uuid=root_uuid,
        runtime_directory_revision=1,
    )
    task.set_current_content({"branch": branch_name, "value": value})
    task.add_scenario({"branch": value})
    return task


@pytest.fixture
def controller_under_test(monkeypatch):
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)
    controller.task_coordinator = None
    controller.is_display = False
    controller.distribute_address = "http://10.0.0.9:30002/distribute"
    controller.local_device = "edgex1"
    controller.cloud_device = "cloudx1"

    controller.runtime_context = RuntimeContext({"local_node": "edgex1", "cloud_node": "cloudx1"})
    controller.runtime_resolver = RuntimeResolver(controller.runtime_context)
    return controller_module, controller


@pytest.mark.unit
def test_send_task_to_other_device_uploads_task_file_and_records_transmit_timestamp(
    controller_under_test, monkeypatch, tmp_path
):
    controller_module, controller = controller_under_test
    task = FakeTask()
    temp_file = tmp_path / task.get_file_path()
    temp_file.write_bytes(b"payload")

    transmit_records = []
    request_calls = []

    monkeypatch.setattr(
        controller_module.FileOps,
        "get_task_file_in_temp",
        staticmethod(lambda cur_task: str(temp_file)),
    )
    monkeypatch.setattr(
        controller_module.Controller,
        "record_transmit_ts",
        staticmethod(lambda cur_task, is_end=False: transmit_records.append((cur_task.get_task_id(), is_end))),
    )
    monkeypatch.setattr(controller_module, "deliver_task", lambda **kwargs: request_calls.append(kwargs) or True)

    assert controller.send_task_to_other_device(task, "cloudx1") is True

    assert transmit_records == [(2, False)]
    assert request_calls[0]["method"] == controller_module.NetworkAPIMethod.CONTROLLER_TASK
    assert request_calls[0]["persistent"] is True
    assert Path(request_calls[0]["file_path"]).read_bytes() == b"payload"


@pytest.mark.unit
def test_send_task_to_distributor_supports_hidden_and_display_upload_modes(
    controller_under_test, monkeypatch, tmp_path
):
    controller_module, controller = controller_under_test
    task = FakeTask()
    temp_file = tmp_path / task.get_file_path()
    temp_file.write_bytes(b"renderable")

    transmit_records = []
    request_calls = []

    monkeypatch.setattr(
        controller_module.FileOps,
        "get_task_file_in_temp",
        staticmethod(lambda cur_task: str(temp_file)),
    )
    monkeypatch.setattr(
        controller_module.Controller,
        "record_transmit_ts",
        staticmethod(lambda cur_task, is_end=False: transmit_records.append((cur_task.get_task_id(), is_end))),
    )
    monkeypatch.setattr(controller_module, "deliver_task", lambda **kwargs: request_calls.append(kwargs) or True)

    controller.is_display = False
    assert controller.send_task_to_distributor(task) is True
    assert request_calls[0]["file_content"] == b""
    assert request_calls[0]["persistent"] is True

    controller.is_display = True
    assert controller.send_task_to_distributor(task) is True
    assert Path(request_calls[1]["file_path"]).read_bytes() == b"renderable"
    assert request_calls[1]["persistent"] is True

    assert transmit_records == [(2, False), (2, False)]


@pytest.mark.unit
def test_send_task_to_distributor_only_requires_file_for_display_mode(controller_under_test, monkeypatch, tmp_path):
    controller_module, controller = controller_under_test
    task = FakeTask(file_path="missing.bin")
    request_calls = []

    monkeypatch.setattr(
        controller_module.FileOps,
        "get_task_file_in_temp",
        staticmethod(lambda cur_task: str(tmp_path / "missing.bin")),
    )
    monkeypatch.setattr(controller_module, "deliver_task", lambda **kwargs: request_calls.append(kwargs) or True)

    controller.is_display = False
    assert controller.send_task_to_distributor(task) is True
    assert request_calls[0]["file_content"] == b""

    controller.is_display = True
    assert controller.send_task_to_distributor(task) is False
    assert len(request_calls) == 1


@pytest.mark.unit
def test_submit_task_start_stage_requires_every_child_ack(controller_under_test, monkeypatch):
    controller_module, controller = controller_under_test
    execute_task = FakeTask(service_name="face-detection", stage_device="edgex1")
    remote_task = FakeTask(service_name="gender-classification", stage_device="cloudx1")
    start_task = StartTask([execute_task, remote_task])

    events = []
    monkeypatch.setattr(controller, "send_task_to_service", lambda cur_task, service: events.append(("service", service)) or True)
    monkeypatch.setattr(controller, "send_task_to_other_device", lambda cur_task, device: events.append(("remote", device)) or False)

    assert controller.submit_task(start_task) is False
    assert events == [("service", "face-detection"), ("remote", "cloudx1")]


@pytest.mark.unit
def test_submit_task_start_stage_succeeds_when_all_children_ack(controller_under_test, monkeypatch):
    _, controller = controller_under_test
    remote_a = FakeTask(service_name="face-detection", stage_device="cloudx1")
    remote_b = FakeTask(service_name="gender-classification", stage_device="cloudx1")
    start_task = StartTask([remote_a, remote_b])

    events = []
    monkeypatch.setattr(controller, "send_task_to_other_device", lambda cur_task, device: events.append(device) or True)

    assert controller.submit_task(start_task) is True
    assert events == ["cloudx1", "cloudx1"]


@pytest.mark.unit
def test_controller_advances_without_per_stage_runtime_lease(controller_under_test, monkeypatch):
    _, controller = controller_under_test
    task = FakeTask(service_name="face-detection", stage_device="edgex1")
    forwarded = []

    monkeypatch.setattr(
        controller,
        "send_task_to_service",
        lambda *args, **kwargs: forwarded.append((args, kwargs)) or True,
    )

    assert not hasattr(controller, "runtime_lease_client")
    assert controller.submit_task(task) is True
    assert len(forwarded) == 1


@pytest.mark.unit
def test_process_return_acknowledges_branch_retained_in_incomplete_barrier():
    controller_module = importlib.import_module("core.controller.controller")
    controller = object.__new__(controller_module.Controller)

    class BrokenCoordinator:
        def arrive(self, task, joint_service_name, required_count):
            return None

        def complete(self, root_uuid, joint_service_name):
            raise AssertionError("incomplete barrier must not complete")

    controller.task_coordinator = BrokenCoordinator()
    submitted_tasks = []
    controller.submit_task = lambda task: submitted_tasks.append(task) or True

    accepted = controller.process_return(build_parallel_branch_task("detector-a", "left"))

    assert accepted is True
    assert submitted_tasks == []


@pytest.mark.unit
def test_controller_timestamp_helpers_default_to_zero_when_estimation_fails(controller_under_test, monkeypatch):
    controller_module, _ = controller_under_test
    task = FakeTask()

    monkeypatch.setattr(
        controller_module.TimeEstimator,
        "record_dag_ts",
        staticmethod(lambda cur_task, is_end=False, sub_tag=None: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    monkeypatch.setattr(
        controller_module.TimeEstimator,
        "erase_dag_ts",
        staticmethod(lambda cur_task, is_end=False, sub_tag=None: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    controller_module.Controller.record_transmit_ts(task, is_end=True)
    controller_module.Controller.record_execute_ts(task, is_end=True)
    controller_module.Controller.erase_transmit_ts(task)
    controller_module.Controller.erase_execute_ts(task)

    assert task.transmit_durations == [0]
    assert task.execute_durations == [0]
