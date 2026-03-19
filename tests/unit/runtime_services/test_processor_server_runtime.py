import asyncio
import importlib
from types import SimpleNamespace

import pytest

from core.lib.common import Queue
from core.lib.content import Task


processor_server_module = importlib.import_module("core.processor.processor_server")


def build_task(service_names, flow_index, file_path="payload.bin"):
    dag_deployment = {}
    for index, service_name in enumerate(service_names):
        dag_deployment[service_name] = {
            "service": {"service_name": service_name, "execute_device": "edge-node"},
            "next_nodes": service_names[index + 1:index + 2],
        }
    return Task(
        source_id=3,
        task_id=4,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=Task.extract_dag_from_dag_deployment(dag_deployment),
        flow_index=flow_index,
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path=file_path,
    )


class DummyThread:
    def __init__(self, target=None, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, task):
        self.calls.append(task)
        task.set_current_content({"processed": task.get_flow_index()})
        return task

    @property
    def flops(self):
        return 456.0


class FakeUploadFile:
    def __init__(self, payload):
        self.payload = payload

    async def read(self):
        return self.payload


@pytest.fixture
def server_context(monkeypatch):
    fake_queue = Queue()
    fake_processor = FakeProcessor()

    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        if algorithm == "PROCESSOR":
            return fake_processor
        if algorithm == "PRO_QUEUE":
            return fake_queue
        raise AssertionError(f"Unexpected algorithm request: {algorithm}")

    monkeypatch.setattr(processor_server_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setattr(processor_server_module.Context, "get_parameter", staticmethod(lambda name: "9004"))
    monkeypatch.setattr(processor_server_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-node"))
    monkeypatch.setattr(processor_server_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: hostname))
    monkeypatch.setattr(
        processor_server_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: 9002),
    )
    monkeypatch.setattr(processor_server_module.threading, "Thread", DummyThread)
    server = processor_server_module.ProcessorServer()
    return SimpleNamespace(server=server, queue=fake_queue, processor=fake_processor)


@pytest.mark.unit
def test_processor_server_background_handlers_queue_tasks_and_persist_temp_files(server_context, monkeypatch):
    saved = []
    server = server_context.server
    task = build_task(["detector"], "detector")

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        lambda current_task, file_data: saved.append((current_task.get_file_path(), file_data)),
    )

    server.process_service_background(task.serialize(), b"payload")
    server.process_local_service_background(task.serialize())

    assert saved == [("payload.bin", b"payload")]
    assert server_context.queue.size() == 2


@pytest.mark.unit
def test_processor_server_process_task_service_records_duration_and_sends_results(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector")
    durations = []
    requests = []

    def fake_record(current_task, is_end, sub_tag="real_execute"):
        durations.append((current_task.get_task_id(), is_end, sub_tag))
        return 0.75 if is_end else 0

    monkeypatch.setattr(processor_server_module.TimeEstimator, "record_dag_ts", staticmethod(fake_record))
    monkeypatch.setattr(
        processor_server_module,
        "http_request",
        lambda url, method=None, **kwargs: requests.append((url, method, kwargs)),
    )

    processed = server.process_task_service(task)
    server.send_result_back_to_controller(processed)

    assert processed.get_current_content() == {"processed": "detector"}
    assert processed.get_service("detector").get_real_execute_time() == 0.75
    assert durations == [(4, False, "real_execute"), (4, True, "real_execute")]
    assert requests == [
        (
            "http://edge-node:9002/process_return_task",
            "POST",
            {"data": {"data": processed.serialize()}},
        )
    ]


@pytest.mark.unit
def test_processor_server_process_return_service_serializes_processed_task_and_cleans_temp_file(
    server_context, monkeypatch
):
    server = server_context.server
    task = build_task(["detector"], "detector", file_path="return.bin")
    saved = []
    removed = []

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        lambda current_task, file_data: saved.append((current_task.get_file_path(), file_data)),
    )
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: removed.append(current_task.get_file_path()),
    )

    upload = FakeUploadFile(b"payload")
    serialized = asyncio.run(server.process_return_service(upload, task.serialize()))
    returned_task = Task.deserialize(serialized)

    assert saved == [("return.bin", b"payload")]
    assert removed == ["return.bin"]
    assert returned_task.get_current_content() == {"processed": "detector"}


@pytest.mark.unit
def test_processor_server_loop_process_consumes_queue_once_and_forwards_results(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector")
    forwarded = []

    class OneShotQueue:
        def __init__(self, item):
            self.item = item
            self.empty_calls = 0

        def empty(self):
            self.empty_calls += 1
            if self.empty_calls == 1:
                return False
            raise StopIteration

        def get(self):
            return self.item

        def size(self):
            return 0

    monkeypatch.setattr(server, "task_queue", OneShotQueue(task))
    monkeypatch.setattr(server, "process_task_service", lambda current_task: current_task)
    monkeypatch.setattr(server, "send_result_back_to_controller", lambda current_task: forwarded.append(current_task))

    with pytest.raises(StopIteration):
        server.loop_process()

    assert forwarded == [task]
