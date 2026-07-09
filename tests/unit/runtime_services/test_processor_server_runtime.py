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
        task.set_current_content(
            {
                "service": task.get_flow_index(),
                "outputs": {"text": [{"frame_index": None, "items": [{"text": "processed"}]}]},
                "profile": {
                    "frame_count": 0,
                },
            }
        )
        return task

    @property
    def flops(self):
        return 456.0


class FakeUploadFile:
    def __init__(self, payload):
        self.payload = payload

    async def read(self):
        return self.payload


class FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


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

    def fake_get_parameter(name, default=None, **kwargs):
        if name == "PROCESSOR_SERVICE_NAME":
            return default
        return "9004"

    monkeypatch.setattr(processor_server_module.Context, "get_parameter", staticmethod(fake_get_parameter))
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
def test_processor_server_async_endpoints_enqueue_background_tasks_and_report_metrics(server_context):
    server = server_context.server
    task = build_task(["detector"], "detector")

    predict_tasks = FakeBackgroundTasks()
    asyncio.run(server.process_service(predict_tasks, FakeUploadFile(b"payload"), task.serialize()))
    assert predict_tasks.tasks == [
        (server.process_service_background, (task.serialize(), b"payload"), {}),
    ]

    local_tasks = FakeBackgroundTasks()
    asyncio.run(server.process_local_service(local_tasks, task.serialize()))
    assert local_tasks.tasks == [
        (server.process_local_service_background, (task.serialize(),), {}),
    ]

    server_context.queue.put(task)
    assert asyncio.run(server.health_check()) == {"status": "ok"}
    assert asyncio.run(server.query_queue_length()) == 1
    assert asyncio.run(server.query_model_flops()) == 456.0
    assert isinstance(asyncio.run(server.query_model_memory()), int)


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

    assert processed.get_current_content() == {
        "service": "detector",
        "outputs": {"text": [{"frame_index": None, "items": [{"text": "processed"}]}]},
        "profile": {
            "frame_count": 0,
        },
    }
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
def test_processor_server_process_task_service_skips_none_result(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector")
    durations = []

    def fake_record(current_task, is_end, sub_tag="real_execute"):
        durations.append((current_task.get_task_id(), is_end, sub_tag))
        return 0.75 if is_end else 0

    monkeypatch.setattr(processor_server_module.TimeEstimator, "record_dag_ts", staticmethod(fake_record))
    monkeypatch.setattr(server_context, "processor", lambda current_task: None)
    server.processor = lambda current_task: None

    assert server.process_task_service(task) is None
    assert durations == [(4, False, "real_execute")]


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
    assert returned_task.get_current_content() == {
        "service": "detector",
        "outputs": {"text": [{"frame_index": None, "items": [{"text": "processed"}]}]},
        "profile": {
            "frame_count": 0,
        },
    }


@pytest.mark.unit
def test_processor_server_process_return_service_handles_processor_returning_none(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector", file_path="none.bin")
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
    server.processor = lambda current_task: None

    response = asyncio.run(server.process_return_service(FakeUploadFile(b"payload"), task.serialize()))

    assert response is None
    assert saved == [("none.bin", b"payload")]
    assert removed == ["none.bin"]


@pytest.mark.unit
def test_processor_server_clear_queue_supports_preview_and_bounded_removal(server_context, monkeypatch):
    server = server_context.server
    first = build_task(["detector"], "detector", file_path="first.bin")
    second = build_task(["detector"], "detector", file_path="second.bin")
    removed = []

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: removed.append(current_task.get_file_path()),
    )

    server_context.queue.put(first)
    server_context.queue.put(second)

    preview = asyncio.run(server.clear_queue('{"dry_run": true, "max_count": 1, "reason": "preview"}'))
    assert preview["ok"] is True
    assert preview["dry_run"] is True
    assert preview["matched_count"] == 1
    assert preview["cleared_count"] == 0
    assert preview["remaining_count"] == 2
    assert preview["dropped_tasks"] == [
        {"source_id": 3, "task_id": 4, "flow_index": "detector", "file_path": "first.bin"}
    ]
    assert removed == []

    cleared = asyncio.run(server.clear_queue('{"max_count": "1", "reason": "drop-one"}'))
    assert cleared["ok"] is True
    assert cleared["dry_run"] is False
    assert cleared["matched_count"] == 1
    assert cleared["cleared_count"] == 1
    assert cleared["remaining_count"] == 1
    assert removed == ["first.bin"]

    cleared_rest = asyncio.run(server.clear_queue("{}"))
    assert cleared_rest["matched_count"] == 1
    assert cleared_rest["remaining_count"] == 0
    assert removed == ["first.bin", "second.bin"]


@pytest.mark.unit
def test_processor_server_clear_queue_reports_invalid_requests_and_legacy_queue_fallback(server_context, monkeypatch):
    server = server_context.server

    invalid = asyncio.run(server.clear_queue("{not-json"))
    assert invalid["ok"] is False
    assert "invalid queue clear request" in invalid["error"]

    server.task_queue = SimpleNamespace(size=lambda: 0)
    unsupported_preview = asyncio.run(server.clear_queue('{"dry_run": true}'))
    assert unsupported_preview == {
        "ok": False,
        "error": "queue does not support dry_run preview",
    }

    class LegacyQueue:
        def __init__(self, items):
            self.items = list(items)

        def get(self):
            if not self.items:
                return None
            return self.items.pop(0)

        def size(self):
            return len(self.items)

    first = build_task(["detector"], "detector", file_path="legacy-first.bin")
    second = build_task(["detector"], "detector", file_path="legacy-second.bin")
    removed = []

    def fake_remove(current_task):
        removed.append(current_task.get_file_path())
        if current_task.get_file_path() == "legacy-first.bin":
            raise RuntimeError("remove failed")

    monkeypatch.setattr(processor_server_module.FileOps, "remove_task_file_in_temp", fake_remove)
    server.task_queue = LegacyQueue([first, second])

    response = asyncio.run(server.clear_queue('{"max_count": 3, "reason": "legacy"}'))

    assert response["ok"] is True
    assert response["matched_count"] == 2
    assert response["cleared_count"] == 2
    assert response["remaining_count"] == 0
    assert [record["file_path"] for record in response["dropped_tasks"]] == [
        "legacy-first.bin",
        "legacy-second.bin",
    ]
    assert removed == ["legacy-first.bin", "legacy-second.bin"]


@pytest.mark.unit
def test_processor_server_clear_queue_normalizes_payload_and_drop_records(server_context):
    server = server_context.server
    task = build_task(["detector"], "detector", file_path="payload.bin")

    server_context.queue.put(task)
    response = asyncio.run(server.clear_queue('"not-a-dict"'))

    assert response["ok"] is True
    assert response["matched_count"] == 1
    assert response["remaining_count"] == 0

    class BrokenTask:
        def get_source_id(self):
            raise RuntimeError("no source")

        def get_task_id(self):
            return 42

        file_path = "not-callable"

    assert server._normalize_queue_clear_limit("bad") is None
    assert server._normalize_queue_clear_limit(0) is None
    assert server._task_drop_record(BrokenTask()) == {
        "source_id": None,
        "task_id": 42,
        "flow_index": None,
        "file_path": None,
    }


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


@pytest.mark.unit
def test_processor_server_loop_process_skips_empty_none_error_and_none_result(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector")
    forwarded = []

    class BranchQueue:
        def __init__(self):
            self.steps = iter(["empty", "none-task", "error", "none-result", "stop"])
            self.current = None

        def empty(self):
            self.current = next(self.steps)
            if self.current == "stop":
                raise StopIteration
            return self.current == "empty"

        def get(self):
            if self.current == "none-task":
                return None
            return task

        def size(self):
            return 0

    outcomes = iter([RuntimeError("processor failed"), None])

    def fake_process(current_task):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(server, "task_queue", BranchQueue())
    monkeypatch.setattr(server, "process_task_service", fake_process)
    monkeypatch.setattr(server, "send_result_back_to_controller", lambda current_task: forwarded.append(current_task))

    with pytest.raises(StopIteration):
        server.loop_process()

    assert forwarded == []
