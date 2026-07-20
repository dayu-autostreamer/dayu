import asyncio
import importlib
from types import SimpleNamespace

import pytest

from core.lib.common import Queue
from core.lib.content import Task
from core.lib.runtime import RuntimeContext


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
        runtime_directory_revision=1,
        runtime_routes=[{
            "slot": {"component": "controller", "target_node": "edge-node"},
            "runtime_id": "controller-edge-node-r1",
            "runtime_revision": 1,
            "endpoint": {
                "dns_name": "controller-edge-node.dayu.svc",
                "port": 9002,
                "runtime_service_uid": "rs-controller",
                "service_uid": "svc-controller",
                "pod_uid": "pod-controller",
            },
        }],
    )


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
    monkeypatch.setenv("DAYU_RUNTIME_BOOTSTRAP", '{"local_node":"edge-node","cloud_node":"cloud-node"}')
    RuntimeContext.reset_default()
    loop_process = processor_server_module.ProcessorServer.loop_process
    monkeypatch.setattr(processor_server_module.ProcessorServer, "loop_process", lambda self: None)
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "touch_task_file_in_temp",
        staticmethod(lambda task: True),
    )
    server = processor_server_module.ProcessorServer()
    monkeypatch.setattr(processor_server_module.ProcessorServer, "loop_process", loop_process)
    return SimpleNamespace(
        server=server,
        queue=fake_queue,
        processor=fake_processor,
    )


@pytest.mark.unit
def test_processor_server_task_endpoints_return_ack_after_queue_ownership(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector")
    saved = []
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        staticmethod(lambda current, payload: saved.append((current.get_task_uuid(), payload))),
    )

    expected_ack = {"accepted": True, "task_uuid": task.get_task_uuid()}
    assert asyncio.run(server.process_service(FakeUploadFile(b"payload"), task.serialize())) == expected_ack
    assert asyncio.run(server.process_local_service(task.serialize())) == expected_ack
    assert saved == [(task.get_task_uuid(), b"payload")]

    assert asyncio.run(server.health_check()) == {"status": "ok"}
    queue_state = asyncio.run(server.query_queue_state())
    assert queue_state["waiting_count"] == 1
    assert [item["root_uuid"] for item in queue_state["waiting_tasks"]] == [
        task.get_root_uuid()
    ]
    assert queue_state["busy"] is False
    assert queue_state["capacity"] == 1
    assert queue_state["running_task"] is None
    assert queue_state["running_phase"] is None
    assert queue_state["phase_elapsed_s"] == 0.0
    assert asyncio.run(server.query_model_flops()) == 456.0
    assert isinstance(asyncio.run(server.query_model_memory()), int)


@pytest.mark.unit
def test_processor_server_accept_handlers_queue_tasks_and_persist_temp_files(server_context, monkeypatch):
    saved = []
    server = server_context.server
    task = build_task(["detector"], "detector")

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        lambda current_task, file_data: saved.append((current_task.get_file_path(), file_data)),
    )

    server.accept_task(task.serialize(), b"payload")
    server.accept_local_task(task.serialize())

    assert saved == [("payload.bin", b"payload")]
    assert server_context.queue.size() == 1


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
        "deliver_task",
        lambda **kwargs: requests.append(kwargs) or True,
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
    assert not hasattr(server, "runtime_lease_client")
    assert processed.get_service("detector").get_real_execute_time() == 0.75
    assert durations == [(4, False, "real_execute"), (4, True, "real_execute")]
    assert requests == [{
        "url": "http://controller-edge-node.dayu.svc:9002/process_return_task",
        "method": "POST",
        "task": processed,
        "persistent": True,
    }]


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
def test_processor_execution_does_not_require_runtime_lease_client(server_context):
    server = server_context.server
    task = build_task(["detector"], "detector")

    assert not hasattr(server, "runtime_lease_client")
    assert server.process_task_service(task) is task
    assert server_context.processor.calls == [task]


@pytest.mark.unit
def test_processor_server_process_return_service_serializes_processed_task_without_branch_cleanup(
    server_context, monkeypatch
):
    server = server_context.server
    task = build_task(["detector"], "detector", file_path="return.bin")
    saved = []

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        lambda current_task, file_data: saved.append((current_task.get_file_path(), file_data)),
    )
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: (_ for _ in ()).throw(AssertionError("must not delete shared artifact")),
    )

    upload = FakeUploadFile(b"payload")
    serialized = asyncio.run(server.process_return_service(upload, task.serialize()))
    returned_task = Task.deserialize(serialized)

    assert saved == [("return.bin", b"payload")]
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

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        lambda current_task, file_data: saved.append((current_task.get_file_path(), file_data)),
    )
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: (_ for _ in ()).throw(AssertionError("must not delete shared artifact")),
    )
    server.processor = lambda current_task: None

    response = asyncio.run(server.process_return_service(FakeUploadFile(b"payload"), task.serialize()))

    assert response is None
    assert saved == [("none.bin", b"payload")]


@pytest.mark.unit
def test_processor_server_process_return_service_preserves_artifact_when_processor_fails(
    server_context, monkeypatch
):
    server = server_context.server
    task = build_task(["detector"], "detector", file_path="failed.bin")

    monkeypatch.setattr(
        processor_server_module.FileOps,
        "save_task_file_in_temp",
        lambda current_task, file_data: None,
    )
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: (_ for _ in ()).throw(AssertionError("must not delete shared artifact")),
    )
    server.processor = lambda current_task: (_ for _ in ()).throw(RuntimeError("processor failed"))

    with pytest.raises(RuntimeError, match="processor failed"):
        asyncio.run(server.process_return_service(FakeUploadFile(b"payload"), task.serialize()))



@pytest.mark.unit
def test_processor_server_clear_queue_supports_preview_and_bounded_removal(server_context, monkeypatch):
    server = server_context.server
    first = build_task(["detector"], "detector", file_path="first.bin")
    second = build_task(["detector"], "detector", file_path="second.bin")
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: (_ for _ in ()).throw(AssertionError("queue clear must not delete root artifact")),
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

    cleared = asyncio.run(server.clear_queue('{"max_count": "1", "reason": "drop-one"}'))
    assert cleared["ok"] is True
    assert cleared["dry_run"] is False
    assert cleared["matched_count"] == 1
    assert cleared["cleared_count"] == 1
    assert cleared["remaining_count"] == 1

    cleared_rest = asyncio.run(server.clear_queue("{}"))
    assert cleared_rest["matched_count"] == 1
    assert cleared_rest["remaining_count"] == 0


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
    monkeypatch.setattr(
        processor_server_module.FileOps,
        "remove_task_file_in_temp",
        lambda current_task: (_ for _ in ()).throw(AssertionError("queue clear must not delete root artifact")),
    )
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
            self.calls = 0

        def get(self):
            self.calls += 1
            if self.calls == 1:
                return self.item
            raise StopIteration

        def size(self):
            return 0

    monkeypatch.setattr(server, "task_queue", OneShotQueue(task))
    monkeypatch.setattr(server, "process_task_service", lambda current_task: current_task)
    observed_states = []

    def forward(current_task):
        observed_states.append(asyncio.run(server.query_queue_state()))
        forwarded.append(current_task)

    monkeypatch.setattr(server, "send_result_back_to_controller", forward)

    with pytest.raises(StopIteration):
        server.loop_process()

    assert forwarded == [task]
    assert observed_states[0]["busy"] is True
    assert observed_states[0]["running_task"]["root_uuid"] == task.get_root_uuid()
    assert observed_states[0]["running_phase"] == "handoff"
    assert asyncio.run(server.query_queue_state())["busy"] is False


@pytest.mark.unit
def test_processor_queue_state_keeps_waiting_and_running_ownership_atomic(server_context):
    server = server_context.server
    first = build_task(["detector"], "detector", file_path="first.bin")
    second = build_task(["detector"], "detector", file_path="second.bin")

    assert server._enqueue_task_once(first) is True
    assert server._enqueue_task_once(second) is True
    queued = asyncio.run(server.query_queue_state())
    assert queued["waiting_count"] == 2
    assert [item["root_uuid"] for item in queued["waiting_tasks"]] == [
        first.get_root_uuid(),
        second.get_root_uuid(),
    ]

    assert server._dequeue_task() is first
    running = asyncio.run(server.query_queue_state())
    assert running["busy"] is True
    assert running["running_phase"] == "processing"
    assert running["running_task"]["root_uuid"] == first.get_root_uuid()
    assert [item["root_uuid"] for item in running["waiting_tasks"]] == [
        second.get_root_uuid()
    ]

    server._finish_running_task(requeue_task=first)
    requeued = asyncio.run(server.query_queue_state())
    assert requeued["busy"] is False
    assert [item["root_uuid"] for item in requeued["waiting_tasks"]] == [
        second.get_root_uuid(),
        first.get_root_uuid(),
    ]


@pytest.mark.unit
def test_processor_server_loop_process_skips_empty_none_error_and_none_result(server_context, monkeypatch):
    server = server_context.server
    task = build_task(["detector"], "detector")
    forwarded = []

    class BranchQueue:
        def __init__(self):
            self.steps = iter([None, task, task])
            self.requeued = []

        def get(self):
            try:
                return next(self.steps)
            except StopIteration:
                raise StopIteration

        def size(self):
            return 0

        def put(self, current_task):
            self.requeued.append(current_task)

    outcomes = iter([RuntimeError("processor failed"), None])

    def fake_process(current_task):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    branch_queue = BranchQueue()
    monkeypatch.setattr(server, "task_queue", branch_queue)
    monkeypatch.setattr(server, "process_task_service", fake_process)
    monkeypatch.setattr(server, "send_result_back_to_controller", lambda current_task: forwarded.append(current_task))
    monkeypatch.setattr(processor_server_module.time, "sleep", lambda seconds: None)

    with pytest.raises(StopIteration):
        server.loop_process()

    assert forwarded == []
    assert branch_queue.requeued == [task, task]
