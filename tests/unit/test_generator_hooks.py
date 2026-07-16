import importlib
import json
from pathlib import Path

import pytest

from core.lib.content import Task
from core.lib.runtime import (
    RuntimeContext,
    RuntimeLeaseIdentityError,
    RuntimeLeaseRetired,
    RuntimeLeaseUnavailable,
)


class StopGeneratorLoop(RuntimeError):
    pass


def build_dag_deployment(execute_device="edge-node"):
    return {
        "face-detection": {
            "service": {
                "service_name": "face-detection",
                "execute_device": execute_device,
            },
            "next_nodes": [],
        }
    }


def runtime_route(component, node, port, logical_service=""):
    slot = {"component": component, "target_node": node}
    if logical_service:
        slot["logical_service"] = logical_service
    return {
        "slot": slot,
        "runtime_id": f"{component}-{logical_service or 'runtime'}-{node}",
        "runtime_revision": 1,
        "endpoint": {
            "dns_name": node,
            "port": port,
            "runtime_service_uid": f"rs-{component}-{node}",
            "service_uid": f"svc-{component}-{node}",
            "pod_uid": f"pod-{component}-{node}",
        },
    }


def runtime_routes(node="edge-node"):
    return [
        runtime_route("controller", node, 9002),
        runtime_route("processor", node, 9004, "face-detection"),
    ]


@pytest.fixture(autouse=True)
def reset_runtime_context():
    RuntimeContext.reset_default()
    yield
    RuntimeContext.reset_default()


def patch_generator_runtime(monkeypatch, generator_module, hooks, video_generator_module=None):
    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        try:
            return hooks[algorithm]
        except KeyError as exc:
            raise AssertionError(f"Unexpected algorithm request: {algorithm}") from exc

    monkeypatch.setattr(generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    if video_generator_module is not None:
        monkeypatch.setattr(video_generator_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))

    monkeypatch.setenv("ALL_EDGE_DEVICES", "['edge-node', 'edge-target']")
    monkeypatch.setenv("REQUEST_SCHEDULING_INTERVAL", "1")
    monkeypatch.setenv("DAYU_RUNTIME_BOOTSTRAP", json.dumps({
        "local_node": "edge-node",
        "cloud_node": "cloud-node",
        "nodes": {
            "edge-node": {"role": "edge", "address": "edge-node"},
            "edge-target": {"role": "edge", "address": "edge-target"},
            "cloud-node": {"role": "cloud", "address": "cloud-node"},
        },
        "endpoints": {
            "scheduler": {"fqdn": "cloud-node", "port": 9001},
        },
    }))
    RuntimeContext.reset_default()


@pytest.mark.unit
def test_generator_request_schedule_policy_and_generate_task_follow_hook_contracts(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")

    hook_calls = {}

    def before_schedule(system):
        hook_calls["before"] = {
            "source_id": system.source_id,
            "meta_data": system.raw_meta_data,
        }
        return hook_calls["before"]

    def after_schedule(system, scheduler_response):
        hook_calls["after"] = scheduler_response

    hooks = {
        "GEN_BSO": before_schedule,
        "GEN_ASO": after_schedule,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": lambda system, task: None,
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks)

    captured_request = {}

    def fake_http_request(url, method=None, data=None, **kwargs):
        captured_request.update(url=url, method=method, data=data)
        return {
            "plan": {"buffer_size": 2},
            "runtime_directory_revision": 1,
            "runtime_routes": runtime_routes("edge-node"),
        }

    monkeypatch.setattr(generator_module, "http_request", fake_http_request)

    class DummyGenerator(generator_module.Generator):
        def run(self):
            raise NotImplementedError

    generator = DummyGenerator(
        source_id=7,
        metadata={"fps": 25, "buffer_size": 1},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )

    task = generator.generate_task(
        task_id=3,
        task_dag=generator.task_dag,
        service_deployment={"edge-target": ["face-detection"]},
        meta_data={"fps": 15},
        compressed_path="payload.bin",
        hash_codes=[11, 12, 13],
    )

    assert task.get_source_id() == 7
    assert task.get_metadata() == {"fps": 15}
    assert task.get_raw_metadata() == {"fps": 25, "buffer_size": 1}
    assert task.get_hash_data() == [11, 12, 13]

    assert generator.request_schedule_policy() is True

    assert captured_request["url"] == "http://cloud-node:9001/schedule"
    assert captured_request["method"] == "GET"
    assert json.loads(captured_request["data"]["data"]) == {
        "source_id": 7,
        "meta_data": {"fps": 25, "buffer_size": 1},
    }
    assert hook_calls["after"]["plan"] == {"buffer_size": 2}
    routed_task = generator.generate_task(
        task_id=4,
        task_dag=generator.task_dag,
        service_deployment={},
        meta_data={"fps": 15},
        compressed_path="payload.bin",
        hash_codes=[],
    )
    assert routed_task.get_runtime_directory_revision() == 1
    assert routed_task.get_runtime_routes() == runtime_routes("edge-node")


@pytest.mark.unit
def test_generator_submit_task_to_controller_invokes_bsto_records_timing_and_uploads_file(
    monkeypatch,
    tmp_path,
):
    generator_module = importlib.import_module("core.generator.generator")

    call_order = []

    def before_submit(system, task):
        call_order.append(("bsto", task.get_task_id()))

    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": before_submit,
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks)

    uploaded = {}

    def fake_http_request(url, method=None, data=None, files=None, **kwargs):
        if url.endswith("/runtime-directory/task-leases"):
            payload = json.loads(data["data"])
            call_order.append(("lease", payload["root_uuid"]))
            return {
                "revision": payload["revision"],
                "root_uuid": payload["root_uuid"],
                "expires_at": 123.0,
                "valid_for_seconds": float(payload["ttl_seconds"]),
            }
        raise AssertionError(f"unexpected request: {url}")

    monkeypatch.setattr(generator_module, "http_request", fake_http_request)
    monkeypatch.setattr(
        generator_module,
        "deliver_task",
        lambda **kwargs: uploaded.update(kwargs) or True,
    )

    class DummyGenerator(generator_module.Generator):
        def run(self):
            raise NotImplementedError

    generator = DummyGenerator(
        source_id=1,
        metadata={"fps": 10},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )
    generator.runtime_directory_revision = 1
    generator.runtime_routes = runtime_routes("edge-target")

    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"payload")

    task = generator.generate_task(
        task_id=5,
        task_dag=Task.extract_dag_from_dag_deployment(build_dag_deployment(execute_device="edge-target")),
        service_deployment={"edge-target": ["face-detection"]},
        meta_data={"fps": 10},
        compressed_path=str(payload_path),
        hash_codes=[],
    )
    task.set_flow_index("face-detection")

    monkeypatch.setattr(
        generator,
        "record_transmit_start_ts",
        lambda cur_task: call_order.append(("record", cur_task.get_task_id())),
    )

    assert generator.submit_task_to_controller(task) is True

    assert call_order == [
        ("bsto", 5),
        ("lease", task.get_root_uuid()),
        ("record", 5),
    ]
    assert uploaded["url"] == "http://edge-target:9002/submit_task"
    assert uploaded["method"] == "POST"
    assert uploaded["task"] is task
    assert uploaded["file_path"] == str(payload_path)
    assert uploaded["persistent"] is True

    with pytest.raises(AssertionError, match="Task is empty when submit to controller"):
        generator.submit_task_to_controller(None)


@pytest.mark.unit
def test_generator_submit_retries_transient_admission_and_rejects_retired_tasks(monkeypatch, tmp_path):
    generator_module = importlib.import_module("core.generator.generator")
    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": lambda system, task: None,
    }
    patch_generator_runtime(monkeypatch, generator_module, hooks)

    class DummyGenerator(generator_module.Generator):
        def run(self):
            raise NotImplementedError

    generator = DummyGenerator(
        source_id=1,
        metadata={"fps": 10},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )
    generator.runtime_directory_revision = 1
    generator.runtime_routes = runtime_routes("edge-target")
    payload_path = tmp_path / "payload.bin"
    payload_path.write_bytes(b"payload")
    task = generator.generate_task(
        task_id=6,
        task_dag=Task.extract_dag_from_dag_deployment(
            build_dag_deployment(execute_device="edge-target")
        ),
        service_deployment={},
        meta_data={},
        compressed_path=str(payload_path),
        hash_codes=[],
    )
    task.set_flow_index("face-detection")

    admission_attempts = []

    def acquire_after_retry(_task):
        admission_attempts.append(True)
        if len(admission_attempts) == 1:
            raise RuntimeLeaseUnavailable("scheduler temporarily unavailable")

    sleeps = []
    generator.runtime_lease_client.acquire = acquire_after_retry
    monkeypatch.setattr(generator_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(generator_module, "deliver_task", lambda **kwargs: True)

    assert generator.submit_task_to_controller(task) is True
    assert len(admission_attempts) == 2
    assert sleeps == [0.5]
    assert generator._runtime_schedule_refresh_required.is_set() is False

    generator.runtime_lease_client.acquire = lambda _task: (_ for _ in ()).throw(
        RuntimeLeaseRetired(1)
    )
    assert generator.submit_task_to_controller(task) is False
    assert generator._runtime_schedule_refresh_required.is_set() is True

    generator._runtime_schedule_refresh_required.clear()
    generator.runtime_lease_client.acquire = lambda _task: (_ for _ in ()).throw(
        RuntimeLeaseIdentityError("invalid lease identity")
    )
    with pytest.raises(RuntimeLeaseIdentityError, match="invalid lease identity"):
        generator.submit_task_to_controller(task)

    generator._runtime_schedule_refresh_required.clear()
    thread = generator_module.threading.Thread(target=generator._runtime_schedule_refresh_required.set)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert generator._runtime_schedule_refresh_required.is_set() is True


@pytest.mark.unit
def test_video_generator_submit_records_total_start_before_parent_submit(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": lambda system: None,
        "GEN_BSTO": lambda system, task: None,
        "GEN_FILTER": object(),
        "GEN_PROCESS": object(),
        "GEN_COMPRESS": object(),
        "GEN_GETTER_FILTER": object(),
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks, video_generator_module=video_generator_module)

    order = []

    monkeypatch.setattr(
        video_generator_module.VideoGenerator,
        "record_total_start_ts",
        staticmethod(lambda task: order.append("total")),
    )
    monkeypatch.setattr(
        generator_module.Generator,
        "submit_task_to_controller",
        lambda self, task: order.append("submit") or True,
    )

    generator = video_generator_module.VideoGenerator(
        source_id=3,
        source_url="http://source/video",
        source_metadata={"fps": 30},
        dag=build_dag_deployment(),
    )

    assert generator.submit_task_to_controller(object()) is True

    assert order == ["total", "submit"]


@pytest.mark.unit
def test_video_generator_run_requests_initial_schedule_after_health_before_data_getter(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    after_schedule_calls = []
    getter_filter_calls = []
    schedule_requests = []
    data_getter_calls = []

    getter_states = iter([False, True, True])

    def after_schedule(system, scheduler_response):
        after_schedule_calls.append(scheduler_response)

    def data_getter(system):
        data_getter_calls.append(system.source_id)
        system.cumulative_scheduling_frame_count = (
            system.request_scheduling_interval * system.raw_meta_data["fps"] + 1
        )

    def getter_filter(system):
        getter_filter_calls.append(system.source_id)
        return next(getter_states)

    hooks = {
        "GEN_BSO": lambda system: {"source_id": system.source_id},
        "GEN_ASO": after_schedule,
        "GEN_GETTER": data_getter,
        "GEN_BSTO": lambda system, task: None,
        "GEN_FILTER": object(),
        "GEN_PROCESS": object(),
        "GEN_COMPRESS": object(),
        "GEN_GETTER_FILTER": getter_filter,
    }

    patch_generator_runtime(monkeypatch, generator_module, hooks, video_generator_module=video_generator_module)

    monkeypatch.setattr(video_generator_module.time, "sleep", lambda *_args, **_kwargs: None)

    generator = video_generator_module.VideoGenerator(
        source_id=9,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )

    request_count = {"value": 0}

    def fake_request_schedule_policy():
        schedule_requests.append(generator.cumulative_scheduling_frame_count)
        request_count["value"] += 1
        if request_count["value"] >= 2:
            raise StopGeneratorLoop
        return True

    monkeypatch.setattr(generator, "request_schedule_policy", fake_request_schedule_policy)

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert after_schedule_calls == [None]
    assert getter_filter_calls == [9, 9, 9]
    assert data_getter_calls == [9]
    assert schedule_requests == [0, 6]


@pytest.mark.unit
def test_video_generator_run_refreshes_schedule_after_retired_lease(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    data_getter_calls = []

    def data_getter(system):
        data_getter_calls.append(system.source_id)
        system._runtime_schedule_refresh_required.set()

    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": data_getter,
        "GEN_BSTO": lambda system, task: None,
        "GEN_FILTER": object(),
        "GEN_PROCESS": object(),
        "GEN_COMPRESS": object(),
        "GEN_GETTER_FILTER": lambda system: True,
    }
    patch_generator_runtime(
        monkeypatch,
        generator_module,
        hooks,
        video_generator_module=video_generator_module,
    )

    generator = video_generator_module.VideoGenerator(
        source_id=10,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )
    schedule_requests = []

    def fake_request_schedule_policy():
        schedule_requests.append(generator.cumulative_scheduling_frame_count)
        if len(schedule_requests) == 2:
            raise StopGeneratorLoop
        return True

    monkeypatch.setattr(generator, "request_schedule_policy", fake_request_schedule_policy)

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert data_getter_calls == [10]
    assert schedule_requests == [0, 0]
