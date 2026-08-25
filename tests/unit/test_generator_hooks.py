import importlib
import json
import threading
from pathlib import Path

import pytest

from core.lib.content import Task, TaskIdentity
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
        return {
            **hook_calls["before"],
            "runtime_directory_revision": 999,
            "runtime_directory_hash": "forged",
            "runtime_route_cache_keys": ["forged"],
        }

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
        captured_request.update(
            url=url,
            method=method,
            data=data,
            timeout=kwargs.get("timeout"),
        )
        return {
            "plan": {"buffer_size": 2},
            "runtime_directory_revision": 1,
            "runtime_directory_hash": "runtime-directory-hash-1",
            "runtime_routes": runtime_routes("edge-target"),
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

    task_identity = TaskIdentity.create(source_id=7, task_id=4)
    assert generator.request_schedule_policy(task_identity) is True

    expected_task_context = task_identity.to_dict()
    assert captured_request["url"] == "http://cloud-node:9001/schedule"
    assert captured_request["method"] == "GET"
    assert captured_request["timeout"] == 5.0
    assert json.loads(captured_request["data"]["data"]) == {
        "source_id": 7,
        "meta_data": {"fps": 25, "buffer_size": 1},
        "current_configuration": {"fps": 25, "buffer_size": 1},
        "source_device": "edge-node",
        "all_edge_devices": ["edge-node", "edge-target"],
        "dag": Task.extract_dag_deployment_from_dag(generator.task_dag),
        "deployment_version": 0,
        "task_context": expected_task_context,
        "schedule_request_attempt": 1,
    }
    assert hook_calls["after"]["plan"] == {"buffer_size": 2}
    routed_task = generator.generate_task(
        task_id=4,
        task_dag=generator.task_dag,
        service_deployment={},
        meta_data={"fps": 15},
        compressed_path="payload.bin",
        hash_codes=[],
        task_identity=task_identity,
    )
    assert routed_task.get_task_uuid() == task_identity.task_uuid
    assert routed_task.get_root_uuid() == task_identity.root_uuid
    assert routed_task.get_schedule_decision_id()
    assert routed_task.get_schedule_plan_digest()
    assert routed_task.get_runtime_directory_revision() == 1
    assert routed_task.get_runtime_routes() == runtime_routes("edge-target")
    assert generator.runtime_directory_hash == "runtime-directory-hash-1"


@pytest.mark.unit
def test_generator_fresh_schedule_failure_never_reuses_existing_routes(monkeypatch):
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
        source_id=7,
        metadata={"fps": 25, "buffer_size": 1},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )
    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 1,
        "runtime_directory_hash": "runtime-directory-hash-1",
        "runtime_routes": runtime_routes("edge-target"),
    }) is True
    assert generator.runtime_routes_ready() is True

    captured = {}

    def unavailable_schedule(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(generator_module, "http_request", unavailable_schedule)
    task_identity = TaskIdentity.create(source_id=7, task_id=8)

    assert generator.request_schedule_policy(task_identity) is False
    assert captured["timeout"] == 5.0
    assert generator.runtime_routes_ready() is True
    assert generator._schedule_request_attempts[task_identity.root_uuid] == 1


@pytest.mark.unit
def test_generator_runtime_directory_hash_allows_plan_specific_compact_routes(monkeypatch):
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
        source_id=7,
        metadata={"fps": 25, "buffer_size": 1},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )

    initial_routes = runtime_routes("edge-node")
    alternate_routes = runtime_routes("edge-target")
    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "runtime-directory-hash-7",
        "runtime_routes": initial_routes,
    }) is True

    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "runtime-directory-hash-7",
        "runtime_routes": alternate_routes,
    }) is True
    assert generator.runtime_routes == alternate_routes

    accepted_state = (
        generator.runtime_directory_revision,
        generator.runtime_directory_hash,
        generator.runtime_routes,
    )
    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "changed-hash",
        "runtime_routes": initial_routes,
    }) is False
    assert (
        generator.runtime_directory_revision,
        generator.runtime_directory_hash,
        generator.runtime_routes,
    ) == accepted_state

    for missing_hash in (None, "", "   "):
        assert generator._accept_runtime_directory({
            "runtime_directory_revision": 8,
            "runtime_directory_hash": missing_hash,
            "runtime_routes": initial_routes,
        }) is False
        assert (
            generator.runtime_directory_revision,
            generator.runtime_directory_hash,
            generator.runtime_routes,
        ) == accepted_state

    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 8,
        "runtime_directory_hash": "runtime-directory-hash-8",
        "runtime_routes": initial_routes,
    }) is True
    assert generator.runtime_directory_revision == 8
    assert generator.runtime_directory_hash == "runtime-directory-hash-8"
    assert generator.runtime_routes == initial_routes


@pytest.mark.unit
def test_generator_reuses_exact_cached_runtime_routes(monkeypatch):
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
        source_id=7,
        metadata={"fps": 25, "buffer_size": 1},
        task_dag=build_dag_deployment(execute_device="edge-target"),
    )
    routes = runtime_routes("edge-node")
    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "runtime-directory-hash-7",
        "runtime_routes": routes,
    }) is True
    cache_key = generator.runtime_route_cache_keys()[0]
    assert generator.schedule_request_context() == {
        "source_id": 7,
        "meta_data": {"fps": 25, "buffer_size": 1},
        "current_configuration": {"fps": 25, "buffer_size": 1},
        "source_device": "edge-node",
        "all_edge_devices": ["edge-node", "edge-target"],
        "dag": Task.extract_dag_deployment_from_dag(generator.task_dag),
        "deployment_version": 0,
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "runtime-directory-hash-7",
        "runtime_route_cache_keys": [cache_key],
    }

    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "runtime-directory-hash-7",
        "runtime_routes_cache_key": cache_key,
        "runtime_routes_cached": True,
    }) is True
    assert generator.runtime_routes == routes

    assert generator._accept_runtime_directory({
        "runtime_directory_revision": 7,
        "runtime_directory_hash": "runtime-directory-hash-7",
        "runtime_routes_cache_key": "missing",
        "runtime_routes_cached": True,
    }) is False


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

    def data_getter(system, task_identity):
        data_getter_calls.append(task_identity)
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

    task_identities = iter([
        TaskIdentity.create(source_id=9, task_id=100),
        TaskIdentity.create(source_id=9, task_id=101),
    ])
    monkeypatch.setattr(generator, "create_task_identity", lambda: next(task_identities))

    def fake_request_schedule_policy(task_identity=None):
        schedule_requests.append((generator.cumulative_scheduling_frame_count, task_identity))
        request_count["value"] += 1
        if request_count["value"] >= 2:
            raise StopGeneratorLoop
        return True

    monkeypatch.setattr(generator, "request_schedule_policy", fake_request_schedule_policy)

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert after_schedule_calls == [None]
    assert getter_filter_calls == [9, 9, 9]
    assert [identity.task_id for identity in data_getter_calls] == [100]
    assert [count for count, _ in schedule_requests] == [0, 6]
    assert [identity.task_id for _, identity in schedule_requests] == [100, 101]
    assert data_getter_calls[0] is schedule_requests[0][1]


@pytest.mark.unit
def test_video_generator_retries_same_identity_before_ingesting_after_schedule_failure(
    monkeypatch,
):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module(
        "core.generator.video_generator"
    )
    data_getter_calls = []

    def data_getter(system, task_identity):
        data_getter_calls.append(task_identity)
        raise StopGeneratorLoop

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
    monkeypatch.setattr(
        video_generator_module.time,
        "sleep",
        lambda *_args, **_kwargs: None,
    )
    generator = video_generator_module.VideoGenerator(
        source_id=9,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )
    identities = iter([
        TaskIdentity.create(source_id=9, task_id=100),
        TaskIdentity.create(source_id=9, task_id=101),
    ])
    monkeypatch.setattr(
        generator,
        "create_task_identity",
        lambda: next(identities),
    )
    schedule_requests = []

    def request_schedule(task_identity=None):
        schedule_requests.append(task_identity)
        return len(schedule_requests) > 1

    monkeypatch.setattr(
        generator,
        "request_schedule_policy",
        request_schedule,
    )

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert len(schedule_requests) == 2
    assert schedule_requests[0] is schedule_requests[1]
    assert schedule_requests[0].task_id == 100
    assert data_getter_calls == [schedule_requests[0]]


@pytest.mark.unit
def test_video_generator_run_refreshes_schedule_after_retired_lease(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    data_getter_calls = []

    def data_getter(system, task_identity):
        data_getter_calls.append(task_identity)
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

    task_identities = iter([
        TaskIdentity.create(source_id=10, task_id=200),
        TaskIdentity.create(source_id=10, task_id=201),
    ])
    monkeypatch.setattr(generator, "create_task_identity", lambda: next(task_identities))

    def fake_request_schedule_policy(task_identity=None):
        schedule_requests.append((generator.cumulative_scheduling_frame_count, task_identity))
        if len(schedule_requests) == 2:
            raise StopGeneratorLoop
        return True

    monkeypatch.setattr(generator, "request_schedule_policy", fake_request_schedule_policy)

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert [identity.task_id for identity in data_getter_calls] == [200]
    assert [count for count, _ in schedule_requests] == [0, 0]
    assert [identity.task_id for _, identity in schedule_requests] == [200, 201]


@pytest.mark.unit
def test_video_generator_quiesces_after_exact_offered_task_limit(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")
    getter_calls = []

    def data_getter(system, task_identity):
        getter_calls.append(task_identity)
        return True

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
    monkeypatch.setenv("TASK_OFFER_LIMIT", "1")

    generator = video_generator_module.VideoGenerator(
        source_id=10,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )
    identities = iter([
        TaskIdentity.create(source_id=10, task_id=200),
        TaskIdentity.create(source_id=10, task_id=201),
    ])
    monkeypatch.setattr(generator, "create_task_identity", lambda: next(identities))
    monkeypatch.setattr(generator, "request_schedule_policy", lambda identity=None: True)
    monkeypatch.setattr(
        video_generator_module.time,
        "sleep",
        lambda _delay: (_ for _ in ()).throw(StopGeneratorLoop),
    )

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert [identity.task_id for identity in getter_calls] == [200]
    assert generator._offered_task_count == 1
    assert generator._offer_limit_reached_logged is True


@pytest.mark.unit
def test_video_generator_cancels_eof_reservation_and_waits_for_reset(monkeypatch):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")

    getter_calls = []

    def data_getter(system, task_identity):
        getter_calls.append(task_identity)
        if len(getter_calls) == 2:
            raise StopGeneratorLoop
        return video_generator_module.DataGetterStatus.EXHAUSTED

    reset_states = iter([False, True])
    data_getter.datasource_reset_ready = lambda system: next(reset_states)

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
        source_id=11,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )
    identities = [
        TaskIdentity.create(source_id=11, task_id=300),
        TaskIdentity.create(source_id=11, task_id=301),
    ]
    identity_iter = iter(identities)
    monkeypatch.setattr(generator, "create_task_identity", lambda: next(identity_iter))

    schedule_requests = []

    def fake_request_schedule_policy(task_identity=None):
        schedule_requests.append(task_identity)
        return True

    monkeypatch.setattr(generator, "request_schedule_policy", fake_request_schedule_policy)
    cancellations = []
    monkeypatch.setattr(
        generator,
        "cancel_schedule_reservation",
        lambda task_identity: cancellations.append(task_identity) or True,
    )

    with pytest.raises(StopGeneratorLoop):
        generator.run()

    assert getter_calls == identities
    assert schedule_requests == identities
    assert cancellations == [identities[0]]


@pytest.mark.unit
def test_video_generator_async_submission_snapshots_payload_and_preserves_fifo(
    monkeypatch, tmp_path
):
    generator_module = importlib.import_module("core.generator.generator")
    video_generator_module = importlib.import_module("core.generator.video_generator")
    hooks = {
        "GEN_BSO": lambda system: {},
        "GEN_ASO": lambda system, response: None,
        "GEN_GETTER": lambda system, identity: True,
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
    monkeypatch.setenv("ASYNC_TASK_SUBMISSION", "true")
    monkeypatch.setenv("TASK_SUBMISSION_QUEUE_DEPTH", "2")
    monkeypatch.setenv("TASK_SUBMISSION_WORKERS", "2")
    generator = video_generator_module.VideoGenerator(
        source_id=10,
        source_url="http://source/video",
        source_metadata={"fps": 5},
        dag=build_dag_deployment(),
    )
    payload = tmp_path / "payload.mp4"
    payload.write_bytes(b"immutable-payload")
    submitted = []
    submit_done = threading.Event()

    def submit(current, *, file_path=None, file_content=None):
        submitted.append((current.get_task_id(), file_path, file_content))
        submit_done.set()
        return True

    monkeypatch.setattr(generator, "_submit_task_to_controller", submit)
    task = generator.generate_task(
        task_id=7,
        task_dag=Task.extract_dag_from_dag_deployment(build_dag_deployment()),
        service_deployment={"edge-node": ["face-detection"]},
        meta_data={"fps": 5},
        compressed_path=str(payload),
        hash_codes=[],
        task_identity=TaskIdentity.create(10, 7),
    )

    assert generator.submit_task_to_controller(task) is True
    payload.unlink()
    assert submit_done.wait(timeout=1.0)
    generator._submission_queue.join()

    assert generator.task_submission_workers == 2
    assert len(generator._submission_worker_threads) == 2
    assert submitted == [(7, None, b"immutable-payload")]
