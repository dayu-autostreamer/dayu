import importlib

import pytest

from core.lib.content import Task
from core.lib.runtime import RuntimeContext


def build_join_task(past_flow_index, current_flow_index="join", root_uuid="root-task-0", value=None):
    dag = Task.extract_dag_from_dag_deployment(
        {
            "detector-a": {
                "service": {"service_name": "detector-a", "execute_device": "edge-node"},
                "next_nodes": [current_flow_index],
            },
            "detector-b": {
                "service": {"service_name": "detector-b", "execute_device": "edge-node"},
                "next_nodes": [current_flow_index],
            },
            current_flow_index: {
                "service": {"service_name": current_flow_index, "execute_device": "edge-node"},
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
        flow_index=current_flow_index,
        past_flow_index=past_flow_index,
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="payload.bin",
        root_uuid=root_uuid,
        runtime_directory_revision=1,
    )
    task.set_tmp_data({"value": value})
    return task


class FakeRedis:
    def __init__(self):
        self.storage = {}
        self.expiry = {}
        self.deleted = []
        self.fail_eval = None
        self.fail_delete = None

    def eval(self, script, num_keys, storage_key, branch, payload, timeout, required_count):
        if self.fail_eval:
            raise self.fail_eval
        values = self.storage.setdefault(storage_key, {})
        values[branch] = payload
        self.expiry[storage_key] = timeout
        if len(values) < required_count:
            return []
        result = []
        for field, value in values.items():
            result.extend([field, value])
        return result

    def delete(self, storage_key):
        if self.fail_delete:
            raise self.fail_delete
        self.deleted.append(storage_key)
        self.storage.pop(storage_key, None)


def build_coordinator(redis_client, storage_timeout=3600):
    module = importlib.import_module("core.controller.task_coordinator")
    coordinator = object.__new__(module.TaskCoordinator)
    coordinator.redis = redis_client
    coordinator.storage_timeout = storage_timeout
    coordinator.joint_service_key_prefix = "dayu:dag:joint_service"
    return coordinator


@pytest.mark.unit
def test_arrive_is_idempotent_by_predecessor_and_returns_ready_tasks():
    redis_client = FakeRedis()
    coordinator = build_coordinator(redis_client)
    first = build_join_task("detector-a", value="old")
    repeated = build_join_task("detector-a", value="new")
    second = build_join_task("detector-b", value="right")

    assert coordinator.arrive(first, "join", required_count=2) is None
    assert coordinator.arrive(repeated, "join", required_count=2) is None
    ready = coordinator.arrive(second, "join", required_count=2)

    storage_key = f"dayu:dag:joint_service:{first.get_root_uuid()}:join"
    assert set(redis_client.storage[storage_key]) == {"detector-a", "detector-b"}
    assert redis_client.expiry[storage_key] == 3600
    assert {task.get_past_flow_index() for task in ready} == {"detector-a", "detector-b"}
    assert next(task for task in ready if task.get_past_flow_index() == "detector-a").get_tmp_data() == {
        "value": "new"
    }


@pytest.mark.unit
def test_ready_barrier_is_retained_until_downstream_completion():
    redis_client = FakeRedis()
    coordinator = build_coordinator(redis_client)
    left = build_join_task("detector-a")
    right = build_join_task("detector-b")

    assert coordinator.arrive(left, "join", 2) is None
    assert coordinator.arrive(right, "join", 2)
    storage_key = f"dayu:dag:joint_service:{left.get_root_uuid()}:join"
    assert storage_key in redis_client.storage

    coordinator.complete(left.get_root_uuid(), "join")

    assert storage_key not in redis_client.storage
    assert redis_client.deleted == [storage_key]


@pytest.mark.unit
def test_arrive_rejects_corrupt_or_conflicting_barriers():
    module = importlib.import_module("core.controller.task_coordinator")
    redis_client = FakeRedis()
    coordinator = build_coordinator(redis_client)
    left = build_join_task("detector-a", current_flow_index="join")
    conflicting = build_join_task("detector-b", current_flow_index="other-join")

    redis_client.eval = lambda *args: [
        "detector-a",
        left.serialize(),
        "detector-b",
        conflicting.serialize(),
    ]

    with pytest.raises(module.TaskCoordinationError, match="conflicting services"):
        coordinator.arrive(left, "join", 2)


@pytest.mark.unit
def test_coordinator_uses_runtime_lease_ttl_and_normalizes_redis_pool(monkeypatch):
    module = importlib.import_module("core.controller.task_coordinator")
    pool_calls = []
    monkeypatch.setattr(
        module.Context,
        "get_parameter",
        staticmethod(lambda name, default=None, direct=False: "12" if name == "MAX_REDIS_CONNECTIONS" else default),
    )
    monkeypatch.setattr(
        module.redis,
        "ConnectionPool",
        lambda **kwargs: pool_calls.append(kwargs) or {"pool": kwargs},
    )
    monkeypatch.setattr(module.redis, "Redis", lambda connection_pool: {"redis": connection_pool})
    runtime_context = RuntimeContext({
        "cloud_node": "cloudx1",
        "lease_ttl_seconds": 900.2,
        "endpoints": {"redis": {"fqdn": "redis.dayu.svc.cluster.local", "port": 6379}},
    })

    coordinator = module.TaskCoordinator(runtime_context=runtime_context)

    assert coordinator.max_connections == 12
    assert coordinator.storage_timeout == 901
    assert pool_calls == [{
        "host": "redis.dayu.svc.cluster.local.",
        "port": 6379,
        "max_connections": 12,
    }]


@pytest.mark.unit
def test_redis_failures_propagate_instead_of_becoming_false_waits():
    module = importlib.import_module("core.controller.task_coordinator")
    redis_client = FakeRedis()
    coordinator = build_coordinator(redis_client)
    task = build_join_task("detector-a")
    redis_client.fail_eval = RuntimeError("redis unavailable")

    with pytest.raises(module.TaskCoordinationError, match="failed to retain"):
        coordinator.arrive(task, "join", 2)

    redis_client.fail_eval = None
    redis_client.fail_delete = RuntimeError("redis unavailable")
    with pytest.raises(module.TaskCoordinationError, match="failed to complete"):
        coordinator.complete(task.get_root_uuid(), "join")
