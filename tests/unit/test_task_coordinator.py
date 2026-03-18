import importlib

import pytest

from core.lib.content import Task


def build_join_task(past_flow_index, current_flow_index="join", root_uuid="root-task-0"):
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
    return Task(
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
    )


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakePipeline:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.storage_key = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def hset(self, storage_key, task_uuid, payload):
        self.storage_key = storage_key
        self.redis_client.storage.setdefault(storage_key, {})[task_uuid] = payload

    def expire(self, storage_key, timeout):
        self.redis_client.expiry[storage_key] = timeout

    def hlen(self, storage_key):
        self.storage_key = storage_key

    def execute(self):
        count = len(self.redis_client.storage.get(self.storage_key, {}))
        return 1, 1, count


class FakeRedis:
    def __init__(self, eval_result=None):
        self.eval_result = eval_result
        self.storage = {}
        self.expiry = {}
        self.eval_calls = []
        self.lock_calls = []

    def pipeline(self):
        return FakePipeline(self)

    def lock(self, key, timeout):
        self.lock_calls.append((key, timeout))
        return FakeLock()

    def eval(self, script, num_keys, storage_key, required_count):
        self.eval_calls.append((storage_key, required_count))
        return self.eval_result


def build_eval_result(*tasks):
    payload = []
    for index, task in enumerate(tasks):
        payload.extend([f"task-{index}", task.serialize()])
    return payload


def build_coordinator(redis_client):
    task_coordinator_module = importlib.import_module("core.controller.task_coordinator")
    coordinator = object.__new__(task_coordinator_module.TaskCoordinator)
    coordinator.redis = redis_client
    coordinator.storage_timeout = 3600
    coordinator.lock_prefix = "dayu:dag:lock"
    coordinator.joint_service_key_prefix = "dayu:dag:joint_service"
    return coordinator


@pytest.mark.unit
def test_store_task_data_persists_serialized_task_and_returns_count():
    task = build_join_task("detector-a")
    redis_client = FakeRedis()
    coordinator = build_coordinator(redis_client)

    count = coordinator.store_task_data(task, "join")

    storage_key = f"dayu:dag:joint_service:{task.get_root_uuid()}:join"
    assert count == 1
    assert storage_key in redis_client.storage
    assert task.get_task_uuid() in redis_client.storage[storage_key]
    assert redis_client.expiry[storage_key] == 3600


@pytest.mark.unit
def test_retrieve_task_data_returns_tasks_for_distinct_parallel_branches():
    left_task = build_join_task("detector-a")
    right_task = build_join_task("detector-b")
    redis_client = FakeRedis(build_eval_result(left_task, right_task))
    coordinator = build_coordinator(redis_client)

    tasks = coordinator.retrieve_task_data(left_task.get_root_uuid(), "join", required_count=2)

    assert [task.get_past_flow_index() for task in tasks] == ["detector-a", "detector-b"]
    assert redis_client.lock_calls == [(f"dayu:dag:lock:{left_task.get_root_uuid()}:join", 10)]
    assert redis_client.eval_calls == [(f"dayu:dag:joint_service:{left_task.get_root_uuid()}:join", 2)]


@pytest.mark.unit
def test_retrieve_task_data_rejects_duplicate_parallel_branch_inputs():
    first_task = build_join_task("detector-a")
    duplicate_task = build_join_task("detector-a")
    redis_client = FakeRedis(build_eval_result(first_task, duplicate_task))
    coordinator = build_coordinator(redis_client)

    tasks = coordinator.retrieve_task_data(first_task.get_root_uuid(), "join", required_count=2)

    assert tasks is None


@pytest.mark.unit
def test_retrieve_task_data_rejects_conflicting_joint_service_names():
    first_task = build_join_task("detector-a", current_flow_index="join")
    conflicting_task = build_join_task("detector-b", current_flow_index="other-join")
    redis_client = FakeRedis(build_eval_result(first_task, conflicting_task))
    coordinator = build_coordinator(redis_client)

    tasks = coordinator.retrieve_task_data(first_task.get_root_uuid(), "join", required_count=2)

    assert tasks is None
