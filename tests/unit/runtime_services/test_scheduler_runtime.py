import asyncio
import importlib
from types import SimpleNamespace

import pytest

from core.lib.content import Task
from core.lib.scheduling import build_schedule_decision
from core.scheduler.runtime_directory import RuntimeDirectoryStore
from core.scheduler.task_lease import InMemoryTaskLeaseStore


scheduler_module = importlib.import_module("core.scheduler.scheduler")


class FakeRuntimeContext:
    bootstrap = {}
    install_id = "test-install"
    directory_revision = 0
    cloud_node = "cloud-node"
    local_node = "edge-node"

    @staticmethod
    def resolve_static_endpoint(component, required=True, **kwargs):
        return None

    @staticmethod
    def edge_nodes():
        return ["edge-node"]


def build_task(source_id=7):
    dag_deployment = {
        "detector": {
            "service": {"service_name": "detector", "execute_device": "edge-node"},
            "next_nodes": [],
        }
    }
    return Task(
        source_id=source_id,
        task_id=1,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=Task.extract_dag_from_dag_deployment(dag_deployment),
        flow_index="detector",
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="payload.bin",
    )


@pytest.mark.unit
def test_schedule_decision_preserves_task_identity_creation_time():
    decision = build_schedule_decision(
        {
            "source_id": 7,
            "task_context": {
                "source_id": 7,
                "task_id": 3,
                "root_uuid": "root-3",
                "created_at": 12.5,
            },
        },
        {"dag": {}},
        deployment_version=1,
        runtime_directory_revision=2,
    )

    assert decision["created_at"] == 12.5


class FakeAgent:
    def __init__(self):
        self.schedule_plan = None
        self.scenarios = []
        self.policies = []
        self.tasks = []
        self.resources = []
        self.ran = False

    def run(self):
        self.ran = True

    def get_schedule_plan(self, info):
        return self.schedule_plan

    def update_scenario(self, scenario):
        self.scenarios.append(scenario)

    def update_policy(self, policy):
        self.policies.append(policy)

    def update_task(self, task):
        self.tasks.append(task)

    def update_resource(self, device, resource):
        self.resources.append((device, resource))

    def get_source_selection_plan(self, data):
        return {"selected": data["node_set"][0]}

    def get_initial_deployment_plan(self, data):
        return {service: list(data["node_set"]) for service in data["dag"]}

    def get_redeployment_plan(self, data):
        return {service: list(data["node_set"]) for service in data["dag"]}

    def should_generate(self, data):
        return {"generate": data.get("allow", True), "reason": "fake_agent"}

    def get_schedule_overhead(self):
        return 0.2


class DummyThread:
    def __init__(self, target=None):
        self.target = target
        self.started = False

    def start(self):
        self.started = True
        self.target()


@pytest.mark.unit
def test_scheduler_initializes_algorithms_and_handles_schedule_fallback(monkeypatch):
    config_calls = []
    agent = FakeAgent()

    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: config_calls.append("config")
        if name == "SCH_SCENARIO_RETRIEVAL":
            return lambda task: {"objects": task.get_task_id()}
        if name == "SCH_POLICY_RETRIEVAL":
            return lambda task: {"policy": task.get_source_id()}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {
                "dag": {
                    "detector": {"service": {"execute_device": "edge-node"}},
                    "_start": {"service": {"execute_device": "edge-node"}},
                }
            }
        if name == "SCH_AGENT":
            return agent
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(scheduler_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setattr(scheduler_module.threading, "Thread", DummyThread)

    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=RuntimeDirectoryStore(),
        task_lease_store=InMemoryTaskLeaseStore(),
    )
    scheduler.register_schedule_table(7)

    assert config_calls == ["config"]
    assert scheduler.cloud_device == "cloud-node"
    assert scheduler.schedule_table[7] is agent
    assert agent.ran is True

    plan = scheduler.get_schedule_plan({"source_id": 7})
    assert plan["dag"]["detector"]["service"]["execute_device"] == "edge-node"
    assert plan["dag"]["_start"]["service"]["execute_device"] == "edge-node"


@pytest.mark.unit
def test_scheduler_updates_scenarios_resources_and_supports_plans_and_overhead(monkeypatch):
    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: None
        if name == "SCH_SCENARIO_RETRIEVAL":
            return lambda task: {"scenario": task.get_task_id()}
        if name == "SCH_POLICY_RETRIEVAL":
            return lambda task: {"policy": task.get_source_id()}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {"dag": {}}
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(scheduler_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=RuntimeDirectoryStore(),
        task_lease_store=InMemoryTaskLeaseStore(),
    )
    agent_a = FakeAgent()
    agent_b = FakeAgent()
    scheduler.schedule_table = {1: agent_a, 2: agent_b}

    task = build_task(source_id=1)
    assert scheduler.update_scheduler_scenario(task) is True
    assert scheduler.update_scheduler_scenario(build_task(source_id=99)) is False

    scheduler.register_resource_table("edge-node")
    scheduler.register_resource_table("edge-node")
    scheduler.update_scheduler_resource({"device": "edge-node", "resource": {"cpu": 0.5}})

    assert agent_a.scenarios == [{"scenario": 1}]
    assert agent_a.policies == [{"policy": 1}]
    assert agent_a.tasks[0].get_source_id() == 1
    assert scheduler.resource_table == {"edge-node": {"cpu": 0.5}}
    assert agent_a.resources == [("edge-node", {"cpu": 0.5})]
    assert agent_b.resources == [("edge-node", {"cpu": 0.5})]
    assert scheduler.get_scheduler_resource() == {"edge-node": {"cpu": 0.5}}
    assert scheduler.get_source_node_selection_plan(1, {"node_set": ["edgex1"]}) == {"selected": "edgex1"}
    deployment_info = {"node_set": ["edgex1"], "dag": {"detector": {}}}
    assert scheduler.get_initial_deployment_plan(1, deployment_info) == {"detector": ["edgex1"]}
    assert scheduler.get_redeployment_plan(1, deployment_info) == {"detector": ["edgex1"]}
    assert scheduler.should_generate(1, {"allow": False}) == {"generate": False, "reason": "fake_agent"}
    assert scheduler.get_schedule_overhead() == 0.2


@pytest.mark.unit
def test_scheduler_snapshot_tracks_resources_and_active_task_commitments(monkeypatch):
    now = [100.0]
    barrier_requests = []

    class FakeBarrierStore:
        def snapshot(self, requests):
            barrier_requests.append(requests)
            return [
                {
                    "root_uuid": request["root_uuid"],
                    "barrier": request["barrier"],
                    "arrived_branches": [request["expected_branches"][0]],
                    "expected_branches": request["expected_branches"],
                    "required_count": request["required_count"],
                    "ready": False,
                    "expires_in_seconds": 20,
                }
                for request in requests
            ]

    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: None
        if name == "SCH_SCENARIO_RETRIEVAL":
            return lambda task: {}
        if name == "SCH_POLICY_RETRIEVAL":
            return lambda task: {}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {"dag": {}}
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(scheduler_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=RuntimeDirectoryStore(),
        task_lease_store=InMemoryTaskLeaseStore(clock=lambda: now[0]),
        task_barrier_store=FakeBarrierStore(),
    )
    scheduler._runtime_clock = lambda: now[0]
    scheduler.runtime_directory_revision = lambda: 1
    scheduler.update_scheduler_resource({
        "device": "edge-node",
        "runtime_directory_revision": 1,
        "resource": {"queue_state": {"detector": {"waiting_count": 2, "busy": False}}},
    })
    scheduler.acquire_task_lease(
        1,
        "root-1",
        ttl_seconds=20,
        commitment={
            "source_id": 3,
            "task_id": 8,
            "root_uuid": "root-1",
            "runtime_directory_revision": 1,
            "decision_id": "decision-1",
            "plan_digest": "digest-1",
            "dag": {
                "detector-a": {"prev_nodes": []},
                "detector-b": {"prev_nodes": []},
                "join": {"prev_nodes": ["detector-a", "detector-b"]},
            },
        },
    )
    scheduler.task_leases.reserve(
        2,
        "root-stale-reservation",
        {"root_uuid": "root-stale-reservation"},
        active_revision=2,
        ttl_seconds=20,
    )
    scheduler.task_leases.acquire(
        2,
        "root-old-revision",
        active_revision=2,
        ttl_seconds=20,
        context={
            "root_uuid": "root-old-revision",
            "runtime_directory_revision": 2,
            "dag": {"detector": {"prev_nodes": []}},
        },
    )

    snapshot = scheduler.get_scheduling_snapshot()
    assert snapshot["runtime_directory_revision"] == 1
    assert snapshot["deployment"] == {}
    assert snapshot["resource_runtime_revision"] == {"edge-node": 1}
    assert snapshot["reservations"] == []
    assert snapshot["resources"] == {
        "edge-node": {"queue_state": {"detector": {"waiting_count": 2, "busy": False}}}
    }
    assert snapshot["commitments"][0]["root_uuid"] == "root-1"
    assert len(snapshot["commitments"]) == 1
    assert snapshot["commitments"][0]["decision_id"] == "decision-1"
    assert snapshot["task_barriers"][0]["barrier"] == "join"
    assert barrier_requests[0] == [{
        "root_uuid": "root-1",
        "barrier": "join",
        "expected_branches": ["detector-a", "detector-b"],
        "required_count": 2,
    }]
    snapshot["resources"]["edge-node"]["queue_state"]["detector"]["waiting_count"] = 99
    assert scheduler.get_scheduler_resource()["edge-node"]["queue_state"]["detector"]["waiting_count"] == 2

    changed_commitment = {
        key: value for key, value in snapshot["commitments"][0].items()
        if key not in {"admitted_at", "expires_at", "status"}
    }
    changed_commitment["plan_digest"] = "digest-2"
    with pytest.raises(scheduler_module.RuntimeDirectoryConflict):
        scheduler.acquire_task_lease(
            1,
            "root-1",
            ttl_seconds=40,
            commitment=changed_commitment,
        )
    assert scheduler.get_scheduling_snapshot()["commitments"][0]["plan_digest"] == "digest-1"

    renewed = scheduler.renew_task_lease(1, "root-1", ttl_seconds=30)
    assert scheduler.get_scheduling_snapshot()["commitments"][0]["expires_at"] == renewed["expires_at"]
    scheduler.release_task_lease(1, "root-1")
    assert scheduler.get_scheduling_snapshot()["commitments"] == []


@pytest.mark.unit
def test_scheduler_resource_lock_passthrough_and_existing_registration(monkeypatch):
    lock_calls = []

    class FakeLockManager:
        async def acquire_lock(self, resource, device):
            lock_calls.append((resource, device))
            return device

    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: None
        if name == "SCH_SCENARIO_RETRIEVAL":
            return lambda task: {}
        if name == "SCH_POLICY_RETRIEVAL":
            return lambda task: {}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {"dag": {}}
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(scheduler_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    monkeypatch.setattr(scheduler_module, "ResourceLockManager", FakeLockManager)

    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=RuntimeDirectoryStore(),
        task_lease_store=InMemoryTaskLeaseStore(),
    )
    scheduler.schedule_table[5] = FakeAgent()
    scheduler.register_schedule_table(5)

    holder = asyncio.run(scheduler.get_resource_lock({"resource": "camera-0", "device": "edgex1"}))

    assert holder == "edgex1"
    assert lock_calls == [("camera-0", "edgex1")]
