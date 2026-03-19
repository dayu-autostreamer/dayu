import asyncio
import importlib
from types import SimpleNamespace

import pytest

from core.lib.content import Task


scheduler_module = importlib.import_module("core.scheduler.scheduler")


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
        return {"initial": data["node_set"]}

    def get_redeployment_plan(self, data):
        return {"redeploy": data["node_set"]}

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
def test_scheduler_initializes_algorithms_and_handles_schedule_fallback_and_backup(monkeypatch):
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
    monkeypatch.setattr(scheduler_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(scheduler_module.KubeConfig, "check_services_running", staticmethod(lambda: False))
    monkeypatch.setattr(scheduler_module.threading, "Thread", DummyThread)

    scheduler = scheduler_module.Scheduler()
    scheduler.register_schedule_table(7)

    assert config_calls == ["config"]
    assert scheduler.cloud_device == "cloud-node"
    assert scheduler.schedule_table[7] is agent
    assert agent.ran is True

    plan = scheduler.get_schedule_plan({"source_id": 7})
    assert plan["dag"]["detector"]["service"]["execute_device"] == "cloud-node"
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
    monkeypatch.setattr(scheduler_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))

    scheduler = scheduler_module.Scheduler()
    agent_a = FakeAgent()
    agent_b = FakeAgent()
    scheduler.schedule_table = {1: agent_a, 2: agent_b}

    task = build_task(source_id=1)
    scheduler.update_scheduler_scenario(task)
    scheduler.update_scheduler_scenario(build_task(source_id=99))

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
    assert scheduler.get_initial_deployment_plan(1, {"node_set": ["edgex1"]}) == {"initial": ["edgex1"]}
    assert scheduler.get_redeployment_plan(1, {"node_set": ["edgex1"]}) == {"redeploy": ["edgex1"]}
    assert scheduler.get_schedule_overhead() == 0.2


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
    monkeypatch.setattr(scheduler_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(scheduler_module, "ResourceLockManager", FakeLockManager)

    scheduler = scheduler_module.Scheduler()
    scheduler.schedule_table[5] = FakeAgent()
    scheduler.register_schedule_table(5)

    holder = asyncio.run(scheduler.get_resource_lock({"resource": "camera-0", "device": "edgex1"}))

    assert holder == "edgex1"
    assert lock_calls == [("camera-0", "edgex1")]
