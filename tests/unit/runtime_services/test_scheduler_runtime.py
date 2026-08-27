import asyncio
import importlib
import threading
from types import SimpleNamespace

import pytest

from core.lib.content import Task
from core.lib.scheduling import build_schedule_decision, SchedulingSnapshotScope
from core.scheduler.runtime_directory import RuntimeDirectoryStore
from core.scheduler.task_lease import InMemoryTaskLeaseStore


scheduler_module = importlib.import_module("core.scheduler.scheduler")
scheduler_server_module = importlib.import_module("core.scheduler.scheduler_server")


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
def test_schedule_decision_contains_only_scheduler_owned_identity():
    decision = build_schedule_decision(
        {
            "source_id": 7,
            "task_context": {
                "source_id": 7,
                "task_id": 3,
                "root_uuid": "root-3",
            },
        },
        {"dag": {}},
        deployment_version=1,
        runtime_directory_revision=2,
    )

    assert decision["source_id"] == 7
    assert decision["task_id"] == 3
    assert decision["root_uuid"] == "root-3"
    assert "created_at" not in decision


@pytest.mark.unit
def test_scheduler_completes_each_partial_plan_from_current_state():
    current_dag = {
        "detector": {
            "service": {
                "service_name": "detector",
                "execute_device": "edge-node",
            },
            "next_nodes": [],
        }
    }
    startup_dag = {
        "detector": {
            "service": {
                "service_name": "detector",
                "execute_device": "cloud-node",
            },
            "next_nodes": [],
        }
    }
    server = object.__new__(scheduler_server_module.SchedulerServer)
    server.scheduler = SimpleNamespace(
        cloud_device="cloud-node",
        get_startup_policy=lambda request: {"dag": startup_dag},
    )
    request = {
        "current_configuration": {"fps": 10, "buffer_size": 2},
        "dag": current_dag,
    }

    configuration_only = server._complete_schedule_plan({"fps": 5}, request)
    assert configuration_only == {
        "fps": 5,
        "buffer_size": 2,
        "dag": current_dag,
    }

    offloading_only = server._complete_schedule_plan(
        {"dag": startup_dag},
        request,
    )
    assert offloading_only == {
        "fps": 10,
        "buffer_size": 2,
        "dag": startup_dag,
    }

    unrouted_request = {**request, "dag": {
        "detector": {
            "service": {"service_name": "detector", "execute_device": ""},
            "next_nodes": [],
        }
    }}
    assert server._complete_schedule_plan({}, unrouted_request)["dag"] == startup_dag
    assert server._dag_has_runtime_targets({}) is False
    assert server._split_schedule_plan({}, 7) == ({}, 7)
    assert server._split_schedule_plan({"deployment_version": None}, 7) == ({}, 7)
    with pytest.raises(scheduler_server_module.HTTPException) as exc_info:
        server._complete_schedule_plan({}, {**request, "current_configuration": []})
    assert "current_configuration" in str(exc_info.value.detail)


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
def test_scheduler_initializes_algorithms_and_preserves_no_update(monkeypatch):
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

    assert scheduler.get_schedule_plan({"source_id": 7}) == {}


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
    scheduler.runtime_directory_revision = lambda: 1
    assert scheduler.should_generate(1, {"allow": False}) == {
        "generate": False,
        "reason": "fake_agent",
        "runtime_directory_revision": 1,
    }
    assert scheduler.get_schedule_overhead() == 0.2


@pytest.mark.unit
def test_scheduler_blocks_generation_before_first_runtime_directory_commit(monkeypatch):
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
        task_lease_store=InMemoryTaskLeaseStore(),
    )
    scheduler.schedule_table = {1: FakeAgent()}

    assert scheduler.should_generate(1, {}) == {
        "generate": False,
        "reason": "runtime_directory_not_ready",
        "runtime_directory_revision": 0,
    }


@pytest.mark.unit
def test_scheduler_attaches_runtime_context_without_interpreting_actions():
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    opaque_action = {
        "type": "extension_defined_action",
        "payload": {"value": 7},
    }
    scheduler.schedule_table = {
        1: SimpleNamespace(
            should_generate=lambda data: {
                "generate": False,
                "reason": "extension_guard",
                "actions": [opaque_action],
            }
        )
    }
    scheduler.runtime_directory_revision = lambda: 5
    scheduler.runtime_directory_snapshot = lambda: {
        "revision": 5,
        "hash": "directory-hash",
        "routes": [{"slot": {"component": "custom"}}],
    }

    decision = scheduler.should_generate(1, {})

    assert decision["actions"] == [opaque_action]
    assert decision["runtime_directory_revision"] == 5
    assert decision["runtime_directory"] == {
        "revision": 5,
        "hash": "directory-hash",
        "routes": [{"slot": {"component": "custom"}}],
    }


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

    live_snapshot = scheduler.get_scheduling_snapshot(
        SchedulingSnapshotScope.LIVE
    )
    assert live_snapshot["resources"] == {
        "edge-node": {
            "queue_state": {
                "detector": {"waiting_count": 2, "busy": False},
            },
        },
    }
    assert live_snapshot["reservations"] == []
    assert live_snapshot["commitments"] == []
    assert live_snapshot["task_barriers"] == []
    assert barrier_requests == []

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
def test_scheduler_stages_pre_admission_plan_locally_until_lease_acquire(monkeypatch):
    now = [100.0]

    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: None
        if name in {"SCH_SCENARIO_RETRIEVAL", "SCH_POLICY_RETRIEVAL"}:
            return lambda task: {}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {"dag": {}}
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(
        scheduler_module.Context,
        "get_algorithm",
        staticmethod(fake_get_algorithm),
    )
    leases = InMemoryTaskLeaseStore(clock=lambda: now[0])
    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=RuntimeDirectoryStore(),
        task_lease_store=leases,
    )
    scheduler._runtime_clock = lambda: now[0]
    scheduler.runtime_directory_revision = lambda: 1
    scheduler.runtime_service_nodes = lambda: {"detector": ["edge-node"]}
    context = {
        "source_id": 3,
        "task_id": 8,
        "root_uuid": "root-staged",
        "runtime_directory_revision": 1,
        "decision_id": "decision-staged",
        "plan_digest": "digest-staged",
        "deployment_version": 2,
        "plan": {"detector": "edge-node"},
    }

    staged = scheduler.stage_task_context(1, "root-staged", context, ttl_seconds=20)

    assert staged["status"] == "pending"
    assert leases.list_reservations() == []
    assert scheduler.get_task_reservation(
        1,
        "root-staged",
        {"source_id": 3, "task_id": 8},
    )["plan"] == {
        "detector": "edge-node"
    }
    assert scheduler.get_scheduling_snapshot()["reservations"][0]["root_uuid"] == (
        "root-staged"
    )

    with pytest.raises(
        scheduler_module.RuntimeDirectoryConflict,
        match="does not match its reservation",
    ):
        scheduler.acquire_task_lease(
            1,
            "root-staged",
            ttl_seconds=20,
            commitment={**context, "plan_digest": "different"},
        )

    lease = scheduler.acquire_task_lease(
        1,
        "root-staged",
        ttl_seconds=20,
        commitment=context,
    )
    assert lease["root_uuid"] == "root-staged"
    assert scheduler.get_task_reservation(
        1,
        "root-staged",
        {"source_id": 3, "task_id": 8},
    ) is None
    assert scheduler.get_scheduling_snapshot()["reservations"] == []
    assert scheduler.get_scheduling_snapshot()["commitments"][0]["root_uuid"] == (
        "root-staged"
    )


@pytest.mark.unit
def test_slow_lease_admission_does_not_block_schedule_or_staging_state(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class BlockingLeaseStore(InMemoryTaskLeaseStore):
        def acquire(self, *args, **kwargs):
            entered.set()
            assert release.wait(timeout=2.0)
            return super().acquire(*args, **kwargs)

    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: None
        if name in {"SCH_SCENARIO_RETRIEVAL", "SCH_POLICY_RETRIEVAL"}:
            return lambda task: {}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {"dag": {}}
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(
        scheduler_module.Context,
        "get_algorithm",
        staticmethod(fake_get_algorithm),
    )
    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=RuntimeDirectoryStore(),
        task_lease_store=BlockingLeaseStore(),
    )
    scheduler.runtime_directory_revision = lambda: 1
    commitment = {
        "source_id": 3,
        "task_id": 8,
        "root_uuid": "root-blocked",
        "runtime_directory_revision": 1,
        "decision_id": "decision-blocked",
        "plan_digest": "digest-blocked",
        "deployment_version": 2,
        "plan": {"detector": "edge-node"},
    }
    scheduler.stage_task_context(1, "root-blocked", commitment)

    admission_done = threading.Event()

    def acquire():
        try:
            scheduler.acquire_task_lease(
                1,
                "root-blocked",
                ttl_seconds=20,
                commitment=commitment,
            )
        finally:
            admission_done.set()

    admission = threading.Thread(target=acquire)
    admission.start()
    assert entered.wait(timeout=1.0)

    state_done = threading.Event()

    def update_local_state():
        scheduler.register_resource_table("edge-node")
        scheduler.stage_task_context(
            1,
            "root-next",
            {
                **commitment,
                "task_id": 9,
                "root_uuid": "root-next",
                "decision_id": "decision-next",
                "plan_digest": "digest-next",
            },
        )
        state_done.set()

    state_update = threading.Thread(target=update_local_state)
    state_update.start()
    assert state_done.wait(timeout=0.2), (
        "durable lease I/O must not hold Scheduler local state locks"
    )

    release.set()
    assert admission_done.wait(timeout=1.0)
    admission.join(timeout=1.0)
    state_update.join(timeout=1.0)
    assert scheduler.resource_table["edge-node"] == {}
    assert scheduler.get_task_reservation(
        1,
        "root-next",
        {"source_id": 3, "task_id": 9},
    ) is not None


@pytest.mark.unit
def test_scheduler_runtime_snapshot_cache_tracks_owned_directory_changes(monkeypatch):
    reads = []

    class CountingDirectory(RuntimeDirectoryStore):
        def snapshot_model(self):
            reads.append("read")
            return super().snapshot_model()

    def fake_get_algorithm(name, **kwargs):
        if name == "SCH_CONFIG_EXTRACTION":
            return lambda scheduler: None
        if name in {"SCH_SCENARIO_RETRIEVAL", "SCH_POLICY_RETRIEVAL"}:
            return lambda task: {}
        if name == "SCH_STARTUP_POLICY":
            return lambda info: {"dag": {}}
        raise AssertionError(f"Unexpected algorithm request: {name}")

    monkeypatch.setattr(
        scheduler_module.Context,
        "get_algorithm",
        staticmethod(fake_get_algorithm),
    )
    directory = CountingDirectory()
    scheduler = scheduler_module.Scheduler(
        runtime_context=FakeRuntimeContext(),
        runtime_directory=directory,
        task_lease_store=InMemoryTaskLeaseStore(),
    )

    assert scheduler.runtime_directory_revision() == 0
    assert scheduler.runtime_directory_revision() == 0
    assert reads == ["read"]

    scheduler.replace_runtime_directory(
        {"install_id": "test-install", "revision": 1, "routes": []},
        expected_revision=0,
    )
    assert scheduler.runtime_directory_revision() == 1
    assert reads == ["read"]


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
