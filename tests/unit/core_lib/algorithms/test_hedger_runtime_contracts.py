import copy
import types

import pytest

pytestmark = pytest.mark.ml
pytest.importorskip(
    "torch",
    reason="Hedger runtime tests require the real PyTorch runtime",
    exc_type=ModuleNotFoundError,
)

from core.lib.algorithms.schedule_agent.hedger_agent import HedgerAgent
from core.lib.algorithms.schedule_agent.hedger_deployment_only_agent import (
    HedgerDeploymentOnlyAgent,
)
from core.lib.algorithms.schedule_initial_deployment_policy.hedger_flat_initial_deployment_policy import (
    HedgerFlatInitialDeploymentPolicy,
)
from core.lib.algorithms.schedule_initial_deployment_policy.hedger_initial_deployment_policy import (
    HedgerInitialDeploymentPolicy,
)
from core.lib.algorithms.schedule_initial_deployment_policy.hedger_offloading_only_initial_deployment_policy import (
    HedgerOffloadingOnlyInitialDeploymentPolicy,
)
from core.lib.algorithms.schedule_redeployment_policy.hedger_flat_redeployment_policy import (
    HedgerFlatRedeploymentPolicy,
)
from core.lib.algorithms.schedule_redeployment_policy.hedger_redeployment_policy import (
    HedgerRedeploymentPolicy,
)
from core.lib.algorithms.schedule_redeployment_policy.hedger_offloading_only_redeployment_policy import (
    HedgerOffloadingOnlyRedeploymentPolicy,
)
from core.lib.scheduling import SchedulingSnapshotScope


def _dag():
    return {
        "svc-a": {
            "service": {"service_name": "svc-a"},
            "next_nodes": ["svc-b"],
        },
        "svc-b": {
            "service": {"service_name": "svc-b"},
            "next_nodes": [],
        },
    }


def _deployment_info():
    return {
        "source": {"id": 1, "source_device": "edge-a"},
        "dag": _dag(),
        "node_set": ["edge-a", "edge-b"],
    }


def _snapshot_system(runtime):
    def get_scheduling_snapshot(scope):
        assert scope is SchedulingSnapshotScope.LIVE
        return copy.deepcopy(runtime)

    return types.SimpleNamespace(get_scheduling_snapshot=get_scheduling_snapshot)


class _ScheduleHedgerStub:
    def __init__(self):
        self.version = 1
        self.state_buffer = None

    def register_logical_topology(self, _dag_value):
        return None

    def register_physical_topology(self, _edge_nodes, _source_device):
        return None

    def register_state_buffer(self):
        return None

    def get_offloading_plan(self):
        return {"svc-a": "edge-b", "svc-b": "cloud-x"}

    def get_active_deployment_version(self):
        return self.version


def _schedule_agent(hedger, runtime, default_offloading=None):
    agent = HedgerAgent.__new__(HedgerAgent)
    agent.cloud_device = "cloud-x"
    agent.default_configuration = None
    agent.default_offloading = default_offloading
    agent.hedger = hedger
    agent.system = _snapshot_system(runtime)
    return agent


def _schedule_info(dag=None):
    return {
        "source_id": 1,
        "source_device": "edge-a",
        "all_edge_devices": ["edge-a", "edge-b"],
        "dag": dag or _dag(),
    }


@pytest.mark.unit
def test_schedule_uses_live_routes_and_live_decision_version():
    runtime = {
        "runtime_directory_revision": 7,
        "deployment": {
            "svc-a": ["edge-a", "cloud-x"],
            "svc-b": ["cloud-x"],
        },
        "resource_runtime_revision": {},
    }
    hedger = _ScheduleHedgerStub()
    hedger._deployment_plan_history = {
        0: {"svc-a": ["edge-a"], "svc-b": ["cloud-x"]},
        1: {"svc-a": ["edge-b"], "svc-b": ["cloud-x"]},
    }
    agent = _schedule_agent(
        hedger,
        runtime,
        default_offloading={"svc-a": "edge-a", "svc-b": "cloud-x"},
    )
    original_dag = _dag()

    old_runtime_policy = agent.get_schedule_plan(_schedule_info(original_dag))

    assert old_runtime_policy["dag"]["svc-a"]["service"]["execute_device"] == "edge-a"
    assert old_runtime_policy["deployment_version"] == 0
    assert "execute_device" not in original_dag["svc-a"]["service"]

    runtime["runtime_directory_revision"] = 8
    runtime["deployment"]["svc-a"] = ["edge-b", "cloud-x"]
    agent.system = _snapshot_system(runtime)
    new_runtime_policy = agent.get_schedule_plan(_schedule_info(original_dag))

    assert new_runtime_policy["dag"]["svc-a"]["service"]["execute_device"] == "edge-b"
    assert new_runtime_policy["deployment_version"] == 1


@pytest.mark.unit
def test_schedule_does_not_invent_a_cloud_route():
    runtime = {
        "runtime_directory_revision": 1,
        "deployment": {
            "svc-a": ["edge-a"],
            "svc-b": ["edge-b"],
        },
        "resource_runtime_revision": {},
    }
    hedger = _ScheduleHedgerStub()
    hedger.get_offloading_plan = lambda: {}
    agent = _schedule_agent(hedger, runtime)

    policy = agent.get_schedule_plan(_schedule_info())

    assert policy["dag"]["svc-a"]["service"]["execute_device"] == "edge-a"
    assert policy["dag"]["svc-b"]["service"]["execute_device"] == "edge-b"


@pytest.mark.unit
def test_schedule_rejects_a_service_without_an_active_route():
    runtime = {
        "runtime_directory_revision": 1,
        "deployment": {"svc-a": ["edge-a"]},
        "resource_runtime_revision": {},
    }
    agent = _schedule_agent(_ScheduleHedgerStub(), runtime)

    with pytest.raises(ValueError, match="svc-b.*no processor route"):
        agent.get_schedule_plan(_schedule_info())


@pytest.mark.unit
def test_schedule_requires_the_live_snapshot_api():
    agent = _schedule_agent(
        _ScheduleHedgerStub(),
        {
            "runtime_directory_revision": 1,
            "deployment": {"svc-a": ["edge-a"], "svc-b": ["edge-b"]},
            "resource_runtime_revision": {},
        },
    )
    agent.system = types.SimpleNamespace(
        runtime_service_nodes=lambda: {"svc-a": ["edge-a"], "svc-b": ["edge-b"]},
    )

    with pytest.raises(AttributeError, match="get_scheduling_snapshot"):
        agent.get_schedule_plan(_schedule_info())


@pytest.mark.unit
def test_schedule_rejects_a_scalar_live_route():
    agent = _schedule_agent(
        _ScheduleHedgerStub(),
        {
            "runtime_directory_revision": 1,
            "deployment": {"svc-a": "edge-a", "svc-b": ["edge-b"]},
            "resource_runtime_revision": {},
        },
    )

    with pytest.raises(ValueError, match="svc-a.*must be a node list"):
        agent.get_schedule_plan(_schedule_info())


@pytest.mark.unit
def test_schedule_rejects_runtime_routes_without_a_served_decision():
    hedger = _ScheduleHedgerStub()
    hedger._deployment_plan_history = {
        1: {"svc-a": ["edge-b"], "svc-b": ["cloud-x"]},
    }
    agent = _schedule_agent(
        hedger,
        {
            "runtime_directory_revision": 1,
            "deployment": {"svc-a": ["edge-a"], "svc-b": ["cloud-x"]},
            "resource_runtime_revision": {},
        },
    )

    with pytest.raises(ValueError, match="cannot bind the active runtime routes"):
        agent.get_schedule_plan(_schedule_info())


@pytest.mark.unit
def test_deployment_only_schedule_uses_live_routes():
    hedger = _ScheduleHedgerStub()
    hedger.get_heuristic_offloading_plan = lambda default_offloading=None: {
        "svc-a": "edge-b",
        "svc-b": "edge-b",
    }
    agent = HedgerDeploymentOnlyAgent.__new__(HedgerDeploymentOnlyAgent)
    agent.cloud_device = "cloud-x"
    agent.default_configuration = None
    agent.default_offloading = {"svc-a": "edge-a", "svc-b": "edge-b"}
    agent.hedger = hedger
    agent.system = _snapshot_system({
        "runtime_directory_revision": 1,
        "deployment": {
            "svc-a": ["edge-a"],
            "svc-b": ["edge-b"],
        },
        "resource_runtime_revision": {},
    })

    policy = agent.get_schedule_plan(_schedule_info())

    assert policy["dag"]["svc-a"]["service"]["execute_device"] == "edge-a"
    assert policy["dag"]["svc-b"]["service"]["execute_device"] == "edge-b"


@pytest.mark.unit
def test_resource_hook_accepts_queue_state_only_from_the_live_revision():
    calls = []
    runtime = {
        "runtime_directory_revision": 2,
        "deployment": {"svc-a": ["edge-a"]},
        "resource_runtime_revision": {"edge-a": 1},
    }
    buffer = types.SimpleNamespace(
        add_bandwidths=lambda value: calls.append(("bandwidth", value)),
        add_queue_lengths=lambda device, values, deployment_version=None: calls.append(
            ("queue", device, values, deployment_version)
        ),
    )
    hedger = types.SimpleNamespace(
        state_buffer=buffer,
        get_active_deployment_version=lambda: 2,
        update_latency_guard_queue_lengths=lambda device, values: calls.append(
            ("guard", device, values)
        ),
    )
    agent = HedgerAgent.__new__(HedgerAgent)
    agent.hedger = hedger
    agent.system = _snapshot_system(runtime)
    resource = {
        "available_bandwidth": 12.5,
        "queue_state": {"svc-a": {"waiting_count": 3}},
    }

    agent.update_resource("edge-a", resource)
    assert calls == [("bandwidth", 12.5)]

    runtime["resource_runtime_revision"]["edge-a"] = 2
    agent.system = _snapshot_system(runtime)
    agent.update_resource("edge-a", resource)
    assert calls == [
        ("bandwidth", 12.5),
        ("bandwidth", 12.5),
        ("guard", "edge-a", {"svc-a": 3.0}),
        ("queue", "edge-a", {"svc-a": 3.0}, 2),
    ]


class _DeploymentHedgerStub:
    def __init__(self, plan=None, version=0):
        self.plan = copy.deepcopy(plan)
        self.version = version
        self.registered = []

    def register_logical_topology(self, _dag_value):
        return None

    def register_physical_topology(self, _nodes, _source):
        return None

    def register_state_buffer(self):
        return None

    def register_initial_deployment(self, plan):
        self.registered.append(copy.deepcopy(plan))

    def get_initial_deployment_plan(self):
        if self.plan is None:
            return copy.deepcopy(self.registered[-1])
        return copy.deepcopy(self.plan)

    def get_redeployment_plan(self):
        return copy.deepcopy(self.plan)

    def set_heuristic_deployment_plan(
            self,
            info=None,
            default_deployment=None,
            mark_version=False,
    ):
        del info, mark_version
        if self.plan is None:
            return copy.deepcopy(default_deployment)
        return copy.deepcopy(self.plan)

    def get_active_deployment_version(self):
        return self.version


_INITIAL_POLICY_CLASSES = (
    HedgerInitialDeploymentPolicy,
    HedgerFlatInitialDeploymentPolicy,
    HedgerOffloadingOnlyInitialDeploymentPolicy,
)


_POLICY_CLASSES = (
    *_INITIAL_POLICY_CLASSES,
    HedgerRedeploymentPolicy,
    HedgerFlatRedeploymentPolicy,
    HedgerOffloadingOnlyRedeploymentPolicy,
)


def _deployment_policy(policy_class, hedger, default_plan):
    policy = policy_class.__new__(policy_class)
    policy.system = types.SimpleNamespace(cloud_device="cloud-x")
    policy.default_deployment = copy.deepcopy(default_plan)
    policy.hedger = hedger
    return policy


@pytest.mark.unit
@pytest.mark.parametrize("policy_class", _POLICY_CLASSES)
@pytest.mark.parametrize(
    "plan,message",
    (
        (
            {"svc-a": "edge-a", "svc-b": ["edge-b"]},
            "JSON node list",
        ),
        (
            {
                "svc-a": ["edge-a"],
                "svc-b": ["edge-b"],
                "old-service": ["edge-a"],
            },
            "outside the current DAG",
        ),
        (
            {"svc-a": ["edge-a"]},
            "omitted current DAG services",
        ),
        (
            {"svc-a": ["edge-a"], "svc-b": ["removed-edge"]},
            "selected non-candidate nodes",
        ),
        (
            {"svc-a": ["edge-a"], "svc-b": []},
            "returned no target nodes",
        ),
        (
            {"svc-a": ["edge-a"], "svc-b": [""]},
            "returned an empty node name",
        ),
    ),
)
def test_deployment_hooks_reject_invalid_plan_shapes(policy_class, plan, message):
    default_plan = {"svc-a": ["edge-a"], "svc-b": ["edge-b"]}
    policy = _deployment_policy(
        policy_class,
        _DeploymentHedgerStub(plan=plan, version=4),
        default_plan,
    )

    with pytest.raises(ValueError, match=message):
        policy(_deployment_info())


@pytest.mark.unit
@pytest.mark.parametrize("policy_class", _POLICY_CLASSES)
def test_deployment_hooks_keep_explicit_cloud_and_record_the_plan(policy_class):
    plan = {"svc-a": ["cloud-x"], "svc-b": ["edge-b"]}
    hedger = _DeploymentHedgerStub(plan=plan, version=4)
    policy = _deployment_policy(policy_class, hedger, plan)

    result = policy(_deployment_info())

    assert result == plan
    assert hedger._deployment_plan_history == {4: plan}
    if policy_class in _INITIAL_POLICY_CLASSES:
        assert hedger.registered == [plan]


@pytest.mark.unit
@pytest.mark.parametrize("policy_class", _POLICY_CLASSES)
def test_deployment_hooks_reject_invalid_default_plans(policy_class):
    valid_plan = {"svc-a": ["edge-a"], "svc-b": ["edge-b"]}
    invalid_default = {"svc-a": "edge-a", "svc-b": ["edge-b"]}
    policy = _deployment_policy(
        policy_class,
        _DeploymentHedgerStub(plan=valid_plan, version=4),
        invalid_default,
    )

    with pytest.raises(ValueError, match="JSON node list"):
        policy(_deployment_info())


@pytest.mark.unit
def test_initial_hook_registers_the_same_plan_it_returns():
    plan = {"svc-a": ["cloud-x"], "svc-b": ["edge-b"]}
    hedger = _DeploymentHedgerStub(version=0)
    policy = _deployment_policy(HedgerInitialDeploymentPolicy, hedger, plan)

    result = policy(_deployment_info())

    assert result == plan
    assert hedger.registered == [plan]
    assert hedger._deployment_plan_history == {0: plan}
