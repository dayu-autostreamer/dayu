import importlib

import pytest


@pytest.mark.unit
def test_fixed_selection_policy_can_select_from_all_edge_nodes_scope():
    module = importlib.import_module("core.lib.algorithms.schedule_selection_policy.fixed_selection_policy")

    policy = module.FixedSelectionPolicy(
        system=None,
        agent_id=0,
        fixed_value="edge-free",
        fixed_type="hostname",
        scope="all_edge_nodes",
    )

    selected_node = policy(
        {
            "source": {"id": 0},
            "node_set": ["edge-a", "edge-b"],
            "all_edge_nodes": ["edge-a", "edge-b", "edge-free"],
        }
    )

    assert selected_node == "edge-free"


@pytest.mark.unit
def test_physical_topology_accepts_source_device_outside_processing_node_set(monkeypatch):
    pytest.importorskip("torch", reason="Hedger topology tests require torch in the test environment")
    module = importlib.import_module("core.lib.algorithms.shared.hedger.hedger_config")
    monkeypatch.setattr(module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-master"))

    topology = module.PhysicalTopology(["edge-a", "edge-b"], "edge-free")

    assert topology.nodes == ["edge-free", "edge-a", "edge-b", "cloud-master"]
    assert topology.source_idx == 0
    assert topology.cloud_idx == 3


@pytest.mark.unit
def test_hedger_initial_deployment_policy_returns_processing_plan_when_source_is_external(monkeypatch):
    pytest.importorskip("torch", reason="Hedger policy tests require torch in the test environment")
    module = importlib.import_module(
        "core.lib.algorithms.schedule_initial_deployment_policy.hedger_initial_deployment_policy"
    )

    class FakeHedger:
        def register_logical_topology(self, dag):
            self.logical_topology = dag

        def register_physical_topology(self, edge_nodes, source_device):
            self.edge_nodes = edge_nodes
            self.source_device = source_device

        def register_state_buffer(self):
            return None

        def register_initial_deployment(self, deployment):
            self.deployment = deployment

        def get_initial_deployment_plan(self):
            return {"face-detection": ["edge-free", "edge-a"]}

    class FakeSystem:
        network_params = {}
        hyper_params = {}
        agent_params = {}

    monkeypatch.setattr(
        module.HedgerInitialDeploymentPolicy,
        "register_hedger",
        lambda self, hedger_id="hedger": setattr(self, "hedger", FakeHedger()),
    )

    policy = module.HedgerInitialDeploymentPolicy(
        FakeSystem(),
        agent_id=0,
        deployment={"face-detection": ["edge-a"]},
    )

    deploy_plan = policy(
        {
            "source": {"id": 0, "source_device": "edge-free"},
            "node_set": ["edge-a", "edge-b"],
            "dag": {
                "face-detection": {
                    "service": {"service_name": "face-detection"},
                    "next_nodes": [],
                }
            },
        }
    )

    assert deploy_plan == {"face-detection": ["edge-a"]}
