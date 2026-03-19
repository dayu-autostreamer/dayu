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

