import pytest

from core.lib.content import DAG
from core.lib.solver import IntermediateNodeSolver, LCASolver


def service_entry(name, *, next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": "",
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_solver_dag():
    return DAG.from_dict(
        {
            "start": service_entry("start", next_nodes=["mid"]),
            "mid": service_entry("mid", prev_nodes=["start"], next_nodes=["left", "right"]),
            "left": service_entry("left", prev_nodes=["mid"], next_nodes=["shared"]),
            "right": service_entry("right", prev_nodes=["mid"], next_nodes=["shared"]),
            "shared": service_entry("shared", prev_nodes=["left", "right"], next_nodes=["end"]),
            "end": service_entry("end", prev_nodes=["shared"]),
        }
    )


@pytest.mark.unit
def test_intermediate_node_solver_finds_nodes_on_all_paths_and_validates_inputs():
    solver = IntermediateNodeSolver(build_solver_dag())

    assert solver.get_intermediate_nodes("mid", "shared") == {"left", "right"}
    assert solver.get_intermediate_nodes("left", "right") == set()

    with pytest.raises(KeyError, match="does not exist"):
        solver.get_intermediate_nodes("missing", "shared")


@pytest.mark.unit
def test_lca_solver_handles_ancestor_shortcuts_caching_and_disconnected_graphs():
    solver = LCASolver(build_solver_dag())

    assert solver.find_lca("shared", "end") == "shared"
    assert solver.find_lca("end", "shared") == "shared"
    assert solver.find_lca("left", "right") == "mid"

    ancestors = solver._get_ancestors("left")
    assert {"left", "mid", "start"} <= ancestors
    assert solver._get_ancestors("left") is ancestors
    assert solver._depth_cache["shared"] == 3

    disconnected = DAG.from_dict(
        {
            "root1": service_entry("root1", next_nodes=["leaf1"]),
            "leaf1": service_entry("leaf1", prev_nodes=["root1"]),
            "root2": service_entry("root2", next_nodes=["leaf2"]),
            "leaf2": service_entry("leaf2", prev_nodes=["root2"]),
        }
    )

    with pytest.raises(ValueError, match="No LCA"):
        LCASolver(disconnected).find_lca("leaf1", "leaf2")


@pytest.mark.unit
def test_lca_solver_finds_candidates_from_forward_search_before_backward_queue_drains():
    dag = DAG.from_dict(
        {
            "root": service_entry("root", next_nodes=["left1", "right_a", "right_b"]),
            "left1": service_entry("left1", prev_nodes=["root"], next_nodes=["left2"]),
            "left2": service_entry("left2", prev_nodes=["left1"], next_nodes=["left_leaf"]),
            "left_leaf": service_entry("left_leaf", prev_nodes=["left2"]),
            "right_a": service_entry("right_a", prev_nodes=["root"], next_nodes=["right_leaf"]),
            "right_b": service_entry("right_b", prev_nodes=["root"], next_nodes=["right_leaf"]),
            "right_leaf": service_entry("right_leaf", prev_nodes=["right_a", "right_b"]),
        }
    )

    solver = LCASolver(dag)
    assert solver.find_lca("left_leaf", "right_leaf") == "root"
