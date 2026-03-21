import pytest

from core.lib.content import DAG
from core.lib.solver import PathSolver


def service_entry(name, *, next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": "",
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_path_solver_dag():
    return DAG.from_dict(
        {
            "src": service_entry("src", next_nodes=["a", "b"]),
            "a": service_entry("a", prev_nodes=["src", "b"], next_nodes=["dest"]),
            "b": service_entry("b", prev_nodes=["src"], next_nodes=["a"]),
            "dest": service_entry("dest", prev_nodes=["a"]),
        }
    )


@pytest.mark.unit
def test_path_solver_supports_same_node_queries_and_weighted_paths():
    solver = PathSolver(build_path_solver_dag())

    assert solver.get_shortest_path("src", "src") == ["src"]
    assert solver.get_all_paths("src", "src") == [["src"]]

    shortest_weights = {"src": 0, "a": 10, "b": -9, "dest": 1}
    shortest_weight, shortest_path = solver.get_weighted_shortest_path(
        "src",
        "dest",
        lambda service: shortest_weights[service.get_service_name()],
    )
    assert shortest_weight == 2
    assert shortest_path == ["src", "b", "a", "dest"]

    longest_weights = {"src": 0, "a": 1, "b": 10, "dest": 1}
    longest_weight, longest_path = solver.get_weighted_longest_path(
        "src",
        "dest",
        lambda service: longest_weights[service.get_service_name()],
    )
    assert longest_weight == 12
    assert longest_path == ["src", "b", "a", "dest"]


@pytest.mark.unit
def test_path_solver_raises_for_missing_nodes_and_unreachable_destinations():
    solver = PathSolver(build_path_solver_dag())

    with pytest.raises(KeyError, match="does not exist"):
        solver.get_shortest_path("missing", "dest")

    disconnected = DAG.from_dict(
        {
            "src": service_entry("src", next_nodes=["a"]),
            "a": service_entry("a", prev_nodes=["src"]),
            "dest": service_entry("dest"),
        }
    )
    disconnected_solver = PathSolver(disconnected)

    with pytest.raises(ValueError, match="No paths exist"):
        disconnected_solver.get_all_paths("src", "dest")
    with pytest.raises(ValueError, match="No path exists"):
        disconnected_solver.get_weighted_shortest_path("src", "dest", lambda service: 1)
    with pytest.raises(ValueError, match="No path exists"):
        disconnected_solver.get_weighted_longest_path("src", "dest", lambda service: 1)
