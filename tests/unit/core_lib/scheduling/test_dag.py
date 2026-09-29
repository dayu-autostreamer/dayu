import pytest

from core.lib.scheduling import END, START, service_names, topological_order


def _dag():
    return {
        "join": {
            "prev_nodes": ["left", "right"],
            "next_nodes": [END],
        },
        START: {"prev_nodes": [], "next_nodes": ["right", "left"]},
        "right": {"prev_nodes": [START], "next_nodes": ["join"]},
        END: {"prev_nodes": ["join"], "next_nodes": []},
        "left": {"prev_nodes": [START], "next_nodes": ["join"]},
    }


@pytest.mark.unit
def test_dag_helpers_exclude_boundaries_and_order_deterministically():
    dag = _dag()

    assert service_names(dag) == ["join", "right", "left"]
    assert topological_order(dag) == [START, "left", "right", "join", END]


@pytest.mark.unit
def test_topological_order_rejects_cycles():
    dag = {
        "left": {"prev_nodes": ["right"], "next_nodes": ["right"]},
        "right": {"prev_nodes": ["left"], "next_nodes": ["left"]},
    }

    with pytest.raises(ValueError, match="acyclic DAG"):
        topological_order(dag)


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, [], "dag"])
def test_dag_helpers_require_a_mapping(value):
    with pytest.raises(TypeError, match="DAG must be a mapping"):
        service_names(value)
    with pytest.raises(TypeError, match="DAG must be a mapping"):
        topological_order(value)
