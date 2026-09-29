import copy

import pytest

from core.lib.scheduling import END, START, materialize_offloading_plan


def _dag():
    return {
        START: {
            "service": {"service_name": START, "execute_device": ""},
            "prev_nodes": [],
            "next_nodes": ["detect"],
        },
        "detect": {
            "service": {"service_name": "detect", "execute_device": ""},
            "prev_nodes": [START],
            "next_nodes": [END],
        },
        END: {
            "service": {"service_name": END, "execute_device": ""},
            "prev_nodes": ["detect"],
            "next_nodes": [],
        },
    }


@pytest.mark.unit
def test_materialize_offloading_plan_returns_an_immutable_full_dag_policy():
    configuration = {"fps": 8, "resolution": "720p"}
    dag = _dag()
    original_configuration = copy.deepcopy(configuration)
    original_dag = copy.deepcopy(dag)

    policy = materialize_offloading_plan(
        configuration,
        dag,
        {"detect": "edge-a"},
        "source-node",
        "cloud-node",
    )

    assert policy["fps"] == 8
    assert policy["resolution"] == "720p"
    assert policy["dag"][START]["service"]["execute_device"] == "source-node"
    assert policy["dag"]["detect"]["service"]["execute_device"] == "edge-a"
    assert policy["dag"][END]["service"]["execute_device"] == "cloud-node"
    assert configuration == original_configuration
    assert dag == original_dag


@pytest.mark.unit
@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ({}, "missing=\\['detect'\\]"),
        (
            {"detect": "edge-a", "unknown": "edge-b"},
            "extra=\\['unknown'\\]",
        ),
    ],
)
def test_materialize_offloading_plan_requires_exact_service_coverage(
    plan,
    message,
):
    with pytest.raises(ValueError, match=message):
        materialize_offloading_plan({}, _dag(), plan, "source", "cloud")
