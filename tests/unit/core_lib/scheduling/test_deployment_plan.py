from types import SimpleNamespace

import pytest

from core.lib.scheduling.deployment_plan import (
    allowed_nodes,
    cloud_plan,
    dag_services,
    fixed_plan,
    validate_plan,
)


def deployment_info():
    return {
        "dag": {
            "start": {},
            "detector": {},
            "tracker": {},
            "end": {},
        },
        "node_set": [" edge-b ", "edge-a", "edge-a"],
    }


@pytest.mark.unit
def test_deployment_context_extracts_real_services_and_exact_candidates():
    info = deployment_info()

    assert dag_services(info) == ("detector", "tracker")
    assert allowed_nodes(info, cloud_node=" cloud-a ") == {
        "edge-a",
        "edge-b",
        "cloud-a",
    }

    with pytest.raises(ValueError, match="requires a dag object"):
        dag_services({"dag": []})
    with pytest.raises(ValueError, match="requires a node_set list"):
        allowed_nodes({"node_set": {"edge-a"}})


@pytest.mark.unit
def test_validate_plan_returns_the_only_canonical_public_shape():
    plan = validate_plan(
        {
            "detector": ["edge-b", "edge-a", "edge-b"],
            "tracker": ["cloud-a"],
        },
        deployment_info(),
        cloud_node="cloud-a",
    )

    assert plan == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["cloud-a"],
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (None, "must return an object"),
        ({"detector": ["edge-a"]}, "omitted current DAG services"),
        (
            {
                "detector": ["edge-a"],
                "tracker": ["edge-b"],
                "stale": ["edge-a"],
            },
            "outside the current DAG",
        ),
        (
            {"detector": "edge-a", "tracker": ["edge-b"]},
            "must return a JSON node list",
        ),
        (
            {"detector": [""], "tracker": ["edge-b"]},
            "returned an empty node name",
        ),
        (
            {"detector": ["edge-x"], "tracker": ["edge-b"]},
            "selected non-candidate nodes",
        ),
        (
            {"detector": [], "tracker": ["edge-b"]},
            "returned no target nodes",
        ),
    ],
)
def test_validate_plan_rejects_ambiguous_or_incomplete_contracts(plan, message):
    with pytest.raises(ValueError, match=message):
        validate_plan(plan, deployment_info())


@pytest.mark.unit
def test_fixed_plan_scopes_configuration_to_the_current_dag():
    policy = {
        "detector": ["edge-b", "edge-a"],
        "tracker": ["cloud-a"],
        "stale": ["outside-node"],
    }

    assert fixed_plan(policy, deployment_info(), cloud_node="cloud-a") == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["cloud-a"],
    }
    assert policy["detector"] == ["edge-b", "edge-a"]

    with pytest.raises(ValueError, match="fixed deployment policy must be an object"):
        fixed_plan([], deployment_info())


@pytest.mark.unit
def test_cloud_plan_uses_only_the_injected_cloud_identity():
    system = SimpleNamespace(cloud_device="control-plane-a")

    assert cloud_plan(system, deployment_info()) == {
        "detector": ["control-plane-a"],
        "tracker": ["control-plane-a"],
    }

    with pytest.raises(ValueError, match="requires system.cloud_device"):
        cloud_plan(SimpleNamespace(cloud_device=""), deployment_info())
