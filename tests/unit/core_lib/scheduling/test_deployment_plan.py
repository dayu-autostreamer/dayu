from types import SimpleNamespace

import pytest

from core.lib.scheduling.deployment_plan import (
    CLOUD_NODE_TOKEN,
    allowed_nodes,
    cloud_plan,
    cloud_replica_plan,
    dag_services,
    fixed_plan,
    full_edge_plan,
    full_plan,
    normalize_include_cloud,
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
        "tracker": [CLOUD_NODE_TOKEN],
        "stale": ["outside-node"],
    }

    assert fixed_plan(policy, deployment_info(), cloud_node="cloud-a") == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["cloud-a"],
    }
    assert policy["detector"] == ["edge-b", "edge-a"]
    assert policy["tracker"] == [CLOUD_NODE_TOKEN]

    assert fixed_plan(
        {
            "detector": ["edge-a"],
            "tracker": ["edge-b"],
        },
        deployment_info(),
        cloud_node="cloud-a",
        include_cloud=True,
    ) == {
        "detector": ["cloud-a", "edge-a"],
        "tracker": ["cloud-a", "edge-b"],
    }

    with pytest.raises(ValueError, match="fixed deployment policy must be an object"):
        fixed_plan([], deployment_info())
    with pytest.raises(ValueError, match="@cloud.*requires system.cloud_device"):
        fixed_plan(
            {"detector": [CLOUD_NODE_TOKEN], "tracker": ["edge-b"]},
            deployment_info(),
        )
    with pytest.raises(TypeError, match="include_cloud must be a boolean"):
        fixed_plan(
            {"detector": ["edge-a"], "tracker": ["edge-b"]},
            deployment_info(),
            include_cloud="false",
        )


@pytest.mark.unit
def test_cloud_replica_and_full_plans_use_explicit_cloud_identity():
    assert cloud_replica_plan(
        {"detector": ["edge-a"], "tracker": ["edge-b"]},
        deployment_info(),
        "cloud-a",
    ) == {
        "detector": ["cloud-a", "edge-a"],
        "tracker": ["cloud-a", "edge-b"],
    }
    edge_info = deployment_info()
    edge_info["node_set"].append("cloud-a")
    assert full_edge_plan(edge_info, "cloud-a") == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["edge-a", "edge-b"],
    }
    assert full_plan(deployment_info(), "cloud-a") == {
        "detector": ["cloud-a", "edge-a", "edge-b"],
        "tracker": ["cloud-a", "edge-a", "edge-b"],
    }
    assert normalize_include_cloud(False) is False

    with pytest.raises(ValueError, match="full deployment policy requires system.cloud_device"):
        full_plan(deployment_info(), "")


@pytest.mark.unit
def test_invalid_deployment_names_the_allowed_processor_nodes():
    with pytest.raises(ValueError) as exc_info:
        validate_plan(
            {"detector": ["edge-x"], "tracker": ["edge-b"]},
            deployment_info(),
            cloud_node="cloud-a",
        )

    assert str(exc_info.value) == (
        "deployment policy for service 'detector' selected non-candidate nodes: "
        "['edge-x']; allowed processor nodes: ['cloud-a', 'edge-a', 'edge-b']"
    )


@pytest.mark.unit
def test_cloud_plan_uses_only_the_injected_cloud_identity():
    system = SimpleNamespace(cloud_device="control-plane-a")

    assert cloud_plan(system, deployment_info()) == {
        "detector": ["control-plane-a"],
        "tracker": ["control-plane-a"],
    }

    with pytest.raises(ValueError, match="requires system.cloud_device"):
        cloud_plan(SimpleNamespace(cloud_device=""), deployment_info())
