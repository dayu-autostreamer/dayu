import importlib
from types import SimpleNamespace

import pytest


base_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.base_initial_deployment_policy"
)
fixed_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.fixed_initial_deployment_policy"
)
full_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.full_initial_deployment_policy"
)
random_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.random_initial_deployment_policy"
)
base_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.base_redeployment_policy"
)
fixed_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.fixed_redeployment_policy"
)
non_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.non_redeployment_policy"
)
selection_base_module = importlib.import_module(
    "core.lib.algorithms.schedule_selection_policy.base_selection_policy"
)
fixed_selection_module = importlib.import_module(
    "core.lib.algorithms.schedule_selection_policy.fixed_selection_policy"
)
random_selection_module = importlib.import_module(
    "core.lib.algorithms.schedule_selection_policy.random_selection_policy"
)


def build_deployment_info():
    return {
        "source": {"id": 7},
        "dag": {"detector": {}, "tracker": {}},
        "node_set": ["edge-a", "edge-b"],
        "all_edge_nodes": ["edge-a", "edge-b", "edge-c"],
    }


@pytest.mark.unit
def test_deployment_policy_bases_and_fixed_policies_cover_loading_and_defaults(monkeypatch):
    with pytest.raises(NotImplementedError):
        base_initial_module.BaseInitialDeploymentPolicy()(build_deployment_info())
    with pytest.raises(NotImplementedError):
        base_redeployment_module.BaseRedeploymentPolicy()(build_deployment_info())

    loaded_paths = []
    monkeypatch.setattr(
        fixed_initial_module.Context,
        "get_file_path",
        staticmethod(lambda relative_path: f"/runtime/{relative_path}"),
    )
    monkeypatch.setattr(
        fixed_initial_module.ConfigLoader,
        "load",
        staticmethod(lambda path: loaded_paths.append(path) or {"detector": ["edge-a", "cloud-a"]}),
    )

    initial_policy = fixed_initial_module.FixedInitialDeploymentPolicy(SimpleNamespace(), 0, policy="policy.yaml")
    deploy_plan = initial_policy(build_deployment_info())
    assert loaded_paths == ["/runtime/policy.yaml"]
    assert sorted(deploy_plan["detector"]) == ["edge-a"]
    assert sorted(deploy_plan["tracker"]) == ["edge-a", "edge-b"]

    redeploy_paths = []
    monkeypatch.setattr(
        fixed_redeployment_module.Context,
        "get_file_path",
        staticmethod(lambda relative_path: f"/runtime/{relative_path}"),
    )
    monkeypatch.setattr(
        fixed_redeployment_module.ConfigLoader,
        "load",
        staticmethod(lambda path: redeploy_paths.append(path) or {"tracker": ["edge-b", "cloud-a"]}),
    )
    redeploy_policy = fixed_redeployment_module.FixedRedeploymentPolicy(SimpleNamespace(), 0, policy="redeploy.yaml")
    redeploy_plan = redeploy_policy(build_deployment_info())
    assert redeploy_paths == ["/runtime/redeploy.yaml"]
    assert sorted(redeploy_plan["detector"]) == ["edge-a", "edge-b"]
    assert sorted(redeploy_plan["tracker"]) == ["edge-b"]

    with pytest.raises(TypeError, match="type str or dict"):
        fixed_initial_module.FixedInitialDeploymentPolicy(SimpleNamespace(), 0, policy=object())
    with pytest.raises(TypeError, match="type str or dict"):
        fixed_redeployment_module.FixedRedeploymentPolicy(SimpleNamespace(), 0, policy=object())


@pytest.mark.unit
def test_initial_and_redeployment_policies_cover_full_random_and_non_redeployment(monkeypatch):
    info = build_deployment_info()

    full_policy = full_initial_module.FullInitialDeploymentPolicy(SimpleNamespace(), 0)
    assert full_policy(info) == {
        "edge-a": ["detector", "tracker"],
        "edge-b": ["detector", "tracker"],
    }

    random_policy = random_initial_module.RandomInitialDeploymentPolicy(SimpleNamespace(), 0, max_service_num=-1)
    monkeypatch.setattr(random_initial_module.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random_initial_module.random, "randint", lambda start, end: 1)
    monkeypatch.setattr(random_initial_module.random, "sample", lambda seq, count: sorted(seq)[:count])
    random_plan = random_policy(info)
    assert random_plan["edge-a"] == ["detector", "tracker"]
    assert random_plan["edge-b"] == ["detector"]

    warnings = []
    monkeypatch.setattr(random_initial_module.LOGGER, "warning", lambda message: warnings.append(message))
    bounded_policy = random_initial_module.RandomInitialDeploymentPolicy(SimpleNamespace(), 0, max_service_num=1)
    bounded_plan = bounded_policy(
        {
            "source": {"id": 8},
            "dag": {"detector": {}, "tracker": {}},
            "node_set": ["edge-a"],
        }
    )
    assert bounded_plan == {"edge-a": ["detector", "tracker"]}
    assert any("cannot be deployed" in message for message in warnings)

    monkeypatch.setattr(
        non_redeployment_module.KubeConfig,
        "get_service_nodes_dict",
        staticmethod(lambda: {"detector": ["edge-a"]}),
    )
    non_policy = non_redeployment_module.NonRedeploymentPolicy(SimpleNamespace(), 0)
    assert non_policy(info) == {"detector": ["edge-a"]}

    monkeypatch.setattr(
        non_redeployment_module.KubeConfig,
        "get_service_nodes_dict",
        staticmethod(lambda: None),
    )
    with pytest.raises(RuntimeError, match="returned None"):
        non_redeployment_module.NonRedeploymentPolicy(SimpleNamespace(), 0)


@pytest.mark.unit
def test_selection_policies_cover_invalid_configuration_and_empty_candidates(monkeypatch):
    warnings = []
    monkeypatch.setattr(fixed_selection_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(random_selection_module.LOGGER, "warning", lambda message: warnings.append(message))

    invalid_position = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value=-1, fixed_type="position"
    )
    invalid_hostname = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value=123, fixed_type="hostname"
    )
    invalid_type = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value="edge-a", fixed_type="region"
    )

    assert invalid_position.fixed_value == 0
    assert invalid_hostname.fixed_value == ""
    assert invalid_type.fixed_type == "position"

    info = build_deployment_info()
    fallback_position = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value=9, fixed_type="position"
    )
    fallback_hostname = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value="missing", fixed_type="hostname"
    )

    assert fallback_position(info) == "edge-a"
    assert fallback_hostname(info) == "edge-a"
    assert fixed_selection_module.FixedSelectionPolicy(SimpleNamespace(), 1)({"source": {"id": 1}, "node_set": []}) is None

    selector = selection_base_module.BaseSelectionPolicy(scope="source_bound")
    assert selector.get_candidate_node_set(info) == ["edge-a", "edge-b"]
    selector.scope = "all_edge_nodes"
    monkeypatch.setattr(selection_base_module.NodeInfo, "get_all_edge_nodes", staticmethod(lambda: ["edge-b", "edge-c"]))
    assert selector.get_candidate_node_set({"node_set": ["edge-a"], "all_edge_nodes": None}) == ["edge-b", "edge-c"]

    random_selector = random_selection_module.RandomSelectionPolicy(SimpleNamespace(), 1, scope="node_set")
    monkeypatch.setattr(random_selection_module.random, "choice", lambda seq: seq[-1])
    assert random_selector(info) == "edge-b"
    assert random_selector({"source": {"id": 2}, "node_set": []}) is None
    assert any("illegal" in message or "not supported" in message or "empty" in message for message in warnings)
