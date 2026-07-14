import importlib
from types import SimpleNamespace

import pytest


base_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.base_initial_deployment_policy"
)
fixed_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.fixed_initial_deployment_policy"
)
cloud_initial_module = importlib.import_module(
    "core.lib.algorithms.schedule_initial_deployment_policy.cloud_initial_deployment_policy"
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
cloud_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.cloud_redeployment_policy"
)
non_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.non_redeployment_policy"
)
dynamic_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.dynamic_redeployment_policy"
)
offline_redeployment_module = importlib.import_module(
    "core.lib.algorithms.schedule_redeployment_policy.offline_profiling_redeployment_policy"
)
integrated_predictor_module = importlib.import_module(
    "core.lib.algorithms.schedule_agent.steady.integrated_safe_predictor"
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
        staticmethod(lambda path: loaded_paths.append(path) or {
            "detector": ["edge-a", "cloud-a"],
            "tracker": ["edge-b"],
            "unused-service": ["edge-a"],
        }),
    )

    system = SimpleNamespace(cloud_device="cloud-a")
    initial_policy = fixed_initial_module.FixedInitialDeploymentPolicy(system, 0, policy="policy.yaml")
    deploy_plan = initial_policy(build_deployment_info())
    assert loaded_paths == ["/runtime/policy.yaml"]
    assert deploy_plan == {
        "detector": ["cloud-a", "edge-a"],
        "tracker": ["edge-b"],
    }

    empty_initial_policy = fixed_initial_module.FixedInitialDeploymentPolicy(system, 0)
    with pytest.raises(ValueError, match="omitted current DAG services"):
        empty_initial_policy(build_deployment_info())

    redeploy_paths = []
    monkeypatch.setattr(
        fixed_redeployment_module.Context,
        "get_file_path",
        staticmethod(lambda relative_path: f"/runtime/{relative_path}"),
    )
    monkeypatch.setattr(
        fixed_redeployment_module.ConfigLoader,
        "load",
        staticmethod(lambda path: redeploy_paths.append(path) or {
            "detector": ["edge-a"],
            "tracker": ["edge-b", "cloud-a"],
            "unused-service": ["edge-a"],
        }),
    )
    redeploy_policy = fixed_redeployment_module.FixedRedeploymentPolicy(system, 0, policy="redeploy.yaml")
    redeploy_plan = redeploy_policy(build_deployment_info())
    assert redeploy_paths == ["/runtime/redeploy.yaml"]
    assert redeploy_plan == {
        "detector": ["edge-a"],
        "tracker": ["cloud-a", "edge-b"],
    }

    empty_redeployment_policy = fixed_redeployment_module.FixedRedeploymentPolicy(system, 0)
    with pytest.raises(ValueError, match="omitted current DAG services"):
        empty_redeployment_policy(build_deployment_info())

    with pytest.raises(TypeError, match="type str or dict"):
        fixed_initial_module.FixedInitialDeploymentPolicy(SimpleNamespace(), 0, policy=object())
    with pytest.raises(TypeError, match="type str or dict"):
        fixed_redeployment_module.FixedRedeploymentPolicy(SimpleNamespace(), 0, policy=object())


@pytest.mark.unit
def test_initial_and_redeployment_policies_cover_full_random_and_non_redeployment(monkeypatch):
    info = build_deployment_info()

    full_policy = full_initial_module.FullInitialDeploymentPolicy(SimpleNamespace(), 0)
    assert full_policy(info) == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["edge-a", "edge-b"],
    }

    random_policy = random_initial_module.RandomInitialDeploymentPolicy(SimpleNamespace(), 0, max_service_num=-1)
    monkeypatch.setattr(random_initial_module.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(random_initial_module.random, "randint", lambda start, end: 1)
    monkeypatch.setattr(random_initial_module.random, "sample", lambda seq, count: sorted(seq)[:count])
    random_plan = random_policy(info)
    assert random_plan == {
        "detector": ["edge-a", "edge-b"],
        "tracker": ["edge-a"],
    }

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
    assert bounded_plan == {"detector": ["edge-a"], "tracker": ["edge-a"]}
    assert any("cannot be deployed" in message for message in warnings)

    runtime_system = SimpleNamespace(
        cloud_device="cloud-a",
        runtime_service_nodes=lambda: {"detector": ["edge-a"]}
    )
    non_policy = non_redeployment_module.NonRedeploymentPolicy(runtime_system, 0)
    with pytest.raises(ValueError, match="omitted current DAG services"):
        non_policy(info)

    runtime_system.runtime_service_nodes = lambda: {
        "detector": ["edge-a"], "tracker": ["cloud-a"], "unused-service": ["edge-a"],
    }
    non_policy = non_redeployment_module.NonRedeploymentPolicy(runtime_system, 0)
    assert non_policy(info) == {"detector": ["edge-a"], "tracker": ["cloud-a"]}

    assert cloud_initial_module.CloudInitialDeploymentPolicy(runtime_system, 0)(info) == {
        "detector": ["cloud-a"], "tracker": ["cloud-a"],
    }
    assert cloud_redeployment_module.CloudRedeploymentPolicy(runtime_system, 0)(info) == {
        "detector": ["cloud-a"], "tracker": ["cloud-a"],
    }

    with pytest.raises(RuntimeError, match="runtime directory deployment is not initialized"):
        non_redeployment_module.NonRedeploymentPolicy(
            SimpleNamespace(runtime_service_nodes=lambda: None), 0
        )


@pytest.mark.unit
def test_selection_policies_cover_invalid_configuration_and_empty_candidates(monkeypatch):
    warnings = []
    monkeypatch.setattr(random_selection_module.LOGGER, "warning", lambda message: warnings.append(message))

    with pytest.raises(ValueError, match="non-negative integer"):
        fixed_selection_module.FixedSelectionPolicy(
            SimpleNamespace(), 1, fixed_value=-1, fixed_type="position"
        )
    with pytest.raises(ValueError, match="non-empty string"):
        fixed_selection_module.FixedSelectionPolicy(
            SimpleNamespace(), 1, fixed_value=123, fixed_type="hostname"
        )
    with pytest.raises(ValueError, match="position.*hostname"):
        fixed_selection_module.FixedSelectionPolicy(
            SimpleNamespace(), 1, fixed_value="edge-a", fixed_type="region"
        )

    info = build_deployment_info()
    invalid_position = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value=9, fixed_type="position"
    )
    invalid_hostname = fixed_selection_module.FixedSelectionPolicy(
        SimpleNamespace(), 1, fixed_value="missing", fixed_type="hostname"
    )

    with pytest.raises(ValueError, match="outside the permitted"):
        invalid_position(info)
    with pytest.raises(ValueError, match="not a permitted candidate"):
        invalid_hostname(info)
    with pytest.raises(ValueError, match="no permitted source candidate"):
        fixed_selection_module.FixedSelectionPolicy(SimpleNamespace(), 1)({
            "source": {"id": 1}, "node_set": [],
        })

    selector = selection_base_module.BaseSelectionPolicy(scope="selected_edge_nodes")
    assert selector.get_candidate_node_set(info) == ["edge-a", "edge-b"]
    selector.scope = "all_edge_nodes"
    assert selector.get_candidate_node_set({
        "node_set": ["edge-a"],
        "source_candidate_nodes": ["edge-b", "edge-c"],
    }) == ["edge-b", "edge-c"]

    with pytest.raises(ValueError, match="source selection scope"):
        selection_base_module.BaseSelectionPolicy(scope="source_bound")

    random_selector = random_selection_module.RandomSelectionPolicy(SimpleNamespace(), 1, scope="selected_edge_nodes")
    monkeypatch.setattr(random_selection_module.random, "choice", lambda seq: seq[-1])
    assert random_selector(info) == "edge-b"
    assert random_selector({"source": {"id": 2}, "node_set": []}) is None
    assert any("empty" in message for message in warnings)


@pytest.mark.unit
def test_dynamic_redeployment_scopes_dag_and_uses_injected_cloud_identity():
    class Agent:
        @staticmethod
        def get_latest_offloading_policy():
            return {
                "detector": "control-plane-a",
                "tracker": "edge-a",
                "stale-service": "outside-node",
            }

    system = SimpleNamespace(
        cloud_device="control-plane-a",
        schedule_table={7: Agent()},
    )
    policy = dynamic_redeployment_module.DynamicRedeploymentPolicy(
        system,
        7,
        redeployment_interval_minutes=0,
        default_service_limit=4,
        policy={
            "detector": ["edge-a"],
            "tracker": ["edge-b"],
            "stale-service": ["outside-node"],
        },
    )

    assert policy(build_deployment_info()) == {
        "detector": ["control-plane-a"],
        "tracker": ["edge-a"],
    }
    policy.update_latest_offloading_policy({"detector": "control-plane-a"})
    assert policy.latest_offloading_policy == {"detector": "control-plane-a"}


@pytest.mark.unit
def test_offline_redeployment_covers_current_dag_and_has_no_cloud_hostname_assumption(monkeypatch):
    class FakeOverhead:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def get_latest_overhead():
            return 0.0

    monkeypatch.setattr(offline_redeployment_module, "OverheadEstimator", FakeOverhead)
    policy = offline_redeployment_module.OfflineProfilingRedeploymentPolicy(
        SimpleNamespace(cloud_device="control-plane-a"),
        7,
        latency_profile={"detector": {"edge-a": 1.0}},
        default_service_limit=4,
    )
    info = {
        "source": {"id": 7},
        "dag": {"detector": {}, "tracker": {}},
        "node_set": ["edge-a"],
    }
    assert policy(info) == {
        "detector": ["edge-a"],
        "tracker": ["edge-a"],
    }

    cloud_only = {**info, "node_set": []}
    assert policy(cloud_only) == {
        "detector": ["control-plane-a"],
        "tracker": ["control-plane-a"],
    }


@pytest.mark.unit
def test_execution_profiles_resolve_roles_without_fixed_cluster_hostname():
    resolver = integrated_predictor_module.CorrectedPredictor._execution_profile_value
    role_profile = {
        "execute_role=edge#resolution=540p": 1.0,
        "execute_role=cloud#resolution=540p": 2.0,
    }
    assert resolver(role_profile, "540p", "edge") == 1.0
    assert resolver(role_profile, "540p", "cloud") == 2.0

    labelled_profile = {
        "execute_device=edge7#resolution=720p": 3.0,
        "execute_device=cloud-a#resolution=720p": 4.0,
    }
    assert resolver(labelled_profile, "720p", "edge") == 3.0
    assert resolver(labelled_profile, "720p", "cloud") == 4.0
