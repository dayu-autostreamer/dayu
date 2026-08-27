import ast
import copy
import itertools
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import core.lib.algorithms.schedule_agent.fragsplice_agent as fragsplice_agent_module
from core.lib.common import ClassFactory, ClassType, Context
from core.lib.algorithms.schedule_agent.fragsplice import (
    FragSpliceLatencyModel,
    FragSpliceOptimizer,
    FragSpliceRandomInputOptimizer,
    FragSpliceRandomLatencyModel,
    FragSpliceStagewiseEFTOptimizer,
)
from core.lib.algorithms.schedule_agent.fragsplice.execution_state import (
    FragSpliceExecutionState,
    FragSpliceRandomExecutionState,
)
from core.lib.algorithms.schedule_agent.fragsplice_agent import FragSpliceAgent
from core.lib.algorithms.schedule_agent.fragsplice_cold_sample_agent import (
    FragSpliceColdSampleAgent,
)
from core.lib.algorithms.schedule_agent.fragsplice_no_distribution_profiler_agent import (
    FragSpliceNoDistributionProfilerAgent,
)
from core.lib.algorithms.schedule_agent.fragsplice_no_plan_optimizer_agent import (
    FragSpliceNoPlanOptimizerAgent,
)
from core.lib.algorithms.schedule_agent.fragsplice_no_future_state_estimator_agent import (
    FragSpliceNoFutureStateEstimatorAgent,
)


def dag(device=""):
    return {
        "_start": {
            "service": {"service_name": "_start", "execute_device": "source"},
            "prev_nodes": [],
            "next_nodes": ["detect"],
        },
        "detect": {
            "service": {"service_name": "detect", "execute_device": device},
            "prev_nodes": ["_start"],
            "next_nodes": ["classify"],
        },
        "classify": {
            "service": {"service_name": "classify", "execute_device": device},
            "prev_nodes": ["detect"],
            "next_nodes": ["_end"],
        },
        "_end": {
            "service": {"service_name": "_end", "execute_device": "cloud"},
            "prev_nodes": ["classify"],
            "next_nodes": [],
        },
    }


def one_service_dag(device=""):
    value = dag(device)
    value["detect"]["next_nodes"] = ["_end"]
    value["_end"]["prev_nodes"] = ["detect"]
    value.pop("classify")
    return value


def test_fragsplice_hooks_are_registered():
    assert ClassFactory.get_cls(ClassType.SCH_AGENT, "fragsplice") is FragSpliceAgent
    assert (
        ClassFactory.get_cls(ClassType.SCH_AGENT, "fragsplice_cold_sample")
        is FragSpliceColdSampleAgent
    )
    assert (
        ClassFactory.get_cls(
            ClassType.SCH_AGENT,
            "fragsplice_no_distribution_profiler",
        )
        is FragSpliceNoDistributionProfilerAgent
    )
    assert (
        ClassFactory.get_cls(
            ClassType.SCH_AGENT,
            "fragsplice_no_future_state_estimator",
        )
        is FragSpliceNoFutureStateEstimatorAgent
    )
    assert (
        ClassFactory.get_cls(
            ClassType.SCH_AGENT,
            "fragsplice_no_plan_optimizer",
        )
        is FragSpliceNoPlanOptimizerAgent
    )
@pytest.mark.unit
def test_fragsplice_templates_share_fixed_deployment_and_common_parameters():
    root = Path(__file__).resolve().parents[4]

    def load_template(name):
        data = yaml.safe_load(
            (root / "template" / "scheduler" / name).read_text(encoding="utf-8")
        )
        env = {
            item["name"]: item["value"]
            for item in data["pod-template"]["env"]
        }
        return data, env

    cold, cold_env = load_template("fragsplice-cold-sample.yaml")
    main, main_env = load_template("fragsplice.yaml")
    ablations = {
        "fragsplice-no-distribution-profiler.yaml":
            "fragsplice_no_distribution_profiler",
        "fragsplice-no-future-state-estimator.yaml":
            "fragsplice_no_future_state_estimator",
        "fragsplice-no-plan-optimizer.yaml":
            "fragsplice_no_plan_optimizer",
    }

    cold_initial = ast.literal_eval(
        cold_env["SCH_INITIAL_DEPLOYMENT_POLICY_PARAMETERS"]
    )
    main_initial = ast.literal_eval(
        main_env["SCH_INITIAL_DEPLOYMENT_POLICY_PARAMETERS"]
    )
    cold_agent = ast.literal_eval(cold_env["SCH_AGENT_PARAMETERS"])
    main_agent = ast.literal_eval(main_env["SCH_AGENT_PARAMETERS"])

    assert cold_env["SCH_AGENT_NAME"] == "fragsplice_cold_sample"
    assert main_env["SCH_AGENT_NAME"] == "fragsplice"
    assert cold_initial == main_initial
    assert cold_env["SCH_REDEPLOYMENT_POLICY_NAME"] == "non"
    assert main_env["SCH_REDEPLOYMENT_POLICY_NAME"] == "non"
    assert "SCH_REDEPLOYMENT_POLICY_PARAMETERS" not in cold_env
    assert "SCH_REDEPLOYMENT_POLICY_PARAMETERS" not in main_env
    assert cold_agent["profile_path"] == main_agent["latency_profile"]
    assert cold_agent["configuration"] == main_agent["configuration"]
    assert main_agent["configuration"] == {
        "resolution": "720p",
        "fps": 10,
        "encoding": "mp4v",
        "buffer_size": 3,
    }
    assert cold_agent["max_inflight_tasks"] == 1
    assert cold["file-mount"] == main["file-mount"] == [{
        "pos": "cloud",
        "path": "scheduler/fragsplice/",
    }]
    for filename, agent_name in ablations.items():
        ablation, ablation_env = load_template(filename)
        assert ablation_env["SCH_AGENT_NAME"] == agent_name
        assert ast.literal_eval(
            ablation_env["SCH_INITIAL_DEPLOYMENT_POLICY_PARAMETERS"]
        ) == main_initial
        ablation_agent = ast.literal_eval(
            ablation_env["SCH_AGENT_PARAMETERS"]
        )
        if filename == "fragsplice-no-distribution-profiler.yaml":
            expected = dict(main_agent)
            expected.pop("latency_profile")
            expected.pop("queue_state_max_age_s")
            expected.pop("residual_half_life_tasks")
            expected.update({
                "random_invocation_cost_min_s": 0.0,
                "random_invocation_cost_max_s": 2.0,
                "random_overhead_cost_min_s": 0.0,
                "random_overhead_cost_max_s": 0.2,
                "random_workload_token_min": 0,
                "random_workload_token_max": 8,
            })
            assert ablation_agent == expected
        elif filename == "fragsplice-no-plan-optimizer.yaml":
            expected = dict(main_agent)
            expected.pop("incumbent_neighborhood_size")
            expected.pop("screening_beam_width")
            assert ablation_agent == expected
        else:
            assert ablation_agent == main_agent
        assert ablation_env["SCH_REDEPLOYMENT_POLICY_NAME"] == "non"
        assert "SCH_REDEPLOYMENT_POLICY_PARAMETERS" not in ablation_env
        if filename == "fragsplice-no-distribution-profiler.yaml":
            # Dayu requires a mount for its generated schedule config.  Use a
            # dedicated directory so this ablation cannot see the cold-profile
            # artifact mounted by the main algorithm.
            assert ablation["file-mount"] == [{
                "pos": "cloud",
                "path": "scheduler/fragsplice-no-distribution-profiler/",
            }]
            assert ablation["file-mount"] != main["file-mount"]
        else:
            assert ablation["file-mount"] == main["file-mount"]

    policies = yaml.safe_load(
        (root / "template" / "scheduler_policies.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy_files = {
        item["id"]: item["yaml"]
        for item in policies
    }
    assert policy_files["fragsplice-no-distribution-profiler"] == (
        "fragsplice-no-distribution-profiler.yaml"
    )
    assert "fragsplice-single-sample-profile" not in policy_files
    assert policy_files["fragsplice-no-future-state-estimator"] == (
        "fragsplice-no-future-state-estimator.yaml"
    )
    assert policy_files["fragsplice-no-plan-optimizer"] == (
        "fragsplice-no-plan-optimizer.yaml"
    )
    assert "fragsplice-current-state" not in policy_files
    assert not (
        root / "template" / "scheduler" / "fragsplice-current-state.yaml"
    ).exists()


@pytest.mark.unit
def test_fragsplice_generator_templates_exclude_redundant_inputs():
    root = Path(__file__).resolve().parents[4]

    def load_env(name):
        data = yaml.safe_load(
            (root / "template" / "generator" / name).read_text(
                encoding="utf-8"
            )
        )
        return {
            item["name"]: item["value"]
            for item in data["pod-template"]["env"]
        }

    main_env = load_env("generator-with-bursty.yaml")
    cold_env = load_env("generator-with-scheduler-permitted.yaml")

    assert main_env["GEN_GETTER_FILTER_NAME"] == "simple"
    assert "GEN_GETTER_FILTER_PARAMETERS" not in main_env
    burst = ast.literal_eval(main_env["HTTP_VIDEO_TASK_ARRIVAL_BURST"])
    assert burst == {
        "tasks_per_burst": 8,
        "intra_burst_rate_multiplier": 3.0,
    }
    assert main_env["ASYNC_TASK_SUBMISSION"] == "true"
    assert main_env["TASK_SUBMISSION_QUEUE_DEPTH"] == "16"

    assert cold_env["GEN_GETTER_FILTER_NAME"] == "scheduler_permitted"
    assert "GEN_GETTER_FILTER_PARAMETERS" not in cold_env
    assert "HTTP_VIDEO_TASK_ARRIVAL_BURST" not in cold_env
    assert "ASYNC_TASK_SUBMISSION" not in cold_env
    assert "TASK_SUBMISSION_QUEUE_DEPTH" not in cold_env

    natural_interval_s = 3 / 10
    short_interval_s = (
        natural_interval_s / burst["intra_burst_rate_multiplier"]
    )
    recovery_interval_s = (
        burst["tasks_per_burst"] * natural_interval_s
        - (burst["tasks_per_burst"] - 1) * short_interval_s
    )
    assert natural_interval_s == pytest.approx(0.30)
    assert short_interval_s == pytest.approx(0.10)
    assert recovery_interval_s == pytest.approx(1.70)

    policies = yaml.safe_load(
        (root / "template" / "scheduler_policies.yaml").read_text(
            encoding="utf-8"
        )
    )
    cold_policy = next(
        item for item in policies if item["id"] == "fragsplice-cold-sample"
    )
    assert cold_policy["dependency"]["generator"] == (
        "generator-with-scheduler-permitted.yaml"
    )
    assert not (
        root / "template" / "generator" / "generator-fragsplice-cold.yaml"
    ).exists()


@pytest.mark.unit
def test_fragsplice_shared_state_identity_stays_extension_owned():
    system = FakeSystem()

    first = fragsplice_agent_module._system_instance_token(system)
    second = fragsplice_agent_module._system_instance_token(system)
    another = fragsplice_agent_module._system_instance_token(FakeSystem())

    assert first is second
    assert another is not first
    assert not hasattr(system, "_fragsplice_instance_token")


@pytest.mark.unit
def test_random_latency_model_is_reproducible_and_serializable():
    model = FragSpliceRandomLatencyModel(random_seed=19)
    duplicate = FragSpliceRandomLatencyModel(
        state=model.to_state()
    )

    assert model.estimate("detect", "edge-a") == duplicate.estimate(
        "detect", "edge-a"
    )
    assert model.estimate_handoff(
        "detect", "edge-a"
    ) == duplicate.estimate_handoff("detect", "edge-a")
    samples = model.sample_task(
        1,
        {"detect": "edge-a", "classify": "edge-b"},
        random.Random(3),
    )
    alternate = model.sample_task(
        1,
        {"detect": "edge-b", "classify": "edge-a"},
        random.Random(3),
    )
    repeated = model.sample_task(
        1,
        {"detect": "edge-a", "classify": "edge-b"},
        random.Random(3),
    )
    assert all(0.0 < value <= 2.0 for value in samples.values())
    assert samples == repeated
    assert samples["detect"] != alternate["detect"]
    assert model.to_state() == {
        "random_invocation_cost_min_s": 0.0,
        "random_invocation_cost_max_s": 2.0,
        "random_overhead_cost_min_s": 0.0,
        "random_overhead_cost_max_s": 0.2,
        "random_seed": 19,
    }


@pytest.mark.unit
def test_random_execution_state_discards_observed_and_committed_work():
    snapshot = {
        "captured_at": 10.0,
        "commitments": [{
            "root_uuid": "root-1",
            "source_id": 1,
            "dag": dag(),
        }],
        "task_barriers": [{
            "root_uuid": "root-1",
            "arrived_branches": ["detect"],
        }],
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "waiting_count": 99,
                "running_task": {"root_uuid": "root-1"},
                "waiting_tasks": [{"root_uuid": "root-2"}],
            }}},
        },
    }
    state = FragSpliceRandomExecutionState(0, 8, 2.0)
    candidates = {
        "detect": ["edge-a", "edge-b"],
        "classify": ["edge-c"],
    }

    synthetic, summary = state.synthetic_snapshot(
        snapshot, candidates, seed=7
    )
    repeated, repeated_summary = state.synthetic_snapshot(
        snapshot, candidates, seed=7
    )

    assert synthetic == repeated
    assert summary == repeated_summary
    assert synthetic["commitments"] == []
    assert synthetic["reservations"] == []
    assert synthetic["task_barriers"] == []
    assert len(summary["replicas"]) == 3
    assert all(0 <= value <= 8 for value in summary["replicas"].values())
    assert "root-1" not in repr(synthetic)
    assert "root-2" not in repr(synthetic)
    assert synthetic["resources"]["edge-a"]["queue_state"][
        "detect"
    ]["waiting_count"] != 99


@pytest.mark.unit
def test_random_input_optimizer_is_independent_of_actual_system_state():
    optimizer = FragSpliceRandomInputOptimizer(
        FragSpliceRandomLatencyModel(random_seed=23),
        scenario_count=8,
        max_scenarios=8,
        random_seed=23,
    )
    task_dag = one_service_dag("edge-a")
    congested_a = {
        "captured_at": 5.0,
        "runtime_directory_revision": 1,
        "commitments": [{"root_uuid": "real-root", "dag": task_dag}],
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "waiting_count": 100,
            }}},
            "edge-b": {"queue_state": {"detect": {
                "busy": False,
                "waiting_count": 0,
                "waiting_tasks": [],
            }}},
        },
    }
    congested_b = copy.deepcopy(congested_a)
    congested_b["resources"]["edge-a"]["queue_state"]["detect"] = {
        "busy": False, "waiting_count": 0,
    }
    congested_b["resources"]["edge-b"]["queue_state"]["detect"] = {
        "busy": True, "waiting_count": 100,
    }
    info = {
        "source_id": 1,
        "source_device": "source",
        "task_context": {"root_uuid": "candidate-root"},
        "dag": task_dag,
        "meta_data": {"slo_seconds": 2.5},
    }

    result_a = optimizer.solve(
        copy.deepcopy(info),
        congested_a,
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )
    result_b = optimizer.solve(
        copy.deepcopy(info),
        congested_b,
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result_a["plan"] == result_b["plan"]
    assert result_a["score"] == result_b["score"]
    assert result_a["synthetic_replica_tokens"] == result_b[
        "synthetic_replica_tokens"
    ]
    assert result_a["planning_cost_domain"] == "random_uninformed"
    assert result_a["temporal_prediction_available"] is False
    assert result_a["prediction_is_synthetic"] is True
    assert result_a["actual_state_consumed"] is False
    assert result_a["scenario_count"] == 8


@pytest.mark.unit
def test_optimizer_exactly_searches_small_space_and_uses_committed_running_work():
    profile = {
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        }
    }
    model = FragSpliceLatencyModel(profile)
    optimizer = FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
    )
    old_dag = {
        "_start": {
            "service": {"service_name": "_start", "execute_device": "source"},
            "prev_nodes": [],
            "next_nodes": ["detect"],
        },
        "detect": {
            "service": {"service_name": "detect", "execute_device": "edge-a"},
            "prev_nodes": ["_start"],
            "next_nodes": ["_end"],
        },
        "_end": {
            "service": {"service_name": "_end", "execute_device": "cloud"},
            "prev_nodes": ["detect"],
            "next_nodes": [],
        },
    }
    snapshot = {
        "captured_at": 5.0,
        "commitments": [{
            "root_uuid": "old",
            "source_id": 1,
            "admitted_at": 4.0,
            "dag": old_dag,
        }],
        "task_barriers": [],
        "resources": {
            "edge-a": {
                "queue_state": {
                    "detect": {
                        "busy": True,
                        "running_elapsed_s": 0.0,
                        "running_task": {"root_uuid": "old"},
                        "waiting_tasks": [],
                    }
                }
            },
            "edge-b": {
                "queue_state": {
                    "detect": {
                        "busy": False,
                        "waiting_tasks": [],
                    }
                }
            },
        },
        "resource_runtime_revision": {},
    }
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "dag": old_dag,
            "meta_data": {"slo_seconds": 10.0},
        },
        snapshot,
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-b"}
    assert result["candidate_count"] == 2
    assert result["optimality_proven"] is True
    assert len(result["evaluated"]) == 2


@pytest.mark.unit
def test_stagewise_eft_ablation_uses_future_state_without_plan_optimization():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceStagewiseEFTOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
    )
    task_dag = one_service_dag()
    snapshot = {
        "captured_at": 5.0,
        "runtime_directory_revision": 1,
        "reservations": [],
        "commitments": [],
        "task_barriers": [],
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "running_phase": "processing",
                "phase_elapsed_s": 0.0,
                "observed_at": 5.0,
            }}},
            "edge-b": {"queue_state": {"detect": {
                "busy": False,
                "observed_at": 5.0,
            }}},
        },
        "resource_received_at": {
            "edge-a": 5.0,
            "edge-b": 5.0,
        },
        "resource_runtime_revision": {
            "edge-a": 1,
            "edge-b": 1,
        },
    }
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": task_dag,
            "meta_data": {"slo_seconds": 10.0},
        },
        snapshot,
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-b"}
    assert result["expanded"] == 2
    assert result["screened"] == 0
    assert len(result["evaluated"]) == 1
    assert result["optimality_proven"] is False


@pytest.mark.unit
def test_stagewise_eft_ablation_does_not_jointly_optimize_complete_plan():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [2.0]},
            },
            "classify": {
                "edge-b": {"samples": [0.1]},
            },
        },
        "transfer_pairs": {
            "classify": {
                "edge-b": {"samples": [10.0]},
            },
        },
    })
    common = {
        "info": {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": dag(),
            "meta_data": {"slo_seconds": 20.0},
        },
        "snapshot": {
            "captured_at": 5.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        "deployment": {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-b"],
        },
        "cloud_device": "cloud",
    }
    stagewise = FragSpliceStagewiseEFTOptimizer(
        model,
        default_slo_s=20.0,
        scenario_count=8,
        max_scenarios=8,
    ).solve(**common)
    plan_optimizer_result = FragSpliceOptimizer(
        model,
        default_slo_s=20.0,
        scenario_count=8,
        max_scenarios=8,
    ).solve(**common)

    assert stagewise["plan"] == {
        "detect": "edge-a",
        "classify": "edge-b",
    }
    assert plan_optimizer_result["plan"] == {
        "detect": "edge-b",
        "classify": "edge-b",
    }


@pytest.mark.unit
def test_branch_and_bound_matches_exhaustive_complete_plan_search():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
            "classify": {
                "edge-c": {"samples": [0.5]},
                "edge-d": {"samples": [0.7]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=10.0, scenario_count=8, max_scenarios=8
    )
    task_dag = dag()
    old_dag = dag()
    old_dag["detect"]["service"]["execute_device"] = "edge-a"
    old_dag["classify"]["service"]["execute_device"] = "edge-c"
    snapshot = {
        "captured_at": 10.0,
        "runtime_directory_revision": 1,
        "reservations": [],
        "commitments": [{
            "root_uuid": "old",
            "source_id": 1,
            "reserved_at": 9.5,
            "runtime_directory_revision": 1,
            "dag": old_dag,
        }],
        "task_barriers": [],
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "running_task": {
                    "root_uuid": "old",
                    "runtime_directory_revision": 1,
                },
                "running_phase": "processing",
                "phase_elapsed_s": 0.0,
                "observed_at": 10.0,
            }}},
        },
        "resource_received_at": {"edge-a": 10.0},
        "resource_runtime_revision": {"edge-a": 1},
    }
    deployment = {
        "detect": ["edge-a", "edge-b"],
        "classify": ["edge-c", "edge-d"],
    }
    info = {
        "source_id": 1,
        "source_device": "source",
        "task_context": {"root_uuid": "new"},
        "dag": task_dag,
        "meta_data": {"slo_seconds": 10.0},
    }
    result = optimizer.solve(info, snapshot, deployment, "cloud")

    state = FragSpliceExecutionState(snapshot, model, default_slo_s=10.0)
    seeds = [optimizer.random_seed + 1_000_003 * index for index in range(8)]
    baseline_cache = {}
    outcome_cache = {}
    exhaustive = []
    for detect, classify in itertools.product(
        deployment["detect"], deployment["classify"]
    ):
        plan = {"detect": detect, "classify": classify}
        score = optimizer._score_plan(
            state,
            task_dag,
            plan,
            1,
            "new",
            10.0,
            10.0,
            seeds,
            baseline_cache,
            outcome_cache,
        )
        exhaustive.append((score, plan))

    assert result["plan"] == min(exhaustive, key=lambda item: item[0])[1]
    assert result["optimality_proven"] is True


@pytest.mark.unit
def test_optimizer_accounts_for_pending_reservations_before_admission():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=10.0, scenario_count=8, max_scenarios=8
    )
    reserved_dag = one_service_dag("edge-a")
    snapshot = {
        "captured_at": 10.0,
        "runtime_directory_revision": 1,
        "reservations": [{
            "root_uuid": "reserved",
            "source_id": 1,
            "reserved_at": 9.5,
            "runtime_directory_revision": 1,
            "plan": {"dag": reserved_dag},
        }],
        "commitments": [],
        "task_barriers": [],
        "resources": {},
    }
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        snapshot,
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-b"}


@pytest.mark.unit
def test_optimizer_excludes_pre_materialization_offered_arrival_delay():
    model = FragSpliceLatencyModel({
        "pairs": {"detect": {"edge-a": {"samples": [1.0]}}},
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=1.5, scenario_count=8, max_scenarios=8
    )
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 1.5},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        {"detect": ["edge-a"]},
        "cloud",
    )

    assert result["score"][0] == 0.0
    assert result["score"][2] == pytest.approx(1.0)
    assert result["intrinsic_slo_infeasible"] is False
    assert result["unschedulable"] is False


@pytest.mark.unit
def test_optimizer_prioritizes_incremental_latency_before_fragmentation():
    """Fragmentation cannot beat latency when SLO risk is identical."""

    class StubState:
        @staticmethod
        def simulate(candidate, seed, include_calendar=False):
            if candidate is None:
                return {
                    "latency": {"old": 1.0},
                    "deadlines": {"old": 10.0},
                    "candidate_noqueue": 0.0,
                    "replica_work": {},
                }
            device = candidate["plan"]["detect"]
            if device == "fast-fragmented":
                candidate_latency = 2.0
                candidate_noqueue = 1.0
                replica_work = {("detect", device): 2.0}
            else:
                candidate_latency = 3.0
                candidate_noqueue = 3.0
                replica_work = {("detect", device): 1.0}
            return {
                "latency": {"old": 1.0, "new": candidate_latency},
                "deadlines": {"old": 10.0, "new": 10.0},
                "candidate_noqueue": candidate_noqueue,
                "replica_work": replica_work,
            }

    common = dict(
        state=StubState(),
        dag=one_service_dag(),
        source_id=1,
        candidate_root="new",
        candidate_ready_at=0.0,
        slo=10.0,
        seeds=[1],
        baseline_cache={},
        outcome_cache={},
    )
    fast_score = FragSpliceOptimizer._score_plan(
        plan={"detect": "fast-fragmented"}, **common
    )
    slow_score = FragSpliceOptimizer._score_plan(
        plan={"detect": "slow-compact"}, **common
    )

    assert fast_score[:2] == slow_score[:2] == (0.0, 0.0)
    assert fast_score[2] < slow_score[2]
    assert fast_score[3] > slow_score[3]
    assert fast_score < slow_score


@pytest.mark.unit
def test_execution_state_models_causal_non_processor_overheads_and_exact_slo_start():
    model = FragSpliceLatencyModel({
        "pairs": {"detect": {"edge-a": {"samples": [1.0]}}},
        "handoff_pairs": {"detect": {"edge-a": {"samples": [0.1]}}},
        "transfer_pairs": {
            "detect": {"edge-a": {"samples": [0.2]}},
            "_end": {"cloud": {"samples": [0.5]}},
        },
        "dispatch_pairs": {
            "detect": {"edge-a": {"samples": [0.4]}},
        },
        "control_pairs": {
            "detect": {"edge-a": {"samples": [0.3]}},
            "_end": {"cloud": {"samples": [0.6]}},
        },
        "completion_overhead": {"1": {"samples": [0.7]}},
    })
    task_dag = one_service_dag("edge-a")
    snapshot = {
        "captured_at": 10.0,
        "runtime_directory_revision": 1,
        "reservations": [],
        "commitments": [{
            "root_uuid": "old",
            "source_id": 1,
            "source_device": "source",
            "slo_started_at": 8.0,
            "runtime_directory_revision": 1,
            "dag": task_dag,
        }],
        "task_barriers": [],
        "resources": {},
    }
    state = FragSpliceExecutionState(snapshot, model, default_slo_s=20.0)

    active = state.simulate(None, seed=7)
    empty_state = FragSpliceExecutionState(
        {**snapshot, "commitments": []}, model, default_slo_s=20.0
    )
    candidate = empty_state.simulate({
        "root": "new",
        "source": 1,
        "dag": task_dag,
        "plan": {"detect": "edge-a"},
        "ready_at": 1.0,
        "slo": 20.0,
    }, seed=7)

    # No-queue path: control .3 + transfer .2 + dispatch .4 + processing 1
    # + handoff .1 + end control .6 + end transfer .5 + completion .7.
    # A candidate that missed its offered-arrival slot exposes that lateness.
    assert candidate["candidate_noqueue"] == pytest.approx(12.8)
    assert candidate["latency"]["new"] == pytest.approx(12.8)
    # The admitted task uses its exact SLO start: 2 seconds have elapsed.
    assert active["latency"]["old"] == pytest.approx(5.8)


@pytest.mark.unit
def test_execution_state_does_not_release_candidate_before_future_arrival():
    model = FragSpliceLatencyModel({
        "pairs": {"detect": {"edge-a": {"samples": [1.0]}}},
    })
    task_dag = one_service_dag("edge-a")
    state = FragSpliceExecutionState(
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        model,
        default_slo_s=10.0,
    )

    outcome = state.simulate({
        "root": "new",
        "source": 1,
        "dag": task_dag,
        "plan": {"detect": "edge-a"},
        "ready_at": 11.0,
        "slo": 10.0,
    }, seed=7, include_service_finish=True)

    assert outcome["latency"]["new"] == pytest.approx(1.0)
    assert outcome["service_finish"]["new"]["detect"] == pytest.approx(12.0)


@pytest.mark.unit
def test_stale_queue_state_is_not_treated_as_current_busy_work():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
        queue_state_max_age_s=1.5,
    )
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {
                "edge-a": {"queue_state": {"detect": {
                    "busy": True,
                    "running_phase": "processing",
                    "phase_elapsed_s": 0.0,
                    # A remote wall clock can be ahead; freshness must use the
                    # Scheduler receive timestamp below.
                    "observed_at": 100.0,
                }}},
                "edge-b": {"queue_state": {"detect": {
                    "busy": False,
                    "observed_at": 10.0,
                }}},
            },
            "resource_received_at": {"edge-a": 5.0, "edge-b": 10.0},
            "resource_runtime_revision": {"edge-a": 1, "edge-b": 1},
        },
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-a"}


@pytest.mark.unit
def test_old_revision_queue_state_is_not_projected_on_current_replicas():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=10.0, scenario_count=8, max_scenarios=8
    )
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 2,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {
                "edge-a": {"queue_state": {"detect": {
                    "busy": True,
                    "running_phase": "processing",
                    "phase_elapsed_s": 0.0,
                    "observed_at": 10.0,
                }}},
            },
            "resource_runtime_revision": {"edge-a": 1},
        },
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-a"}


@pytest.mark.unit
def test_adaptive_search_reuses_previous_round_incumbent(monkeypatch):
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.1]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=10.0, scenario_count=8, max_scenarios=16
    )
    original_search = optimizer._search
    rounds = []

    def wrapped_search(*args, **kwargs):
        initial = copy.deepcopy(kwargs.get("initial_plan"))
        result = original_search(*args, **kwargs)
        rounds.append((initial, copy.deepcopy(result["plan"])))
        return result

    monkeypatch.setattr(optimizer, "_search", wrapped_search)
    monkeypatch.setattr(optimizer, "_ranking_is_stable", lambda *args: False)
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
            "resource_runtime_revision": {},
        },
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["scenario_count"] == 16
    assert rounds[0][0] is None
    assert rounds[1][0] == rounds[0][1]


@pytest.mark.unit
def test_handoff_phase_keeps_replica_occupied_until_controller_ack():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
        "handoff_pairs": {
            "detect": {"edge-a": {"samples": [1.0]}},
        },
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=10.0, scenario_count=8, max_scenarios=8
    )
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {
                "edge-a": {"queue_state": {"detect": {
                    "busy": True,
                    "running_phase": "handoff",
                    "phase_elapsed_s": 0.0,
                    "observed_at": 10.0,
                }}},
            },
            "resource_runtime_revision": {"edge-a": 1},
        },
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-b"}


@pytest.mark.unit
def test_busy_observation_conditions_an_exhausted_duration_sample():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model, default_slo_s=20.0, scenario_count=8, max_scenarios=8
    )
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 20.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {
                "edge-a": {"queue_state": {"detect": {
                    "busy": True,
                    "running_phase": "processing",
                    "phase_elapsed_s": 10.0,
                    "observed_at": 10.0,
                }}},
            },
            "resource_received_at": {"edge-a": 10.0},
            "resource_runtime_revision": {"edge-a": 1},
        },
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-b"}


@pytest.mark.unit
def test_execution_state_releases_join_only_after_all_branches_finish():
    fork_dag = {
        "_start": {"prev_nodes": [], "next_nodes": ["a", "b"], "service": {}},
        "a": {"prev_nodes": ["_start"], "next_nodes": ["join"],
              "service": {"execute_device": "edge-a"}},
        "b": {"prev_nodes": ["_start"], "next_nodes": ["join"],
              "service": {"execute_device": "edge-b"}},
        "join": {"prev_nodes": ["a", "b"], "next_nodes": ["_end"],
                 "service": {"execute_device": "edge-j"}},
        "_end": {"prev_nodes": ["join"], "next_nodes": [], "service": {}},
    }
    model = FragSpliceLatencyModel({
        "pairs": {
            "a": {"edge-a": {"samples": [1.0]}},
            "b": {"edge-b": {"samples": [1.0]}},
            "join": {"edge-j": {"samples": [1.0]}},
        },
    })
    state = FragSpliceExecutionState(
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [{
                "root_uuid": "old",
                "source_id": 1,
                "reserved_at": 9.0,
                "runtime_directory_revision": 1,
                "dag": fork_dag,
            }],
            "task_barriers": [{
                "root_uuid": "old",
                "barrier": "join",
                "arrived_branches": ["a"],
            }],
            "resources": {
                "edge-b": {"queue_state": {"b": {
                    "busy": True,
                    "running_task": {
                        "root_uuid": "old",
                        "runtime_directory_revision": 1,
                    },
                    "running_phase": "processing",
                    "phase_elapsed_s": 0.0,
                    "observed_at": 10.0,
                }}},
            },
            "resource_runtime_revision": {"edge-b": 1},
        },
        model,
        default_slo_s=10.0,
    )

    outcome = state.simulate(None, seed=0)
    assert outcome["latency"]["old"] == pytest.approx(3.0)


@pytest.mark.unit
def test_joint_residual_sampling_does_not_resample_content_variation_twice():
    model = FragSpliceLatencyModel({
        "pairs": {"detect": {"edge-a": {"samples": [1.0, 4.0]}}},
        "task_residuals": {
            "1": [
                {"detect": 0.0, "__shared__": 0.0},
                {"detect": math.log(4.0), "__shared__": math.log(4.0)},
            ],
        },
    })
    rng = random.Random(7)
    values = {
        model.sample_task(1, {"detect": "edge-a"}, rng)["detect"]
        for _ in range(200)
    }

    assert values == {1.0, 4.0}


@pytest.mark.unit
def test_joint_residual_sampling_uses_exponential_recency_weights():
    histories = [
        {"detect": float(index), "__shared__": float(index)}
        for index in range(17)
    ]
    model = FragSpliceLatencyModel(
        {
            "pairs": {"detect": {"edge-a": {"samples": [1.0]}}},
            "task_residuals": {"1": histories},
        },
        residual_half_life_tasks=8.0,
    )

    class CapturingRandom:
        def __init__(self):
            self.weights = None

        def choices(self, population, weights, k):
            self.weights = list(weights)
            return [population[-1]]

    rng = CapturingRandom()
    model.sample_task(1, {"detect": "edge-a"}, rng)

    assert rng.weights[-1] == pytest.approx(1.0)
    assert rng.weights[0] == pytest.approx(0.25)


@pytest.mark.unit
def test_task_quantile_estimate_uses_recent_service_residuals():
    model = FragSpliceLatencyModel(
        {
            "pairs": {"detect": {"edge-a": {"samples": [1.0]}}},
            "task_residuals": {
                "1": [
                    {"detect": 0.0},
                    {"detect": 0.0},
                    {"detect": math.log(8.0)},
                ],
            },
        },
        residual_half_life_tasks=8.0,
    )
    model.ensure_profile_context(
        configuration={"fps": 6},
        deployment={"detect": ["edge-a"]},
        dag=one_service_dag("edge-a"),
        require_complete=True,
    )
    assert model.estimate("detect", "edge-a", 0.9) == pytest.approx(1.0)
    assert model.estimate_task(1, "detect", "edge-a", 0.9) == pytest.approx(8.0)


@pytest.mark.unit
def test_rolling_task_demand_is_memoized_within_one_decision():
    class CountingModel:
        def __init__(self):
            self.calls = 0

        def estimate_task(self, source_id, service, device, quantile):
            self.calls += 1
            return 7.0

    agent = object.__new__(FragSpliceAgent)
    agent.latency_model = CountingModel()
    cache = {}

    first = agent._planning_task_demand(1, "detect", "edge-a", 0.9, cache)
    second = agent._planning_task_demand(1, "detect", "edge-a", 0.9, cache)

    assert first == second == pytest.approx(7.0)
    assert agent.latency_model.calls == 1


@pytest.mark.unit
def test_source_sample_lower_bound_is_admissible_with_drift_and_residuals():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0, 1.0]},
            },
        },
        "pair_log_drift": {
            "detect": {"edge-a": math.log(10.0)},
        },
        "task_residuals": {
            "1": [
                {"detect": math.log(0.01), "__shared__": math.log(0.01)},
                {"detect": math.log(0.02), "__shared__": math.log(0.02)},
            ],
        },
    })
    plan = {"detect": "edge-a"}
    bound = model.sample_lower_bound(1, "detect", "edge-a")
    samples = [
        model.sample_task(1, plan, random.Random(seed))["detect"]
        for seed in range(100)
    ]

    assert model.lower_bound("detect", "edge-a") == pytest.approx(10.0)
    assert bound == pytest.approx(0.1)
    assert min(samples) >= bound
    optimizer = FragSpliceOptimizer(
        model, scenario_count=8, max_scenarios=8
    )
    assert optimizer._optimistic_latency(
        one_service_dag(),
        {"detect": ["edge-a"]},
        {},
        source_id=1,
    ) == pytest.approx(bound)


@pytest.mark.unit
def test_incumbent_neighborhood_checks_high_cost_service_first():
    model = FragSpliceLatencyModel({
        "pairs": {
            "light": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.2]},
            },
            "heavy": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        scenario_count=8,
        max_scenarios=8,
        incumbent_neighborhood_size=1,
    )
    preferred = {"light": "edge-a", "heavy": "edge-a"}

    plans = optimizer._incumbent_neighborhood(
        ["light", "heavy"],
        {
            "light": ["edge-a", "edge-b"],
            "heavy": ["edge-a", "edge-b"],
        },
        preferred,
    )

    assert plans == [{"light": "edge-a", "heavy": "edge-b"}]


@pytest.mark.unit
def test_prescreen_incumbent_escapes_a_newly_congested_previous_plan():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.2]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
        incumbent_neighborhood_size=1,
    )
    task_dag = one_service_dag("edge-a")
    snapshot = {
        "captured_at": 5.0,
        "runtime_directory_revision": 1,
        "reservations": [],
        "commitments": [{
            "root_uuid": "old",
            "source_id": 1,
            "admitted_at": 4.0,
            "dag": task_dag,
        }],
        "task_barriers": [],
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "running_elapsed_s": 0.0,
                "running_task": {"root_uuid": "old"},
                "waiting_tasks": [],
                "observed_at": 5.0,
            }}},
            "edge-b": {"queue_state": {"detect": {
                "busy": False,
                "waiting_tasks": [],
                "observed_at": 5.0,
            }}},
        },
        "resource_received_at": {"edge-a": 5.0, "edge-b": 5.0},
        "resource_runtime_revision": {"edge-a": 1, "edge-b": 1},
    }
    state = FragSpliceExecutionState(
        snapshot, model, default_slo_s=10.0
    )
    seed = optimizer.random_seed
    baseline = state.simulate(None, seed, include_calendar=True)

    plan, screened, completed = optimizer._prescreen_incumbent(
        state,
        task_dag,
        ["detect"],
        {"detect": ["edge-a", "edge-b"]},
        {"detect": "edge-a"},
        1,
        "new",
        5.0,
        10.0,
        seed,
        baseline,
    )

    assert completed is True
    assert len(screened) == 2
    assert plan == {"detect": "edge-b"}


@pytest.mark.unit
def test_commitment_calendar_screening_warm_start_can_change_multiple_stages():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.0]},
            },
            "classify": {
                "edge-c": {"samples": [1.0]},
                "edge-d": {"samples": [1.0]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        scenario_count=8,
        max_scenarios=8,
        screening_beam_width=4,
    )
    task_dag = dag()
    snapshot = {
        "captured_at": 10.0,
        "runtime_directory_revision": 1,
        "reservations": [],
        "commitments": [],
        "task_barriers": [],
        "resources": {
            "edge-a": {"queue_state": {"detect": {
                "busy": True,
                "running_phase": "processing",
                "phase_elapsed_s": 0.0,
            }}},
            "edge-c": {"queue_state": {"classify": {
                "busy": True,
                "running_phase": "processing",
                "phase_elapsed_s": 0.0,
            }}},
        },
        "resource_received_at": {"edge-a": 10.0, "edge-c": 10.0},
        "resource_runtime_revision": {"edge-a": 1, "edge-c": 1},
    }
    state = FragSpliceExecutionState(snapshot, model, default_slo_s=10.0)
    seed = optimizer.random_seed
    baseline = state.simulate(None, seed, include_calendar=True)

    plans = optimizer._screening_plans(
        state,
        task_dag,
        ["detect", "classify"],
        {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-c", "edge-d"],
        },
        {"detect": "edge-a", "classify": "edge-c"},
        1,
        "new",
        10.0,
        10.0,
        seed,
        baseline,
    )

    assert plans[0] == {"detect": "edge-b", "classify": "edge-d"}


@pytest.mark.unit
def test_calendar_screening_prioritizes_late_congested_service():
    model = FragSpliceLatencyModel({
        "pairs": {
            "early": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
            "late-heavy": {
                "edge-c": {"samples": [0.8]},
                "edge-d": {"samples": [0.9]},
            },
            "fixed": {
                "edge-e": {"samples": [2.0]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(model)
    order = optimizer._screening_service_order(
        ["early", "fixed", "late-heavy"],
        {
            "early": ["edge-a", "edge-b"],
            "fixed": ["edge-e"],
            "late-heavy": ["edge-c", "edge-d"],
        },
        {
            "early": "edge-a",
            "fixed": "edge-e",
            "late-heavy": "edge-c",
        },
        {
            "replica_intervals": {
                ("late-heavy", "edge-c"): [(10.0, 16.0)],
            },
        },
        now=10.0,
    )

    assert order == ["late-heavy", "early"]


@pytest.mark.unit
def test_candidate_only_sampling_preserves_common_random_numbers():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {"edge-a": {"samples": [0.5, 1.0]}},
            "classify": {"edge-b": {"samples": [0.25, 0.75]}},
        },
    })
    state = FragSpliceExecutionState(
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        model,
        default_slo_s=10.0,
    )
    candidate = {
        "root": "new",
        "source": 1,
        "dag": dag(),
        "plan": {"detect": "edge-a", "classify": "edge-b"},
        "slo": 10.0,
    }

    from_full_simulation_path = state._sample_roots(candidate, 17)[-1]
    candidate_only = state._sample_candidate_root(candidate, 17)

    assert candidate_only["durations"] == from_full_simulation_path["durations"]
    assert candidate_only["handoffs"] == from_full_simulation_path["handoffs"]
    assert candidate_only["overheads"] == from_full_simulation_path["overheads"]


@pytest.mark.unit
def test_screening_deadline_returns_a_complete_preferred_plan():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {"edge-a": {"samples": [1.0]}},
            "classify": {"edge-b": {"samples": [1.0]}},
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        scenario_count=8,
        max_scenarios=8,
    )
    preferred = {"detect": "edge-a", "classify": "edge-b"}

    plans = optimizer._screening_plans(
        SimpleNamespace(),
        dag(),
        ["detect", "classify"],
        {"detect": ["edge-a"], "classify": ["edge-b"]},
        preferred,
        1,
        "new",
        10.0,
        10.0,
        0,
        {},
        deadline=time.monotonic() - 1.0,
    )

    assert plans == [preferred]


@pytest.mark.unit
def test_tight_budget_still_scores_incumbent_after_bounded_screening(monkeypatch):
    original = FragSpliceExecutionState.screen_candidate

    def slow_screen(self, *args, **kwargs):
        time.sleep(0.04)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        FragSpliceExecutionState,
        "screen_candidate",
        slow_screen,
    )
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
            "classify": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
        search_time_limit_s=0.12,
    )

    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-a", "edge-b"],
        },
        "cloud",
    )

    assert result["score_evaluated"] is True
    assert result["prediction_complete"] is True
    assert result["selected_outcome_scenarios"] == 8
    assert result["screening_completed"] is False
    assert result["fallback_reason"] != "budget_exhausted_during_screening"


@pytest.mark.unit
def test_first_exact_evaluation_uses_full_dag_calendar_screening(monkeypatch):
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [1.0]},
            },
            "classify": {
                "edge-c": {"samples": [1.0]},
                "edge-d": {"samples": [1.0]},
            },
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        scenario_count=8,
        max_scenarios=8,
        search_time_limit_s=0.12,
        screening_beam_width=4,
    )
    original_score = optimizer._score_plan
    scored_plans = []

    def record_score(*args, **kwargs):
        scored_plans.append(copy.deepcopy(args[2]))
        return original_score(*args, **kwargs)

    monkeypatch.setattr(optimizer, "_score_plan", record_score)
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {
                "edge-a": {"queue_state": {"detect": {
                    "busy": True,
                    "running_phase": "processing",
                    "phase_elapsed_s": 0.0,
                    "observed_at": 10.0,
                }}},
                "edge-c": {"queue_state": {"classify": {
                    "busy": True,
                    "running_phase": "processing",
                    "phase_elapsed_s": 0.0,
                    "observed_at": 10.0,
                }}},
            },
            "resource_received_at": {
                "edge-a": 10.0,
                "edge-c": 10.0,
            },
            "resource_runtime_revision": {
                "edge-a": 1,
                "edge-c": 1,
            },
        },
        {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-c", "edge-d"],
        },
        "cloud",
        initial_plan={"detect": "edge-a", "classify": "edge-c"},
    )

    assert scored_plans
    assert scored_plans[0] == {
        "detect": "edge-b",
        "classify": "edge-d",
    }
    assert result["score_evaluated"] is True


@pytest.mark.unit
def test_incomplete_scenario_refinement_preserves_complete_incumbent(
    monkeypatch,
):
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {"edge-a": {"samples": [0.1]}},
            "classify": {"edge-a": {"samples": [0.1]}},
        },
    })
    optimizer = FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=16,
    )
    plan = {"detect": "edge-a", "classify": "edge-a"}
    score = (0.0, 0.0, 0.2, 0.0, 0.2)

    def fake_search(
        state,
        task_dag,
        candidates,
        source_id,
        candidate_root,
        candidate_ready_at,
        slo,
        seeds,
        deadline,
        baseline_cache,
        outcome_cache,
        initial_plan=None,
    ):
        del (
            state,
            task_dag,
            candidates,
            source_id,
            candidate_ready_at,
            slo,
            deadline,
            baseline_cache,
            initial_plan,
        )
        if len(seeds) == 8:
            key = tuple(sorted(plan.items()))
            for seed in seeds:
                outcome_cache[(key, seed)] = {
                    "latency": {candidate_root: 0.2},
                }
            return {
                "plan": dict(plan),
                "score": score,
                "evaluated": [(score, dict(plan))],
                "expanded": 1,
                "screened": 1,
                "screening_completed": True,
                "score_evaluated": True,
                "fallback_reason": "",
                "optimality_proven": True,
                "best_open_lower_bound": score,
            }
        return {
            "plan": dict(plan),
            "score": score,
            "evaluated": [],
            "expanded": 0,
            "screened": 0,
            "screening_completed": False,
            "score_evaluated": False,
            "fallback_reason": "budget_exhausted_during_incumbent_evaluation",
            "optimality_proven": False,
            "best_open_lower_bound": (0.0, 0.0, 0.1, 0.0, 0.0),
        }

    monkeypatch.setattr(optimizer, "_search", fake_search)
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        {
            "detect": ["edge-a"],
            "classify": ["edge-a"],
        },
        "cloud",
    )

    assert result["scenario_count"] == 8
    assert result["selected_outcome_scenarios"] == 8
    assert result["score_evaluated"] is True
    assert result["prediction_complete"] is True
    assert result["scenario_refinement_exhausted"] is True
    assert result["fallback_reason"] == (
        "budget_exhausted_during_scenario_refinement"
    )


@pytest.mark.unit
def test_whole_decision_budget_bounds_large_commitment_fallback():
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
            "classify": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
        },
    })
    commitments = []
    for index in range(50):
        committed_dag = dag(
            "edge-a" if index % 2 == 0 else "edge-b"
        )
        commitments.append({
            "root_uuid": f"old-{index}",
            "source_id": 1,
            "reserved_at": 9.0 + index * 0.001,
            "runtime_directory_revision": 1,
            "dag": committed_dag,
        })
    optimizer = FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
        search_time_limit_s=1e-9,
    )
    started = time.monotonic()
    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": commitments,
            "task_barriers": [],
            "resources": {},
        },
        {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-a", "edge-b"],
        },
        "cloud",
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.25
    assert set(result["plan"]) == {"detect", "classify"}
    assert result["score_evaluated"] is False
    assert result["budget_exhausted"] is True
    assert result["prediction_complete"] is False
    assert result["selected_outcome_scenarios"] == 0
    assert result["fallback_reason"] == (
        "budget_exhausted_before_state_evaluation"
    )


@pytest.mark.unit
def test_scenario_scoring_stops_at_budget_boundary(monkeypatch):
    import core.lib.algorithms.schedule_agent.fragsplice.optimizer as optimizer_module

    original_state = optimizer_module.FragSpliceExecutionState

    class SlowExecutionState(original_state):
        def simulate(self, *args, **kwargs):
            time.sleep(0.01)
            return super().simulate(*args, **kwargs)

    monkeypatch.setattr(
        optimizer_module,
        "FragSpliceExecutionState",
        SlowExecutionState,
    )
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
            "classify": {
                "edge-a": {"samples": [0.1]},
                "edge-b": {"samples": [0.1]},
            },
        },
    })
    optimizer = optimizer_module.FragSpliceOptimizer(
        model,
        default_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
        search_time_limit_s=0.035,
    )

    result = optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "task_context": {"root_uuid": "new"},
            "dag": dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        {
            "captured_at": 10.0,
            "runtime_directory_revision": 1,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-a", "edge-b"],
        },
        "cloud",
    )

    assert result["search_seconds"] < 0.12
    assert result["budget_overrun_seconds"] < 0.08
    assert result["score_evaluated"] is False
    assert result["prediction_complete"] is False
    assert 0 < result["selected_outcome_scenarios"] < 8
    assert result["fallback_reason"] == (
        "budget_exhausted_during_incumbent_evaluation"
    )


class FakeService:
    def __init__(self, device, duration, timing=None):
        self.device = device
        self.duration = duration
        self.timing = dict(timing or {})

    def get_execute_device(self):
        return self.device

    def get_real_execute_time(self):
        return self.duration

    def get_tmp_data(self):
        return self.timing


class FakeTask:
    def __init__(self, task_id, assignments, dag_value=None, deployment=None):
        self.task_id = task_id
        self.dag_value = copy.deepcopy(dag_value or dag())
        self.deployment = copy.deepcopy(
            deployment or FakeSystem().runtime_service_nodes()
        )
        self.services = {
            name: FakeService(device, 0.1 + 0.01 * task_id)
            for name, device in assignments.items()
        }
        self.graph = SimpleNamespace(
            nodes={
                "_start": None,
                **{name: None for name in assignments},
                "_end": None,
            },
            to_dict=lambda: copy.deepcopy(self.dag_value),
        )

    def get_dag(self):
        return self.graph

    def get_service(self, name):
        return self.services[name]

    def get_deployment(self):
        return copy.deepcopy(self.deployment)

    def get_source_id(self):
        return 1

    def get_task_id(self):
        return self.task_id

    def get_root_uuid(self):
        return f"root-{self.task_id}"

    def get_slo_end_time(self):
        return 0.0


class FakeSystem:
    cloud_device = "cloud"

    def __init__(self, revision=1):
        self.revision = revision

    def runtime_service_nodes(self):
        return {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-c"],
        }

    def runtime_directory_revision(self):
        return self.revision

    def get_scheduling_snapshot(self, scope=None):
        return {
            "captured_at": 1.0,
            "runtime_directory_revision": self.revision,
            "deployment": self.runtime_service_nodes(),
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
            "resource_received_at": {},
            "resource_runtime_revision": {},
        }


@pytest.mark.unit
def test_cold_overhead_uses_actual_start_delivery_not_offered_arrival():
    task_dag = one_service_dag("edge-a")
    services = {
        "_start": FakeService(
            "source",
            0.0,
            {"transmit_start": 10.0, "transmit_end": 10.1},
        ),
        "detect": FakeService(
            "edge-a",
            1.0,
            {
                "transmit_start": 10.3,
                "transmit_end": 10.4,
                "execute_start": 10.4,
                "real_execute_start": 10.5,
                "real_execute_end": 11.5,
                "execute_end": 11.6,
            },
        ),
        "_end": FakeService(
            "cloud",
            0.0,
            {"transmit_start": 11.8, "transmit_end": 11.9},
        ),
    }

    class ColdTask:
        def get_dag(self):
            return SimpleNamespace(to_dict=lambda: copy.deepcopy(task_dag))

        def get_service(self, name):
            return services[name]

        def get_source_id(self):
            return 1

        def get_slo_start_time(self):
            # The task was offered long before the single-in-flight cold
            # profiler actually admitted it.
            return 1.0

        def get_slo_end_time(self):
            return 12.0

    model = FragSpliceLatencyModel()
    assert model.record_task_overheads(ColdTask()) is True
    # With no explicit transfer metric the recorder anchors the release at
    # execute_start (10.4), yielding the actual 0.3-second Controller delay;
    # it must not charge the 9.4 seconds since the offered arrival at 1.0.
    assert model.control_values("detect", "edge-a") == pytest.approx([0.3])


@pytest.mark.unit
def test_online_feedback_freezes_cold_non_processor_distributions(monkeypatch):
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {"edge-a": {"samples": [0.1]}},
            "classify": {"edge-c": {"samples": [0.2]}},
        },
        "handoff_pairs": {
            "detect": {"edge-a": {"samples": [0.01]}},
            "classify": {"edge-c": {"samples": [0.02]}},
        },
        "transfer_pairs": {
            "detect": {"edge-a": {"samples": [0.03]}},
        },
        "dispatch_pairs": {
            "detect": {"edge-a": {"samples": [0.04]}},
        },
        "control_pairs": {
            "detect": {"edge-a": {"samples": [0.05]}},
        },
        "completion_overhead": {
            "1": {"samples": [0.06]},
        },
    })
    task = FakeTask(
        1,
        {"detect": "edge-a", "classify": "edge-c"},
    )
    task.services["detect"].timing = {
        "real_execute_end": 1.0,
        "execute_end": 1001.0,
    }

    def unexpected_online_overhead_update(*args, **kwargs):
        raise AssertionError(
            "online traces must not update cold non-Processor distributions"
        )

    monkeypatch.setattr(
        model,
        "record_task_overheads",
        unexpected_online_overhead_update,
    )
    assert model.update_task(task) is True

    assert model.handoff_values("detect", "edge-a") == [0.01]
    assert model.transfer_values("detect", "edge-a") == [0.03]
    assert model.dispatch_values("detect", "edge-a") == [0.04]
    assert model.control_values("detect", "edge-a") == [0.05]
    assert model.completion_values(1) == [0.06]
    assert model._task_residuals["1"]
    # A fresh source first establishes its service-content scale; one task is
    # not enough evidence to relabel content variation as pair drift.
    assert model._pair_log_drift["detect"]["edge-a"] == pytest.approx(0.0)


@pytest.mark.unit
def test_online_feedback_keeps_service_content_out_of_pair_drift():
    hard_residual = math.log(10.0)
    model = FragSpliceLatencyModel({
        "pairs": {
            "detect": {"edge-a": {"samples": [1.0]}},
            "classify": {"edge-c": {"samples": [1.0]}},
        },
        "task_residuals": {
            "1": [
                {
                    "detect": 0.0,
                    "classify": hard_residual,
                    "__shared__": hard_residual / 2.0,
                }
                for _ in range(8)
            ],
        },
    })
    task = FakeTask(1, {"detect": "edge-a", "classify": "edge-c"})
    task.services["detect"].duration = 1.0
    task.services["classify"].duration = 10.0

    assert model.update_task(task) is True

    assert model._pair_log_drift["detect"]["edge-a"] == pytest.approx(0.0)
    assert model._pair_log_drift["classify"]["edge-c"] == pytest.approx(0.0)
    latest = model._task_residuals["1"][-1]
    assert latest["detect"] == pytest.approx(0.0)
    assert latest["classify"] == pytest.approx(hard_residual)


def contextual_profile(configuration, deployment, dag_value, pairs, **extra):
    pairs = {
        str(service): {
            str(device): (
                copy.deepcopy(value)
                if isinstance(value, dict)
                else {"samples": list(value)}
            )
            for device, value in devices.items()
        }
        for service, devices in pairs.items()
    }
    profile = {
        "version": FragSpliceLatencyModel.PROFILE_VERSION,
        "metric": FragSpliceLatencyModel.PROFILE_METRIC,
        "context": FragSpliceLatencyModel.build_profile_context(
            configuration,
            deployment,
            dag_value,
        ),
        "pairs": pairs,
    }
    profile.update(extra)
    return profile


@pytest.mark.unit
def test_ablation_agents_disable_exactly_one_fragsplice_module(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    configuration = {"fps": 6}
    deployment = FakeSystem().runtime_service_nodes()
    task_dag = dag()
    profile = contextual_profile(
        configuration,
        deployment,
        task_dag,
        {
            "detect": {"edge-a": [0.1], "edge-b": [0.2]},
            "classify": {"edge-c": [0.1]},
        },
    )

    no_profiler = FragSpliceNoDistributionProfilerAgent(
        FakeSystem(revision=101),
        agent_id=101,
        configuration=configuration,
        scenario_count=8,
        max_scenarios=8,
    )
    assert isinstance(
        no_profiler.latency_model,
        FragSpliceRandomLatencyModel,
    )
    assert no_profiler.distribution_profiler_enabled is False
    assert no_profiler.future_state_estimator_enabled is True
    assert no_profiler.plan_optimizer_enabled is True
    assert type(no_profiler.optimizer) is FragSpliceRandomInputOptimizer
    assert no_profiler.optimizer.latency_model is no_profiler.latency_model
    no_profiler.update_task(FakeTask(
        1,
        {"detect": "edge-a", "classify": "edge-c"},
        dag_value=task_dag,
        deployment=deployment,
    ))
    no_profiler.update_task(FakeTask(
        9,
        {"detect": "edge-b", "classify": "edge-c"},
        dag_value=task_dag,
        deployment=deployment,
    ))
    assert no_profiler._feedback_count == 0

    rolling_payload = no_profiler._rolling_payload(
        {
            "info": {
                "source_id": 1,
                "dag": task_dag,
                "task_context": {
                    "task_id": 9,
                },
            },
            "generation": 1,
            "initial_plan": None,
        },
        {"captured_at": 2.0},
        deployment,
    )
    assert "latency_profile" not in rolling_payload
    assert rolling_payload["random_latency_state"] == (
        no_profiler.latency_model.to_state()
    )
    assert rolling_payload["plan_optimizer_enabled"] is True
    assert rolling_payload["optimizer_parameters"][
        "random_workload_token_min"
    ] == 0
    assert rolling_payload["optimizer_parameters"][
        "random_workload_token_max"
    ] == 8

    with pytest.raises(ValueError, match="must not receive a latency_profile"):
        FragSpliceNoDistributionProfilerAgent(
            FakeSystem(revision=111),
            agent_id=111,
            configuration=configuration,
            latency_profile=profile,
            scenario_count=8,
            max_scenarios=8,
        )

    no_optimizer = FragSpliceNoPlanOptimizerAgent(
        FakeSystem(revision=102),
        agent_id=102,
        configuration=configuration,
        latency_profile=profile,
        scenario_count=8,
        max_scenarios=8,
    )
    assert no_optimizer.optimizer.latency_model is no_optimizer.latency_model
    assert no_optimizer.distribution_profiler_enabled is True
    assert no_optimizer.future_state_estimator_enabled is True
    assert no_optimizer.plan_optimizer_enabled is False
    assert isinstance(
        no_optimizer.optimizer,
        FragSpliceStagewiseEFTOptimizer,
    )


@pytest.mark.unit
def test_main_agent_returns_complete_plan_without_mutating_request(tmp_path, monkeypatch):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    original = dag()
    agent = FragSpliceAgent(
        FakeSystem(),
        agent_id=1,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            FakeSystem().runtime_service_nodes(),
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        latency_slo_s=10.0,
        scenario_count=8,
        max_scenarios=8,
    )
    plan = agent.get_schedule_plan({
        "source_id": 1,
        "source_device": "source",
        "all_edge_devices": ["edge-a", "edge-b", "edge-c"],
        "dag": original,
        "meta_data": {},
    })

    assert original["detect"]["service"]["execute_device"] == ""
    assert plan["dag"]["detect"]["service"]["execute_device"] == "edge-a"
    assert plan["dag"]["classify"]["service"]["execute_device"] == "edge-c"
    assert plan["dag"]["_start"]["service"]["execute_device"] == "source"
    assert plan["dag"]["_end"]["service"]["execute_device"] == "cloud"
    assert agent.last_decision["optimality_proven"] is True


def _stub_optimizer_result(plan, search_seconds=0.01):
    return {
        "plan": copy.deepcopy(plan),
        "candidate_count": 2,
        "screened": 2,
        "scenario_count": 8,
        "selected_outcome_scenarios": 8,
        "evaluated": [((), copy.deepcopy(plan))],
        "expanded": 2,
        "optimality_proven": False,
        "unschedulable": False,
        "intrinsic_slo_infeasible": False,
        "budget_exhausted": False,
        "scenario_refinement_exhausted": False,
        "score_evaluated": True,
        "prediction_complete": True,
        "fallback_reason": "",
        "predicted_miss_probability": 0.0,
        "score": (0.0, 0.0, 0.2, 0.0, 0.0),
        "best_open_lower_bound": (0.0, 0.0, 0.1, 0.0, 0.0),
        "state_build_seconds": 0.001,
        "search_seconds": search_seconds,
        "budget_overrun_seconds": 0.0,
    }


@pytest.mark.unit
def test_rolling_planner_serves_next_task_from_background_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })

    class RollingSystem(FakeSystem):
        def __init__(self):
            super().__init__(revision=41)
            self.reserved_roots = {"root-1"}

        def get_scheduling_snapshot(self):
            return {
                "captured_at": time.time(),
                "runtime_directory_revision": self.revision,
                "deployment": self.runtime_service_nodes(),
                "reservations": [
                    {"root_uuid": root}
                    for root in sorted(self.reserved_roots)
                ],
                "commitments": [],
                "task_barriers": [],
                "resources": {},
            }

    system = RollingSystem()
    original = dag()
    agent = FragSpliceAgent(
        system,
        agent_id=41,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            system.runtime_service_nodes(),
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_reservation_wait_s=0.0,
        rolling_use_process=False,
    )
    synchronous_calls = []

    def synchronous_solve(*args, **kwargs):
        synchronous_calls.append(1)
        return _stub_optimizer_result({
            "detect": "edge-a",
            "classify": "edge-c",
        })

    def background_solve(payload):
        assert "_candidate_arrival_at" not in payload["info"]
        assert "task_arrival_interval_s" not in payload["info"]
        assert "created_at" not in payload["info"]["task_context"]
        return _stub_optimizer_result({
            "detect": "edge-b",
            "classify": "edge-c",
        })

    monkeypatch.setattr(agent.optimizer, "solve", synchronous_solve)
    monkeypatch.setattr(
        fragsplice_agent_module,
        "_solve_rolling_plan",
        background_solve,
    )
    first = agent.get_schedule_plan({
        "source_id": 1,
        "source_device": "source",
        "all_edge_devices": ["edge-a", "edge-b", "edge-c"],
        "dag": original,
        "meta_data": {},
        "task_context": {
            "task_id": 1,
            "root_uuid": "root-1",
        },
    })
    assert first["dag"]["detect"]["service"]["execute_device"] == "edge-a"

    deadline = time.monotonic() + 1.0
    while agent._rolling_cache is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert agent._rolling_cache is not None

    second = agent.get_schedule_plan({
        "source_id": 1,
        "source_device": "source",
        "all_edge_devices": ["edge-a", "edge-b", "edge-c"],
        "dag": original,
        "meta_data": {},
        "task_context": {
            "task_id": 2,
            "root_uuid": "root-2",
        },
    })

    assert len(synchronous_calls) == 1
    assert second["dag"]["detect"]["service"]["execute_device"] == "edge-b"
    assert agent.last_decision["planner_mode"] == "rolling_cache"


@pytest.mark.unit
def test_rolling_fallback_reranks_plan_proposal_pool_after_close_commitment(
    tmp_path, monkeypatch
):
    """A close second arrival must not blindly reuse the first full-DAG plan."""

    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    system = FakeSystem(revision=44)
    original = dag()
    agent = FragSpliceAgent(
        system,
        agent_id=44,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            system.runtime_service_nodes(),
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        latency_slo_s=1.0,
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_use_process=False,
    )
    plan_a = {"detect": "edge-a", "classify": "edge-c"}
    plan_b = {"detect": "edge-b", "classify": "edge-c"}

    def synchronous_solve(*args, **kwargs):
        result = _stub_optimizer_result(plan_a)
        result["candidate_pool"] = [
            {
                "plan": copy.deepcopy(plan_a),
                "screen_score": (0.0, 0.0, 0.2, 0.0, 0.1),
            },
            {
                "plan": copy.deepcopy(plan_b),
                "screen_score": (0.0, 0.0, 0.3, 0.0, 0.2),
            },
        ]
        return result

    monkeypatch.setattr(agent.optimizer, "solve", synchronous_solve)
    # Keep the background search unavailable so the second request exercises
    # the exact rolling-fallback path that occurs inside a microburst.
    monkeypatch.setattr(agent, "_ensure_rolling_worker", lambda: None)

    first = agent.get_schedule_plan({
        "source_id": 1,
        "source_device": "source",
        "dag": original,
        "meta_data": {"slo_seconds": 1.0},
        "task_context": {
            "task_id": 1,
            "root_uuid": "close-root-1",
        },
    })
    second = agent.get_schedule_plan({
        "source_id": 1,
        "source_device": "source",
        "dag": original,
        "meta_data": {"slo_seconds": 1.0},
        "task_context": {
            "task_id": 2,
            "root_uuid": "close-root-2",
        },
    })

    assert first["dag"]["detect"]["service"]["execute_device"] == "edge-a"
    assert second["dag"]["detect"]["service"]["execute_device"] == "edge-b"
    assert agent.last_decision["online_rerank_delta_tasks"] == 1
    assert agent.last_decision["online_rerank_changed"] is True
    assert agent.last_decision["fallback_reason"] == (
        "rolling_plan_not_ready_commitment_rerank"
    )


@pytest.mark.unit
def test_rolling_fallback_repairs_multiple_congested_services_together(
    tmp_path, monkeypatch
):
    """A microburst repair must not be limited to one stage exchange."""

    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    deployment = {
        "detect": ["edge-a", "edge-b"],
        "classify": ["edge-c", "edge-d"],
    }
    system = FakeSystem(revision=47)
    system.runtime_service_nodes = lambda: copy.deepcopy(deployment)
    original = dag()
    agent = FragSpliceAgent(
        system,
        agent_id=47,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            deployment,
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.1]},
                "classify": {"edge-c": [0.1], "edge-d": [0.1]},
            },
        ),
        latency_slo_s=1.0,
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_use_process=False,
    )
    base = {"detect": "edge-a", "classify": "edge-c"}

    def synchronous_solve(*args, **kwargs):
        result = _stub_optimizer_result(base)
        result["candidate_pool"] = [{
            "plan": copy.deepcopy(base),
            "screen_score": (0.0, 0.0, 0.2, 0.0, 0.1),
        }]
        return result

    monkeypatch.setattr(agent.optimizer, "solve", synchronous_solve)
    monkeypatch.setattr(agent, "_ensure_rolling_worker", lambda: None)

    def request(task_id):
        return agent.get_schedule_plan({
            "source_id": 1,
            "source_device": "source",
            "dag": original,
            "meta_data": {"slo_seconds": 1.0},
            "task_context": {
                "task_id": task_id,
                "root_uuid": f"joint-root-{task_id}",
            },
        })

    first = request(1)
    second = request(2)

    assert first["dag"]["detect"]["service"]["execute_device"] == "edge-a"
    assert first["dag"]["classify"]["service"]["execute_device"] == "edge-c"
    assert second["dag"]["detect"]["service"]["execute_device"] == "edge-b"
    assert second["dag"]["classify"]["service"]["execute_device"] == "edge-d"
    assert agent.last_decision["online_active_commitment_tasks"] == 1
    assert agent.last_decision["online_repair_plan_count"] >= 2


@pytest.mark.unit
def test_rolling_plan_pool_bounds_background_and_covers_each_replica(
    tmp_path, monkeypatch
):
    """The fast path keeps every direct replica escape before beam entries."""

    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    system = FakeSystem(revision=45)
    original = dag()
    agent = FragSpliceAgent(
        system,
        agent_id=45,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            system.runtime_service_nodes(),
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_use_process=False,
    )
    agent.optimizer.screening_beam_width = 1
    selected = {"detect": "edge-a", "classify": "edge-c"}
    result = _stub_optimizer_result(selected)
    result["candidate_pool"] = [
        {
            "plan": {"detect": "edge-a", "classify": "edge-c"},
            "screen_score": (0.0, 0.0, float(index + 1), 0.0, 0.0),
        }
        for index in range(40)
    ]

    pool = agent._rolling_plan_pool(
        result,
        {
            "detect": ["edge-a", "edge-b"],
            "classify": ["edge-c"],
        },
    )

    assert len(pool) <= 16
    assert any(item["plan"]["detect"] == "edge-b" for item in pool)


@pytest.mark.unit
def test_rolling_plan_pool_upgrades_neighbor_score_and_uses_safe_fallback(
    tmp_path, monkeypatch
):
    """A direct escape keeps a real beam score or the selected baseline."""

    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    system = FakeSystem(revision=46)
    deployment = {
        "detect": ["edge-a", "edge-b", "edge-d"],
        "classify": ["edge-c"],
    }
    system.runtime_service_nodes = lambda: copy.deepcopy(deployment)
    original = dag()
    agent = FragSpliceAgent(
        system,
        agent_id=46,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            system.runtime_service_nodes(),
            original,
            {
                "detect": {
                    "edge-a": [0.1],
                    "edge-b": [0.2],
                    "edge-d": [0.3],
                },
                "classify": {"edge-c": [0.1]},
            },
        ),
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_use_process=False,
    )
    selected = {"detect": "edge-a", "classify": "edge-c"}
    scored_neighbor = {"detect": "edge-b", "classify": "edge-c"}
    unscored_neighbor = {"detect": "edge-d", "classify": "edge-c"}
    result = _stub_optimizer_result(selected)
    real_neighbor_score = (0.0, 0.0, 0.7, 0.5, 0.6)
    result["candidate_pool"] = [{
        "plan": copy.deepcopy(scored_neighbor),
        "screen_score": real_neighbor_score,
    }]

    pool = agent._rolling_plan_pool(
        result,
        deployment,
    )
    by_plan = {
        tuple(sorted(item["plan"].items())): item["screen_score"]
        for item in pool
    }

    assert by_plan[tuple(sorted(scored_neighbor.items()))] == real_neighbor_score
    assert by_plan[tuple(sorted(unscored_neighbor.items()))] == result["score"]


@pytest.mark.unit
def test_no_future_state_estimator_validates_commit_before_hiding_future_work(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })

    class CurrentStateSystem(FakeSystem):
        def __init__(self):
            super().__init__(revision=42)

        def get_scheduling_snapshot(self):
            return {
                "captured_at": time.time(),
                "runtime_directory_revision": self.revision,
                "deployment": self.runtime_service_nodes(),
                "reservations": [{"root_uuid": "root-1"}],
                "commitments": [{"root_uuid": "older-root"}],
                "task_barriers": [{"root_uuid": "older-root"}],
                "resources": {
                    "edge-a": {
                        "queue_state": {
                            "detect": {"waiting_count": 3},
                        },
                    },
                },
            }

    system = CurrentStateSystem()
    original = dag()
    agent = FragSpliceNoFutureStateEstimatorAgent(
        system,
        agent_id=42,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            system.runtime_service_nodes(),
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_reservation_wait_s=0.0,
        rolling_use_process=False,
    )

    snapshot = agent._wait_for_committed_snapshot({
        "after_root": "root-1",
    })

    assert snapshot is not None
    assert snapshot["reservations"] == []
    assert snapshot["commitments"] == []
    assert snapshot["task_barriers"] == []
    assert snapshot["resources"]["edge-a"]["queue_state"]["detect"] == {
        "waiting_count": 3,
    }


@pytest.mark.unit
def test_rolling_planner_consumes_bounded_late_result_once(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    system = FakeSystem(revision=43)
    task_dag = dag()
    deployment = system.runtime_service_nodes()
    agent = FragSpliceAgent(
        system,
        agent_id=43,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            deployment,
            task_dag,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        scenario_count=8,
        max_scenarios=8,
        rolling_planner_enabled=True,
        rolling_plan_max_lag_tasks=2,
        rolling_use_process=False,
    )
    info = {"source_id": 1, "dag": task_dag}
    agent._rolling_generation = 3
    agent._rolling_cache = {
        "generation": 1,
        "revision": 43,
        "source_id": 1,
        "dag_signature": agent._dag_signature(task_dag),
        "deployment": deployment,
        "result": _stub_optimizer_result({
            "detect": "edge-b",
            "classify": "edge-c",
        }),
        "published_monotonic": time.monotonic(),
        "background_seconds": 0.4,
    }

    result, cached_deployment, _ = agent._cached_result(info, 43)

    assert result["plan"]["detect"] == "edge-b"
    assert result["rolling_cache_lag_tasks"] == 2
    assert cached_deployment == deployment
    assert agent._cached_result(info, 43) == (None, None, None)


@pytest.mark.unit
def test_optimizer_anchors_candidate_at_snapshot_time(monkeypatch):
    model = FragSpliceLatencyModel({
        "pairs": {"detect": {"edge-a": {"samples": [0.1]}}},
    })
    optimizer = FragSpliceOptimizer(
        model,
        scenario_count=8,
        max_scenarios=8,
    )
    captured = {}

    def fake_search(
        state,
        task_dag,
        candidates,
        source_id,
        candidate_root,
        candidate_ready_at,
        slo,
        seeds,
        deadline,
        baseline_cache,
        outcome_cache,
        initial_plan=None,
    ):
        captured["ready_at"] = candidate_ready_at
        plan = {"detect": "edge-a"}
        return _stub_optimizer_result(plan)

    monkeypatch.setattr(optimizer, "_search", fake_search)
    optimizer.solve(
        {
            "source_id": 1,
            "source_device": "source",
            "dag": one_service_dag(),
            "meta_data": {},
            "task_context": {"root_uuid": "future-root"},
        },
        {
            "captured_at": 10.0,
            "reservations": [],
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        },
        {"detect": ["edge-a"]},
        "cloud",
    )

    assert captured["ready_at"] == pytest.approx(10.0)


@pytest.mark.unit
def test_no_future_state_estimator_removes_only_future_commitments(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })

    class CommitmentSystem(FakeSystem):
        def get_scheduling_snapshot(self):
            return {
                "captured_at": 1.0,
                "deployment": self.runtime_service_nodes(),
                "reservations": [{"root_uuid": "pending"}],
                "commitments": [{"root_uuid": "active"}],
                "task_barriers": [{"root_uuid": "active"}],
                "resources": {"edge-a": {"queue_state": {"detect": {
                    "waiting_count": 2,
                }}}},
            }

    system = CommitmentSystem(revision=31)
    original = dag()
    agent = FragSpliceNoFutureStateEstimatorAgent(
        system,
        agent_id=31,
        configuration={"fps": 6},
        latency_profile=contextual_profile(
            {"fps": 6},
            system.runtime_service_nodes(),
            original,
            {
                "detect": {"edge-a": [0.1], "edge-b": [0.2]},
                "classify": {"edge-c": [0.1]},
            },
        ),
        scenario_count=8,
        max_scenarios=8,
    )
    assert agent.optimizer.latency_model is agent.latency_model
    assert agent.distribution_profiler_enabled is True
    assert type(agent.optimizer) is FragSpliceOptimizer
    assert agent.future_state_estimator_enabled is False
    assert agent.plan_optimizer_enabled is True
    captured = {}

    def fake_solve(info, snapshot, deployment, cloud_device, initial_plan=None):
        captured["snapshot"] = snapshot
        return {
            "plan": {"detect": "edge-a", "classify": "edge-c"},
            "candidate_count": 2,
            "screened": 2,
            "scenario_count": 8,
            "evaluated": [((0.0,), {})],
            "optimality_proven": True,
            "unschedulable": False,
            "intrinsic_slo_infeasible": False,
            "budget_exhausted": False,
            "predicted_miss_probability": 0.0,
            "score": (0.0, 0.0, 0.0, 0.0, 0.2),
            "best_open_lower_bound": (0.0, 0.0, 0.0, 0.0, 0.2),
            "search_seconds": 0.01,
        }

    monkeypatch.setattr(agent.optimizer, "solve", fake_solve)
    agent.get_schedule_plan({
        "source_id": 1,
        "source_device": "source",
        "dag": original,
        "meta_data": {},
    })

    snapshot = captured["snapshot"]
    assert snapshot["reservations"] == []
    assert snapshot["commitments"] == []
    assert snapshot["task_barriers"] == []
    assert snapshot["resources"]["edge-a"]["queue_state"]["detect"][
        "waiting_count"
    ] == 2


@pytest.mark.unit
def test_cold_sampler_profiles_only_pairs_in_fixed_deployment(tmp_path, monkeypatch):
    parameters = {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    }
    monkeypatch.setattr(Context, "parameters", parameters)
    profile_path = tmp_path / "fragsplice.json"
    agent = FragSpliceColdSampleAgent(
        FakeSystem(),
        agent_id=1,
        configuration={"fps": 6},
        profile_path=str(profile_path),
        warmup_samples=0,
        samples_per_pair=1,
    )
    info = {
        "source_device": "source",
        "dag": dag(),
    }

    first = agent.get_schedule_plan(info)
    first_assignments = {
        name: first["dag"][name]["service"]["execute_device"]
        for name in ("detect", "classify")
    }
    agent.update_task(FakeTask(1, first_assignments))
    second = agent.get_schedule_plan(info)
    second_assignments = {
        name: second["dag"][name]["service"]["execute_device"]
        for name in ("detect", "classify")
    }
    agent.update_task(FakeTask(2, second_assignments))

    assert {first_assignments["detect"], second_assignments["detect"]} == {
        "edge-a",
        "edge-b",
    }
    assert second_assignments["classify"] == "edge-c"
    assert agent.is_complete() is True
    saved = json.loads(profile_path.read_text())
    assert set(saved["pairs"]["detect"]) == {"edge-a", "edge-b"}
    assert set(saved["pairs"]["classify"]) == {"edge-c"}
    assert "cloud" not in saved["pairs"]["detect"]
    assert saved["task_residuals"]["1"]
    assert saved["context"] == FragSpliceLatencyModel.build_profile_context(
        {"fps": 6},
        FakeSystem().runtime_service_nodes(),
        dag(),
    )


@pytest.mark.unit
def test_cold_sampler_executes_warmups_before_retaining_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    agent = FragSpliceColdSampleAgent(
        FakeSystem(revision=37),
        agent_id=37,
        configuration={"fps": 6},
        profile_path=str(tmp_path / "fragsplice.json"),
        warmup_samples=2,
        samples_per_pair=1,
    )
    info = {"source_device": "source", "dag": dag()}

    for task_id in range(5):
        plan = agent.get_schedule_plan(info)
        assignments = {
            name: plan["dag"][name]["service"]["execute_device"]
            for name in ("detect", "classify")
        }
        agent.update_task(FakeTask(task_id, assignments))
        assert agent.is_complete() is False

    plan = agent.get_schedule_plan(info)
    assignments = {
        name: plan["dag"][name]["service"]["execute_device"]
        for name in ("detect", "classify")
    }
    agent.update_task(FakeTask(5, assignments))

    assert agent.is_complete() is True
    profile = agent.get_profile()
    assert profile["cold_progress"]["seen"] == {
        "classify": {"edge-c": 3},
        "detect": {"edge-a": 3, "edge-b": 3},
    }
    assert len(profile["pairs"]["classify"]["edge-c"]["samples"]) == 1
    assert len(profile["pairs"]["detect"]["edge-a"]["samples"]) == 1
    assert len(profile["pairs"]["detect"]["edge-b"]["samples"]) == 1


@pytest.mark.unit
def test_cold_sampler_resumes_completed_fixed_deployment_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    profile_path = tmp_path / "fragsplice.json"
    profile_path.write_text(json.dumps(contextual_profile(
        {},
        FakeSystem().runtime_service_nodes(),
        dag(),
        {
            "detect": {"edge-a": [0.1], "edge-b": [0.2]},
            "classify": {"edge-c": [0.1]},
        },
        cold_progress={
            "warmup_samples": 0,
            "samples_per_pair": 1,
            "seen": {
                "detect": {"edge-a": 1, "edge-b": 1},
                "classify": {"edge-c": 1},
            },
        },
    )), encoding="utf-8")

    agent = FragSpliceColdSampleAgent(
        FakeSystem(),
        agent_id=2,
        profile_path=str(profile_path),
        warmup_samples=0,
        samples_per_pair=1,
    )

    assert agent.is_complete() is True
    assert agent.should_generate({})["generate"] is False


@pytest.mark.unit
def test_cold_sampler_bounds_pending_and_active_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })

    class BusySystem(FakeSystem):
        def get_scheduling_snapshot(self, scope=None):
            return {
                "captured_at": 1.0,
                "runtime_directory_revision": self.revision,
                "deployment": self.runtime_service_nodes(),
                "reservations": [
                    {"root_uuid": "pending-root"},
                    {"root_uuid": "shared-root"},
                ],
                "commitments": [
                    {"root_uuid": "shared-root"},
                    {"root_uuid": "active-root"},
                ],
                "task_barriers": [],
                "resources": {},
                "resource_received_at": {},
                "resource_runtime_revision": {},
            }

    agent = FragSpliceColdSampleAgent(
        BusySystem(revision=18),
        agent_id=18,
        profile_path=str(tmp_path / "fragsplice.json"),
        warmup_samples=0,
        samples_per_pair=1,
        max_inflight_tasks=3,
    )

    decision = agent.should_generate({})

    assert decision == {
        "generate": False,
        "reason": "fragsplice_profile_inflight_limit",
        "inflight_tasks": 3,
        "max_inflight_tasks": 3,
    }


@pytest.mark.unit
def test_main_agent_keeps_cold_profile_immutable_and_updates_online_model(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    profile_path = tmp_path / "fragsplice.json"
    profile_path.write_text(json.dumps(contextual_profile(
        {},
        FakeSystem().runtime_service_nodes(),
        dag(),
        {
            "detect": {"edge-a": [0.1], "edge-b": [0.2]},
            "classify": {"edge-c": [0.1]},
        },
    )), encoding="utf-8")
    agent = FragSpliceAgent(
        FakeSystem(),
        agent_id=3,
        latency_profile=str(profile_path),
        scenario_count=8,
        max_scenarios=8,
    )

    before = profile_path.read_bytes()
    assert agent.update_task(
        FakeTask(1, {"detect": "edge-a", "classify": "edge-c"})
    ) is None
    saved = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile_path.read_bytes() == before
    assert saved["version"] == FragSpliceLatencyModel.PROFILE_VERSION
    assert saved["context"] == FragSpliceLatencyModel.build_profile_context(
        {},
        FakeSystem().runtime_service_nodes(),
        dag(),
    )
    online = agent.latency_model.to_profile(
        deployment=FakeSystem().runtime_service_nodes()
    )
    assert online["task_residuals"]["1"]
    assert "pair_log_drift" in online


@pytest.mark.unit
def test_main_agent_rejects_legacy_and_configuration_mismatched_profiles():
    pairs = {
        "detect": {"edge-a": [0.1], "edge-b": [0.2]},
        "classify": {"edge-c": [0.1]},
    }
    with pytest.raises(ValueError, match="strict context version"):
        FragSpliceAgent(
            FakeSystem(),
            agent_id=11,
            configuration={"fps": 6},
            latency_profile={"version": 2, "pairs": pairs},
        )

    profile = contextual_profile(
        {"fps": 6},
        FakeSystem().runtime_service_nodes(),
        dag(),
        pairs,
    )
    with pytest.raises(ValueError, match="configuration"):
        FragSpliceAgent(
            FakeSystem(),
            agent_id=12,
            configuration={"fps": 4},
            latency_profile=profile,
        )


@pytest.mark.unit
def test_cold_sampler_resets_deployment_mismatch_but_main_rejects_dag_mismatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    pairs = {
        "detect": {"edge-a": [0.1], "edge-b": [0.2]},
        "classify": {"edge-c": [0.1]},
    }
    profile_path = tmp_path / "fragsplice.json"
    profile_path.write_text(json.dumps(contextual_profile(
        {},
        {"detect": ["edge-a"], "classify": ["edge-c"]},
        dag(),
        {
            "detect": {"edge-a": [0.1]},
            "classify": {"edge-c": [0.1]},
        },
    )), encoding="utf-8")
    cold_agent = FragSpliceColdSampleAgent(
        FakeSystem(revision=13),
        agent_id=13,
        profile_path=str(profile_path),
    )
    assert cold_agent.is_complete() is False
    cold_agent.get_schedule_plan({
        "source_device": "source",
        "dag": dag(),
    })
    reset_profile = cold_agent.get_profile()
    assert reset_profile["pairs"] == {}
    assert reset_profile["context"] == FragSpliceLatencyModel.build_profile_context(
        {},
        FakeSystem().runtime_service_nodes(),
        dag(),
    )

    with pytest.raises(ValueError, match="dag"):
        main_agent = FragSpliceAgent(
            FakeSystem(revision=14),
            agent_id=14,
            latency_profile=contextual_profile(
                {},
                FakeSystem().runtime_service_nodes(),
                one_service_dag(),
                pairs,
            ),
            scenario_count=8,
            max_scenarios=8,
        )
        main_agent.get_schedule_plan({
            "source_id": 1,
            "source_device": "source",
            "all_edge_devices": ["edge-a", "edge-b", "edge-c"],
            "dag": dag(),
            "meta_data": {},
        })


@pytest.mark.unit
def test_cold_sampler_resets_incompatible_persisted_configuration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    profile_path = tmp_path / "fragsplice.json"
    profile_path.write_text(json.dumps(contextual_profile(
        {"fps": 6},
        FakeSystem().runtime_service_nodes(),
        dag(),
        {
            "detect": {"edge-a": [0.1], "edge-b": [0.2]},
            "classify": {"edge-c": [0.1]},
        },
    )), encoding="utf-8")

    agent = FragSpliceColdSampleAgent(
        FakeSystem(revision=16),
        agent_id=16,
        configuration={"fps": 4},
        profile_path=str(profile_path),
        warmup_samples=0,
        samples_per_pair=1,
    )
    agent.get_schedule_plan({"source_device": "source", "dag": dag()})

    profile = agent.get_profile()
    assert profile["pairs"] == {}
    assert profile["context"]["configuration"] == {"fps": 4}


@pytest.mark.unit
def test_profile_context_rejects_mismatched_feedback_and_outside_pair(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
    })
    pairs = {
        "detect": {"edge-a": [0.1], "edge-b": [0.2]},
        "classify": {"edge-c": [0.1]},
    }
    profile = contextual_profile(
        {},
        FakeSystem().runtime_service_nodes(),
        dag(),
        pairs,
    )
    agent = FragSpliceAgent(
        FakeSystem(revision=15),
        agent_id=15,
        latency_profile=profile,
        scenario_count=8,
        max_scenarios=8,
    )
    with pytest.raises(ValueError, match="dag"):
        agent.update_task(FakeTask(
            1,
            {"detect": "edge-a", "classify": "edge-c"},
            dag_value=one_service_dag(),
        ))

    model = FragSpliceLatencyModel(profile)
    with pytest.raises(ValueError, match="outside its deployment context"):
        model.record_sample("detect", "edge-z", 0.1)
