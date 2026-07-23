import ast
import copy
import itertools
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core.lib.common import ClassFactory, ClassType, Context
from core.lib.algorithms.schedule_agent.fragsplice import (
    FragSpliceLatencyModel,
    FragSpliceOptimizer,
)
from core.lib.algorithms.schedule_agent.fragsplice.execution_state import (
    FragSpliceExecutionState,
)
from core.lib.algorithms.schedule_agent.fragsplice_agent import FragSpliceAgent
from core.lib.algorithms.schedule_agent.fragsplice_cold_sample_agent import (
    FragSpliceColdSampleAgent,
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


@pytest.mark.unit
def test_fragsplice_templates_share_fixed_initial_deployment_and_disable_redeployment():
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
    assert cold_agent["max_inflight_tasks"] == 1
    assert cold["file-mount"] == main["file-mount"] == [{
        "pos": "cloud",
        "path": "scheduler/fragsplice/",
    }]


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
def test_branch_and_bound_matches_exhaustive_full_plan_search():
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
            "created_at": 9.5,
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
        "task_context": {"root_uuid": "new", "created_at": 10.0},
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
            "created_at": 9.5,
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
            "task_context": {"root_uuid": "new", "created_at": 10.0},
            "dag": one_service_dag(),
            "meta_data": {"slo_seconds": 10.0},
        },
        snapshot,
        {"detect": ["edge-a", "edge-b"]},
        "cloud",
    )

    assert result["plan"] == {"detect": "edge-b"}


@pytest.mark.unit
def test_optimizer_starts_candidate_slo_after_identity_reservation():
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
            "task_context": {"root_uuid": "new", "created_at": 8.0},
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
        candidate_created_at=0.0,
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
            "created_at": 1.0,
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
        "created_at": 1.0,
        "slo": 20.0,
    }, seed=7)

    # No-queue path: control .3 + transfer .2 + dispatch .4 + processing 1
    # + handoff .1 + end control .6 + end transfer .5 + completion .7.
    assert candidate["candidate_noqueue"] == pytest.approx(3.8)
    # The candidate starts its SLO clock at captured_at, not created_at.
    assert candidate["latency"]["new"] == pytest.approx(3.8)
    # The admitted task uses its exact SLO start: 2 seconds have elapsed.
    assert active["latency"]["old"] == pytest.approx(5.8)


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
            "task_context": {"root_uuid": "new", "created_at": 10.0},
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
            "task_context": {"root_uuid": "new", "created_at": 10.0},
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
            "task_context": {"root_uuid": "new", "created_at": 10.0},
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
            "task_context": {"root_uuid": "new", "created_at": 10.0},
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
            "task_context": {"root_uuid": "new", "created_at": 10.0},
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
            "a": {"edge-a": [1.0]},
            "b": {"edge-b": [1.0]},
            "join": {"edge-j": [1.0]},
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
                "created_at": 9.0,
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

    def get_scheduling_snapshot(self):
        return {
            "captured_at": 1.0,
            "commitments": [],
            "task_barriers": [],
            "resources": {},
        }


def contextual_profile(configuration, deployment, dag_value, pairs, **extra):
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


@pytest.mark.unit
def test_main_agent_current_state_ablation_removes_only_future_commitments(
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
    agent = FragSpliceAgent(
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
        use_future_commitments=False,
    )
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
        def get_scheduling_snapshot(self):
            return {
                "captured_at": 1.0,
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
def test_main_agent_persists_online_feedback_atomically(tmp_path, monkeypatch):
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

    agent.update_task(FakeTask(1, {"detect": "edge-a", "classify": "edge-c"}))
    saved = json.loads(profile_path.read_text(encoding="utf-8"))

    assert saved["version"] == FragSpliceLatencyModel.PROFILE_VERSION
    assert saved["context"] == FragSpliceLatencyModel.build_profile_context(
        {},
        FakeSystem().runtime_service_nodes(),
        dag(),
    )
    assert saved["task_residuals"]["1"]
    assert "pair_log_drift" in saved


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
