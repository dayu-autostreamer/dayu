import ast
import copy
import importlib
import json
from pathlib import Path

import pytest
import yaml

from core.lib.algorithms.schedule_agent.distream_agent import DistreamAgent
from core.lib.algorithms.schedule_agent.dtodrl_agent.hook import DTODRLAgent
from core.lib.algorithms.schedule_agent.ibdash_agent import IBDASHAgent
from core.lib.common import ClassFactory, ClassType, Context
from core.lib.scheduling import SchedulingSnapshotScope


CONFIGURATION = {
    "resolution": "720p",
    "fps": 8,
    "encoding": "mp4v",
    "buffer_size": 4,
}

PROFILE_VERSION = 5
PROFILE_METRIC = "real_execute_time_seconds"


def profile_context(configuration, deployment, dag):
    normalized_configuration = json.loads(json.dumps(
        configuration,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
    ))
    normalized_deployment = {
        str(service): sorted({
            str(device).strip()
            for device in ([devices] if isinstance(devices, str) else devices)
            if str(device).strip()
        })
        for service, devices in sorted(
            deployment.items(), key=lambda item: str(item[0])
        )
    }
    normalized_dag = {}
    for raw_name, raw_node in sorted(
        dag.items(), key=lambda item: str(item[0])
    ):
        name = str(raw_name)
        node = raw_node if isinstance(raw_node, dict) else {}
        service = node.get("service")
        service = service if isinstance(service, dict) else {}
        normalized_dag[name] = {
            "service_name": str(node.get(
                "service_name",
                service.get("service_name", name),
            )),
            "prev_nodes": sorted(
                str(item) for item in node.get("prev_nodes", [])
            ),
            "next_nodes": sorted(
                str(item) for item in node.get("next_nodes", [])
            ),
        }
    return {
        "configuration": normalized_configuration,
        "deployment": normalized_deployment,
        "dag": normalized_dag,
    }


def chain_dag():
    return {
        "_start": {
            "service": {"service_name": "_start", "execute_device": "source"},
            "prev_nodes": [],
            "next_nodes": ["detect"],
        },
        "detect": {
            "service": {"service_name": "detect", "execute_device": ""},
            "prev_nodes": ["_start"],
            "next_nodes": ["classify"],
        },
        "classify": {
            "service": {"service_name": "classify", "execute_device": ""},
            "prev_nodes": ["detect"],
            "next_nodes": ["_end"],
        },
        "_end": {
            "service": {"service_name": "_end", "execute_device": "cloud"},
            "prev_nodes": ["classify"],
            "next_nodes": [],
        },
    }


def strict_profile(dag, deployment, configuration=CONFIGURATION):
    return {
        "version": PROFILE_VERSION,
        "metric": PROFILE_METRIC,
        "context": profile_context(configuration, deployment, dag),
        "pairs": {
            "detect": {"detect-node": {"samples": [5.0]}},
            "classify": {
                "edge-a": {"samples": [1.0]},
                "edge-b": {"samples": [2.0]},
            },
        },
    }


class FakeSystem:
    cloud_device = "cloud"

    def __init__(self, deployment, snapshot):
        self.deployment = copy.deepcopy(deployment)
        self.snapshot = copy.deepcopy(snapshot)

    def runtime_service_nodes(self):
        return copy.deepcopy(self.deployment)

    def get_scheduling_snapshot(
        self,
        scope=SchedulingSnapshotScope.COMMITTED,
    ):
        return copy.deepcopy(self.snapshot)


class FakeCompletedTask:
    def __init__(self, root_uuid, latency, slo):
        self.root_uuid = root_uuid
        self.latency = latency
        self.slo = slo

    def get_root_uuid(self):
        return self.root_uuid

    def get_real_end_to_end_time(self):
        return self.latency

    def get_metadata(self):
        return {"slo_seconds": self.slo}


def configure_context(monkeypatch, tmp_path):
    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.setattr(Context, "parameters", {
        "DEFAULT_MOUNT_PATH": str(tmp_path),
        "DATA_PATH_PREFIX": str(tmp_path),
        "TEMP_PATH": str(temporary),
    })


def system_fixture():
    dag = chain_dag()
    deployment = {
        "detect": ["detect-node"],
        "classify": ["edge-a", "edge-b"],
    }
    snapshot = {
        "captured_at": 10.0,
        "runtime_directory_revision": 7,
        "deployment": deployment,
        "resources": {
            "edge-a": {
                "queue_state": {
                    "classify": {
                        "busy": True,
                        "running_phase": "processing",
                        "phase_elapsed_s": 0.0,
                        "waiting_count": 2,
                    },
                },
            },
        },
        "resource_received_at": {"edge-a": 10.0},
        "resource_runtime_revision": {"edge-a": 7},
        "reservations": [],
        "commitments": [],
        "task_barriers": [],
    }
    return dag, deployment, FakeSystem(deployment, snapshot)


def schedule_info(dag, root_uuid="root-1"):
    return {
        "source_id": 1,
        "source_device": "source",
        "all_edge_devices": ["detect-node", "edge-a", "edge-b"],
        "dag": copy.deepcopy(dag),
        "meta_data": {"slo_seconds": 10.0},
        "task_context": {"root_uuid": root_uuid},
    }


@pytest.mark.unit
def test_baseline_hooks_are_registered():
    assert ClassFactory.get_cls(ClassType.SCH_AGENT, "ibdash") is IBDASHAgent
    assert ClassFactory.get_cls(ClassType.SCH_AGENT, "distream") is DistreamAgent
    assert ClassFactory.get_cls(ClassType.SCH_AGENT, "dtodrl") is DTODRLAgent


@pytest.mark.unit
def test_baseline_agents_do_not_depend_on_fragsplice_implementation():
    root = Path(__file__).resolve().parents[4]
    for relative_path in (
        "ibdash_agent.py",
        "distream_agent.py",
        "dtodrl_agent/hook.py",
    ):
        source = (
            root
            / "dependency"
            / "core"
            / "lib"
            / "algorithms"
            / "schedule_agent"
            / relative_path
        ).read_text(encoding="utf-8")
        assert "fragsplice" not in source.lower()


@pytest.mark.unit
@pytest.mark.parametrize(
    "agent_cls",
    [IBDASHAgent, DistreamAgent, DTODRLAgent],
)
def test_each_baseline_owns_the_frozen_profile_window(
    tmp_path,
    monkeypatch,
    agent_cls,
):
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    profile = strict_profile(dag, deployment)
    profile["pairs"]["detect"]["detect-node"]["samples"] = list(
        range(1, 130)
    )
    # These optional channels were ignored by the former immutable profile
    # adapter when malformed. Baseline-private readers retain that behavior.
    profile["handoff_pairs"] = []
    profile["pair_log_drift"] = []

    agent = agent_cls(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=profile,
        profile_quantile=0.0,
    )

    assert agent.latency_model.estimate("detect", "detect-node", 0.0) == 2.0
    assert agent.latency_model.estimate_handoff(
        "detect", "detect-node", 0.5
    ) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("agent_cls", "method_name"),
    [
        (IBDASHAgent, "IBDASH"),
        (DistreamAgent, "Distream"),
        (DTODRLAgent, "DTODRL"),
    ],
)
def test_each_baseline_rejects_a_non_mapping_profile_pair_store(
    tmp_path,
    monkeypatch,
    agent_cls,
    method_name,
):
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    profile = strict_profile(dag, deployment)
    profile["pairs"] = []

    with pytest.raises(TypeError, match=f"{method_name} profile pairs"):
        agent_cls(
            system,
            1,
            configuration=CONFIGURATION,
            latency_profile=profile,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("agent_cls", "profile_name"),
    [
        (IBDASHAgent, "ibdash-profile.json"),
        (DistreamAgent, "distream-profile.json"),
        (DTODRLAgent, "dtodrl-profile.json"),
    ],
)
def test_each_baseline_loads_only_its_own_mounted_profile(
    tmp_path,
    monkeypatch,
    agent_cls,
    profile_name,
):
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    (tmp_path / profile_name).write_text(
        json.dumps(strict_profile(dag, deployment)),
        encoding="utf-8",
    )

    agent = agent_cls(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=profile_name,
    )

    assert agent.latency_model.estimate("detect", "detect-node", 0.5) == 5.0


@pytest.mark.unit
def test_profiled_eft_projects_queue_to_future_stage_but_reactive_balancer_does_not(
    tmp_path,
    monkeypatch,
):
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    profile = strict_profile(dag, deployment)
    common = {
        "system": system,
        "agent_id": 1,
        "configuration": CONFIGURATION,
        "latency_profile": profile,
    }
    ibdash = IBDASHAgent(**common)
    distream = DistreamAgent(**common)
    original = schedule_info(dag)

    ibdash_plan = ibdash.get_schedule_plan(original)
    distream_plan = distream.get_schedule_plan(original)

    assert ibdash_plan["dag"]["classify"]["service"]["execute_device"] == "edge-a"
    assert distream_plan["dag"]["classify"]["service"]["execute_device"] == "edge-b"
    assert original["dag"]["classify"]["service"]["execute_device"] == ""
    for policy in (ibdash_plan, distream_plan):
        assert policy["dag"]["detect"]["service"]["execute_device"] == "detect-node"
        assert policy["dag"]["_start"]["service"]["execute_device"] == "source"
        assert policy["dag"]["_end"]["service"]["execute_device"] == "cloud"


@pytest.mark.unit
def test_baselines_ignore_queue_telemetry_from_another_runtime_revision(
    tmp_path,
    monkeypatch,
):
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    system.snapshot["resource_runtime_revision"]["edge-a"] = 6
    profile = strict_profile(dag, deployment)
    agent = DistreamAgent(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=profile,
    )

    policy = agent.get_schedule_plan(schedule_info(dag))

    assert policy["dag"]["classify"]["service"]["execute_device"] == "edge-a"


@pytest.mark.unit
@pytest.mark.parametrize("agent_cls", [IBDASHAgent, DistreamAgent])
def test_reactive_baselines_do_not_build_future_commitment_snapshot(
    tmp_path,
    monkeypatch,
    agent_cls,
):
    configure_context(monkeypatch, tmp_path)
    dag, deployment, base_system = system_fixture()

    class ReactiveSystem(FakeSystem):
        def get_scheduling_snapshot(self, scope=None):
            assert scope is SchedulingSnapshotScope.LIVE
            return copy.deepcopy(self.snapshot)

    system = ReactiveSystem(deployment, base_system.snapshot)
    agent = agent_cls(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=strict_profile(dag, deployment),
    )

    policy = agent.get_schedule_plan(schedule_info(dag))

    assert policy["dag"]["detect"]["service"]["execute_device"] == (
        "detect-node"
    )


@pytest.mark.unit
@pytest.mark.ml
def test_dtodrl_train_then_inference_uses_same_full_plan_action_space(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip(
        "torch",
        reason="DTODRL checkpoint tests require the real PyTorch runtime",
        exc_type=ModuleNotFoundError,
    )
    policy_module = importlib.import_module(
        "core.lib.algorithms.schedule_agent.dtodrl_agent.policy"
    )
    # Thread-pool sizing is process-global and orthogonal to this checkpoint
    # compatibility test.  Some supported PyTorch builds abort, rather than
    # raise RuntimeError, when inter-op threads are configured twice in one
    # pytest process (trainer followed by inference).
    monkeypatch.setattr(policy_module.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(
        policy_module.torch,
        "set_num_interop_threads",
        lambda count: None,
    )
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    profile = strict_profile(dag, deployment)
    checkpoint = tmp_path / "dtodrl.pt"
    trainer = DTODRLAgent(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=profile,
        latency_slo_s=10.0,
        mode="train",
        checkpoint_path=str(checkpoint),
        hidden_dim=16,
        batch_size=1,
        ppo_epochs=1,
        save_interval=1,
        random_seed=7,
    )

    training_policy = trainer.get_schedule_plan(schedule_info(dag, "train-root"))
    trainer.update_task(FakeCompletedTask("train-root", latency=6.0, slo=10.0))

    assert checkpoint.is_file()
    assert trainer.last_training_metrics is not None
    inference = DTODRLAgent(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=profile,
        latency_slo_s=10.0,
        mode="inference",
        checkpoint_path=str(checkpoint),
        hidden_dim=16,
    )
    inference_policy = inference.get_schedule_plan(
        schedule_info(dag, "inference-root")
    )
    for policy in (training_policy, inference_policy):
        assert policy["dag"]["detect"]["service"]["execute_device"] in deployment["detect"]
        assert policy["dag"]["classify"]["service"]["execute_device"] in deployment["classify"]
        assert policy["dag"]["_start"]["service"]["execute_device"] == "source"
        assert policy["dag"]["_end"]["service"]["execute_device"] == "cloud"

    incompatible = DTODRLAgent(
        system,
        1,
        configuration=CONFIGURATION,
        latency_profile=profile,
        latency_slo_s=11.0,
        mode="inference",
        checkpoint_path=str(checkpoint),
        hidden_dim=16,
    )
    with pytest.raises(ValueError, match="active scheduling context"):
        incompatible.get_schedule_plan(schedule_info(dag, "wrong-context"))


@pytest.mark.unit
@pytest.mark.ml
def test_dtodrl_loads_legacy_weights_across_equivalent_video_configuration(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip(
        "torch",
        reason="DTODRL checkpoint tests require the real PyTorch runtime",
        exc_type=ModuleNotFoundError,
    )
    policy_module = importlib.import_module(
        "core.lib.algorithms.schedule_agent.dtodrl_agent.policy"
    )
    monkeypatch.setattr(policy_module.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(
        policy_module.torch,
        "set_num_interop_threads",
        lambda count: None,
    )
    configure_context(monkeypatch, tmp_path)
    dag, deployment, system = system_fixture()
    checkpoint = tmp_path / "legacy-dtodrl.pt"
    legacy_configuration = {
        "resolution": "720p",
        "fps": 16,
        "encoding": "mp4v",
        "buffer_size": 3,
    }
    trainer = DTODRLAgent(
        system,
        1,
        configuration=legacy_configuration,
        latency_profile=strict_profile(
            dag,
            deployment,
            legacy_configuration,
        ),
        latency_slo_s=10.0,
        mode="train",
        checkpoint_path=str(checkpoint),
        hidden_dim=16,
        batch_size=1,
        ppo_epochs=1,
        save_interval=1,
        random_seed=7,
    )
    trainer.get_schedule_plan(schedule_info(dag, "train-root"))
    trainer.update_task(FakeCompletedTask("train-root", latency=6.0, slo=10.0))

    payload = torch.load(checkpoint, map_location="cpu")
    assert "configuration" not in payload["signature"]
    payload["signature"]["configuration"] = copy.deepcopy(
        legacy_configuration
    )
    torch.save(payload, checkpoint)

    current_configuration = {
        "resolution": "720p",
        "fps": 10,
        "encoding": "mp4v",
        "buffer_size": 3,
    }
    inference = DTODRLAgent(
        system,
        1,
        configuration=current_configuration,
        latency_profile=strict_profile(
            dag,
            deployment,
            current_configuration,
        ),
        latency_slo_s=10.0,
        mode="inference",
        checkpoint_path=str(checkpoint),
        hidden_dim=16,
    )

    policy = inference.get_schedule_plan(schedule_info(dag, "inference-root"))

    assert inference.policy.update_count == 1
    assert "configuration" not in inference.policy.signature
    assert policy["dag"]["detect"]["service"]["execute_device"] == (
        "detect-node"
    )


@pytest.mark.unit
def test_baseline_templates_use_independent_artifacts_in_one_fair_context():
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

    reference_deployment = {
        "policy": {
            "traffic-detection": ["edgexn23", "edgexn27", "edgexn33"],
            "road-context-segmentation": [
                "edgexn24", "edgexn28", "edgexn33",
            ],
            "traffic-signal-recognition": [
                "edgexn31", "edgexn32", "edgexn34",
            ],
            "vehicle-tracking": [
                "edgexn23", "edgexn24", "edgexn27", "edgexn34",
            ],
            "vehicle-attribute-recognition": [
                "edgexn26", "edgexn28", "edgexn31", "edgexn32",
                "edgexn34",
            ],
            "vehicle-trajectory-prediction": ["edgexn27", "edgexn32"],
            "pedestrian-pose-estimation": ["edgexn24", "edgexn28"],
            "pedestrian-intent-recognition": ["edgexn28", "edgexn32"],
            "risk-graph-generation": ["edgexn23"],
        },
    }
    reference_configuration = {
        "resolution": "720p",
        "fps": 10,
        "encoding": "mp4v",
        "buffer_size": 3,
    }
    expected = {
        "ibdash.yaml": (
            "ibdash", None, "ibdash-profile.json", "scheduler/ibdash/",
        ),
        "distream.yaml": (
            "distream", None, "distream-profile.json", "scheduler/distream/",
        ),
        "dtodrl-train.yaml": (
            "dtodrl", "train", "dtodrl-profile.json", "scheduler/dtodrl/",
        ),
        "dtodrl.yaml": (
            "dtodrl", "inference", "dtodrl-profile.json", "scheduler/dtodrl/",
        ),
    }
    for filename, (agent_name, mode, profile_name, mount_path) in expected.items():
        template, env = load_template(filename)
        template_source = (
            root / "template" / "scheduler" / filename
        ).read_text(encoding="utf-8")
        assert "fragsplice" not in template_source.lower()
        parameters = ast.literal_eval(env["SCH_AGENT_PARAMETERS"])
        assert env["SCH_AGENT_NAME"] == agent_name
        assert env["SCH_REDEPLOYMENT_POLICY_NAME"] == "non"
        assert ast.literal_eval(
            env["SCH_INITIAL_DEPLOYMENT_POLICY_PARAMETERS"]
        ) == reference_deployment
        assert parameters["configuration"] == reference_configuration
        assert parameters["latency_profile"] == profile_name
        assert template["file-mount"] == [{
            "pos": "cloud",
            "path": mount_path,
        }]
        if mode is not None:
            assert parameters["mode"] == mode

    policies = yaml.safe_load(
        (root / "template" / "scheduler_policies.yaml").read_text(
            encoding="utf-8"
        )
    )
    policy_files = {item["id"]: item["yaml"] for item in policies}
    assert policy_files["ibdash"] == "ibdash.yaml"
    assert policy_files["distream"] == "distream.yaml"
    assert policy_files["dtodrl-train"] == "dtodrl-train.yaml"
    assert policy_files["dtodrl"] == "dtodrl.yaml"
    policy_records = {item["id"]: item for item in policies}
    for policy_id in ("ibdash", "distream", "dtodrl-train", "dtodrl"):
        dependency = policy_records[policy_id]["dependency"]
        assert dependency["generator"] == (
            "generator-with-bursty.yaml"
        )
        assert dependency["monitor"] == (
            "monitor-queue-state.yaml"
        )
    generator = yaml.safe_load(
        (
            root
            / "template"
            / "generator"
            / "generator-with-bursty.yaml"
        ).read_text(encoding="utf-8")
    )
    generator_env = {
        item["name"]: item["value"]
        for item in generator["pod-template"]["env"]
    }
    assert generator_env["GEN_GETTER_FILTER_NAME"] == "simple"
    assert "GEN_GETTER_FILTER_PARAMETERS" not in generator_env
    assert generator_env["REQUEST_SCHEDULING_INTERVAL"] == "0"
    assert ast.literal_eval(
        generator_env["HTTP_VIDEO_TASK_ARRIVAL_BURST"]
    ) == {
        "tasks_per_burst": 8,
        "intra_burst_rate_multiplier": 3.0,
    }
    assert generator_env["ASYNC_TASK_SUBMISSION"] == "true"
    assert generator_env["TASK_SUBMISSION_WORKERS"] == "8"
    assert generator_env["TASK_SUBMISSION_QUEUE_DEPTH"] == "16"
    monitor = yaml.safe_load(
        (
            root / "template" / "monitor" / "monitor-queue-state.yaml"
        ).read_text(encoding="utf-8")
    )
    monitor_env = {
        item["name"]: item["value"]
        for item in monitor["pod-template"]["env"]
    }
    assert monitor_env == {"INTERVAL": "0.5", "MONITORS": "['queue_state']\n"}
    assert not (
        root / "template" / "generator" / "generator-fragsplice.yaml"
    ).exists()
    assert not (
        root / "template" / "monitor" / "monitor-fragsplice.yaml"
    ).exists()


@pytest.mark.unit
def test_no_distribution_profiler_template_uses_uninformed_random_inputs():
    root = Path(__file__).resolve().parents[4]
    data = yaml.safe_load(
        (
            root
            / "template"
            / "scheduler"
            / "fragsplice-no-distribution-profiler.yaml"
        ).read_text(encoding="utf-8")
    )
    env = {
        item["name"]: item["value"]
        for item in data["pod-template"]["env"]
    }
    parameters = ast.literal_eval(env["SCH_AGENT_PARAMETERS"])

    assert env["SCH_AGENT_NAME"] == "fragsplice_no_distribution_profiler"
    assert "latency_profile" not in parameters
    assert parameters["random_invocation_cost_min_s"] == 0.0
    assert parameters["random_invocation_cost_max_s"] == 2.0
    assert parameters["random_overhead_cost_min_s"] == 0.0
    assert parameters["random_overhead_cost_max_s"] == 0.2
    assert parameters["random_workload_token_min"] == 0
    assert parameters["random_workload_token_max"] == 8
