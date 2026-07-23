import ast
import copy
from pathlib import Path

import pytest
import yaml

from core.lib.algorithms.schedule_agent.distream_agent import DistreamAgent
from core.lib.algorithms.schedule_agent.dtodrl_agent import DTODRLAgent
from core.lib.algorithms.schedule_agent.fragsplice import FragSpliceLatencyModel
from core.lib.algorithms.schedule_agent.ibdash_agent import IBDASHAgent
from core.lib.common import ClassFactory, ClassType, Context


CONFIGURATION = {
    "resolution": "720p",
    "fps": 8,
    "encoding": "mp4v",
    "buffer_size": 4,
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


def strict_profile(dag, deployment):
    return {
        "version": FragSpliceLatencyModel.PROFILE_VERSION,
        "metric": FragSpliceLatencyModel.PROFILE_METRIC,
        "context": FragSpliceLatencyModel.build_profile_context(
            CONFIGURATION,
            deployment,
            dag,
        ),
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

    def get_scheduling_snapshot(self):
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
def test_dtodrl_train_then_inference_uses_same_full_plan_action_space(
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("torch")
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
def test_baseline_templates_match_fragsplice_context_and_fixed_deployment():
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

    fragsplice, fragsplice_env = load_template("fragsplice.yaml")
    reference_deployment = ast.literal_eval(
        fragsplice_env["SCH_INITIAL_DEPLOYMENT_POLICY_PARAMETERS"]
    )
    reference_configuration = ast.literal_eval(
        fragsplice_env["SCH_AGENT_PARAMETERS"]
    )["configuration"]
    expected = {
        "ibdash.yaml": ("ibdash", None),
        "distream.yaml": ("distream", None),
        "dtodrl-train.yaml": ("dtodrl", "train"),
        "dtodrl.yaml": ("dtodrl", "inference"),
    }
    for filename, (agent_name, mode) in expected.items():
        template, env = load_template(filename)
        parameters = ast.literal_eval(env["SCH_AGENT_PARAMETERS"])
        assert env["SCH_AGENT_NAME"] == agent_name
        assert env["SCH_REDEPLOYMENT_POLICY_NAME"] == "non"
        assert ast.literal_eval(
            env["SCH_INITIAL_DEPLOYMENT_POLICY_PARAMETERS"]
        ) == reference_deployment
        assert parameters["configuration"] == reference_configuration
        assert parameters["latency_profile"] == "fragsplice-profile.json"
        assert template["file-mount"] == fragsplice["file-mount"]
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
