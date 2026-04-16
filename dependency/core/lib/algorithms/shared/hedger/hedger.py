import copy
from typing import Dict, List, Optional
import threading
import random
import math
from dataclasses import dataclass
import torch
import time
import os
import glob

from core.lib.common import LOGGER, FileOps, Context, Recorder

from .topology_encoder import TopologyEncoders
from .ppo_agent import HedgerOffloadingPPO, HedgerDeploymentPPO
from .hedger_config import from_partial_dict, OffloadingConstraintCfg, DeploymentConstraintCfg, LogicalTopology, \
    PhysicalTopology
from .state_buffer import StateBuffer, BufferWaitCfg

__all__ = ('Hedger',)

TRAINING_STAGE_NAMES = {"offloading_warmup", "deployment_adaptation", "joint_finetune"}


@dataclass(frozen=True)
class HedgerStateCfg:
    max_buffer_size: int
    offloading_seq_len: int
    deployment_seq_len: int
    min_dynamic_len: int
    wait_timeout_s: Optional[float]
    require_full_seq: bool
    latency_slo: Optional[float]
    deployment_reward_window: int


@dataclass(frozen=True)
class HedgerEncoderCfg:
    embedding_dim: int
    logical_heads: int
    physical_role_count: int
    physical_role_embedding_dim: int
    dropout: float


@dataclass(frozen=True)
class HedgerTrainingCfg:
    stage: str
    total_updates: int
    ppo_epochs: int
    deployment_rollout_len: int
    offloading_rollout_len: int
    deployment_batch_size: int
    offloading_batch_size: int


@dataclass(frozen=True)
class HedgerCheckpointLoadCfg:
    enabled: bool = False
    from_stage: Optional[str] = None
    which: str = "latest"
    step: Optional[int] = None
    load_path: Optional[str] = None
    restore_encoder: bool = True
    restore_deployment_agent: bool = True
    restore_offloading_agent: bool = True
    restore_optimizer: bool = True
    reset_stage_counters: bool = False


@dataclass(frozen=True)
class HedgerCheckpointSaveCfg:
    interval_updates: int
    save_latest: bool = True
    save_final: bool = True
    save_history: bool = True
    keep_last_snapshots: Optional[int] = None


@dataclass(frozen=True)
class HedgerCheckpointCfg:
    root_dir: str
    load: HedgerCheckpointLoadCfg
    save: HedgerCheckpointSaveCfg


@dataclass(frozen=True)
class HedgerTrainingStageCfg:
    name: str
    run_deployment_worker: bool
    update_deployment_policy: bool
    run_offloading_worker: bool
    update_offloading_policy: bool
    use_frozen_offloading_rollout: bool = False


class Hedger:
    def __init__(self, config: dict):
        config = copy.deepcopy(config or {})

        self.mode = self._require_choice(config, "mode", {"train", "inference"})
        self.device = torch.device(self._require_str(config, "device"))
        self.seed = int(config.get("seed", 0))

        timing_cfg = self._require_mapping(config, "timing")
        self.deployment_interval = float(timing_cfg["deployment_interval_s"])
        self.offloading_interval = float(timing_cfg["offloading_interval_s"])

        self.encoder_cfg = self._build_encoder_cfg(config)
        self.state_cfg = self._build_state_cfg(config)
        self.training_cfg = self._build_training_cfg(config) if self.mode == "train" else None
        self.stage_cfg = self._build_training_stage_cfg(self.training_cfg.stage) if self.training_cfg is not None else None
        self.checkpoint_cfg = self._build_checkpoint_cfg(config)

        agents_cfg = self._require_mapping(config, "agents")
        self.deployment_agent_params = self._build_deployment_agent_params(agents_cfg)
        self.offloading_agent_params = self._build_offloading_agent_params(agents_cfg)

        self.deployment_thread_stop_event = threading.Event()
        self.offloading_thread_stop_event = threading.Event()

        self.physical_topology = None
        self.logical_topology = None

        self.shared_topology_encoder = None
        self.deployment_agent = None
        self.offloading_agent = None

        self.state_buffer = None

        self._deployment_update_steps = 0
        self._offloading_update_steps = 0
        # Stage-local PPO update counter.
        self._epoch = 0
        # Cross-stage PPO update counter preserved across stage transitions.
        self._global_update_step = 0
        self._loaded_checkpoint_path = None

        self.register_topology_encoder()
        self.register_deployment_agent()
        self.register_offloading_agent()

        self._data_lock = threading.Lock()
        self._run_lock = threading.Lock()
        self._model_lock = threading.RLock()
        self._run_thread = None
        self._run_started = False

        FileOps.create_directory(self.checkpoint_cfg.root_dir)
        if self.checkpoint_cfg.load.enabled:
            self.load_checkpoint()

        self.initial_deployment_plan = None
        self.deployment_plan = None
        self.offloading_plan = None

        self.deployment_transitions: List[dict] = []
        self.offloading_transitions: List[dict] = []

        self.dep_recorder = None
        self.off_recorder = None
        self.dep_update_recorder = None
        self.off_update_recorder = None

        self.cur_deploy_mask = None
        self._frozen_offloading_agent = None

        if self.mode == "inference":
            self.ensure_started("inference initialization")
        else:
            LOGGER.info("[Hedger][Lifecycle] Training start is deferred until the first task update arrives.")

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def ensure_started(self, reason: str = "manual") -> bool:
        with self._run_lock:
            if self._run_started:
                return False
            self._run_started = True
            self._run_thread = threading.Thread(target=self.run, daemon=True)
            self._run_thread.start()
        LOGGER.info(f"[Hedger][Lifecycle] Background worker started: mode={self.mode}, reason={reason}")
        return True

    @staticmethod
    def _require_mapping(config: dict, key: str) -> dict:
        value = config.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Hedger config requires mapping section `{key}`.")
        return value

    @staticmethod
    def _require_str(config: dict, key: str) -> str:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Hedger config requires non-empty string `{key}`.")
        return value.strip()

    @staticmethod
    def _require_choice(config: dict, key: str, choices: set) -> str:
        value = Hedger._require_str(config, key)
        if value not in choices:
            raise ValueError(f"Hedger config `{key}` must be one of {sorted(choices)}, got {value!r}.")
        return value

    def _build_encoder_cfg(self, config: dict) -> HedgerEncoderCfg:
        encoder = self._require_mapping(config, "encoder")
        return HedgerEncoderCfg(
            embedding_dim=max(1, int(encoder["embedding_dim"])),
            logical_heads=max(1, int(encoder["logical_heads"])),
            physical_role_count=max(1, int(encoder["physical_role_count"])),
            physical_role_embedding_dim=max(1, int(encoder["physical_role_embedding_dim"])),
            dropout=float(encoder.get("dropout", 0.0)),
        )

    def _build_training_cfg(self, config: dict) -> HedgerTrainingCfg:
        training = self._require_mapping(config, "training")
        rollout = self._require_mapping(training, "rollout")
        batch_size = self._require_mapping(training, "batch_size")
        stage = self._require_str(training, "stage")
        if stage not in TRAINING_STAGE_NAMES:
            raise ValueError(
                "Hedger training stage must be one of: "
                "'offloading_warmup', 'deployment_adaptation', 'joint_finetune'."
            )
        return HedgerTrainingCfg(
            stage=stage,
            total_updates=max(0, int(training["total_updates"])),
            ppo_epochs=max(1, int(training["ppo_epochs"])),
            deployment_rollout_len=max(1, int(rollout["deployment"])),
            offloading_rollout_len=max(1, int(rollout["offloading"])),
            deployment_batch_size=max(1, int(batch_size["deployment"])),
            offloading_batch_size=max(1, int(batch_size["offloading"])),
        )

    def _build_state_cfg(self, config: dict) -> HedgerStateCfg:
        state = self._require_mapping(config, "state")
        sequence_length = self._require_mapping(state, "sequence_length")
        reward_window_default = max(1, int(math.ceil(self.deployment_interval / max(self.offloading_interval, 1e-6))))
        wait_timeout_s = state.get("wait_timeout_s", 1.0)
        wait_timeout_s = None if wait_timeout_s is None else max(0.0, float(wait_timeout_s))
        return HedgerStateCfg(
            max_buffer_size=max(1, int(state["max_buffer_size"])),
            offloading_seq_len=max(1, int(sequence_length["offloading"])),
            deployment_seq_len=max(1, int(sequence_length["deployment"])),
            min_dynamic_len=max(0, int(state.get("min_dynamic_length", 1))),
            wait_timeout_s=wait_timeout_s,
            require_full_seq=bool(state.get("require_full_sequence", False)),
            latency_slo=float(state["latency_slo_s"]),
            deployment_reward_window=max(1, int(state.get("deployment_reward_window", reward_window_default))),
        )

    def _build_checkpoint_cfg(self, config: dict) -> HedgerCheckpointCfg:
        checkpoint = self._require_mapping(config, "checkpoint")
        load_cfg = checkpoint.get("load") or {}
        save_cfg = checkpoint.get("save") or {}
        if not isinstance(load_cfg, dict):
            raise ValueError("Hedger checkpoint.load must be a mapping when provided.")
        if not isinstance(save_cfg, dict):
            raise ValueError("Hedger checkpoint.save must be a mapping when provided.")

        which = str(load_cfg.get("which", "latest")).strip().lower()
        if which not in {"latest", "final", "step", "path"}:
            raise ValueError("Hedger checkpoint.load.which must be one of: latest, final, step, path.")
        if which == "step" and load_cfg.get("step") is None:
            raise ValueError("Hedger checkpoint.load.which='step' requires checkpoint.load.step.")
        if which == "path" and not load_cfg.get("path"):
            raise ValueError("Hedger checkpoint.load.which='path' requires checkpoint.load.path.")

        restore_cfg = load_cfg.get("restore") or {}
        if not isinstance(restore_cfg, dict):
            raise ValueError("Hedger checkpoint.load.restore must be a mapping when provided.")

        from_stage = load_cfg.get("from_stage")
        if from_stage is not None:
            from_stage = str(from_stage).strip()
            if from_stage not in TRAINING_STAGE_NAMES:
                raise ValueError(
                    "Hedger checkpoint.load.from_stage must be one of: "
                    "'offloading_warmup', 'deployment_adaptation', 'joint_finetune'."
                )
        if self.mode == "inference" and bool(load_cfg.get("enabled", False)) and which != "path" and from_stage is None:
            raise ValueError(
                "Hedger inference mode requires checkpoint.load.from_stage unless checkpoint.load.which='path'."
            )

        keep_last_snapshots = save_cfg.get("keep_last")
        if keep_last_snapshots is not None:
            keep_last_snapshots = max(1, int(keep_last_snapshots))

        return HedgerCheckpointCfg(
            root_dir=self._resolve_path(checkpoint["root_dir"]),
            load=HedgerCheckpointLoadCfg(
                enabled=bool(load_cfg.get("enabled", False)),
                from_stage=from_stage,
                which=which,
                step=None if load_cfg.get("step") is None else int(load_cfg["step"]),
                load_path=load_cfg.get("path"),
                restore_encoder=bool(restore_cfg.get("encoder", True)),
                restore_deployment_agent=bool(restore_cfg.get("deployment_agent", True)),
                restore_offloading_agent=bool(restore_cfg.get("offloading_agent", True)),
                restore_optimizer=bool(restore_cfg.get("optimizer", True)),
                reset_stage_counters=bool(load_cfg.get("reset_stage_counters", False)),
            ),
            save=HedgerCheckpointSaveCfg(
                interval_updates=max(1, int(save_cfg.get("interval_updates", 20))),
                save_latest=bool(save_cfg.get("latest", True)),
                save_final=bool(save_cfg.get("final", True)),
                save_history=bool(save_cfg.get("history", True)),
                keep_last_snapshots=keep_last_snapshots,
            ),
        )

    @staticmethod
    def _resolve_path(path_ref: str) -> str:
        if not isinstance(path_ref, str) or not path_ref.strip():
            raise ValueError("Hedger path fields must be non-empty strings.")
        return Context.get_file_path(path_ref)

    def _stage_log_path(self, filename: str) -> str:
        if self.training_cfg is None:
            raise ValueError("Stage log path is only available in train mode.")
        return os.path.join(self._checkpoint_stage_dir(self.training_cfg.stage), filename)

    @staticmethod
    def _ppo_update_fieldnames() -> List[str]:
        return [
            "agent", "update", "epoch", "used", "remaining",
            "samples", "epochs", "batch_size", "minibatches",
            "reward_mean", "reward_std", "reward_min", "reward_max",
            "value_old_mean", "value_old_std", "value_new_mean",
            "return_mean", "return_std", "adv_mean", "adv_std",
            "last_value", "done_fraction",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "clip_fraction", "ratio_mean", "ratio_std",
            "actor_grad_norm", "critic_grad_norm",
        ]

    def _record_ppo_update(
            self,
            recorder: Optional[Recorder],
            agent_name: str,
            update_step: int,
            used: int,
            remaining: int,
            stats: Optional[Dict[str, float]],
    ):
        if recorder is None or stats is None:
            return
        row = {
            "agent": agent_name,
            "update": update_step,
            "epoch": self._epoch,
            "used": used,
            "remaining": remaining,
        }
        row.update(stats)
        recorder.log_dict(row)

    def _build_training_stage_cfg(self, stage_name: str) -> HedgerTrainingStageCfg:
        if stage_name == "offloading_warmup":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=False,
                update_deployment_policy=False,
                run_offloading_worker=True,
                update_offloading_policy=True,
            )
        if stage_name == "deployment_adaptation":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=True,
                update_deployment_policy=True,
                run_offloading_worker=True,
                update_offloading_policy=False,
                use_frozen_offloading_rollout=True,
            )
        if stage_name == "joint_finetune":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=True,
                update_deployment_policy=True,
                run_offloading_worker=True,
                update_offloading_policy=True,
            )

        raise ValueError(
            f"Unsupported training stage {stage_name!r}. "
            f"Expected one of: offloading_warmup, deployment_adaptation, joint_finetune."
        )

    def _build_deployment_agent_params(self, agents_cfg: dict) -> dict:
        deployment = self._require_mapping(agents_cfg, "deployment")
        reward = self._require_mapping(deployment, "reward")
        penalty = self._require_mapping(deployment, "penalty")
        return {
            "actor_lr": float(deployment["actor_lr"]),
            "critic_lr": float(deployment["critic_lr"]),
            "gamma": float(deployment["gamma"]),
            "lamda": float(deployment["lamda"]),
            "clip_eps": float(deployment["clip_eps"]),
            "update_encoder": bool(deployment.get("update_encoder", True)),
            "reward_dep_offload_weight": float(reward["offloading_weight"]),
            "reward_dep_change_weight": float(reward["change_cost_weight"]),
            "penalty_capacity_relax": float(penalty["capacity_relax"]),
        }

    def _build_offloading_agent_params(self, agents_cfg: dict) -> dict:
        offloading = self._require_mapping(agents_cfg, "offloading")
        reward = self._require_mapping(offloading, "reward")
        penalty = self._require_mapping(offloading, "penalty")
        return {
            "actor_lr": float(offloading["actor_lr"]),
            "critic_lr": float(offloading["critic_lr"]),
            "gamma": float(offloading["gamma"]),
            "lamda": float(offloading["lamda"]),
            "clip_eps": float(offloading["clip_eps"]),
            "update_encoder": bool(offloading.get("update_encoder", True)),
            "reward_off_latency_weight": float(reward["latency_weight"]),
            "reward_off_slo_weight": float(reward["slo_weight"]),
            "reward_off_cloud_weight": float(reward["cloud_weight"]),
            "penalty_switch": float(penalty["switch"]),
            "penalty_relax": float(penalty["correction"]),
        }

    def _sync_agent_topology_bindings(self):
        """Synchronize agent-side source/cloud indices with the registered physical topology."""
        if self.physical_topology is None:
            return
        if self.deployment_agent is not None:
            self.deployment_agent.cloud_idx = self.physical_topology.cloud_idx
        if self.offloading_agent is not None:
            self.offloading_agent.source = self.physical_topology.source_idx
            self.offloading_agent.cloud_idx = self.physical_topology.cloud_idx

    @staticmethod
    def _format_log_value(value, digits: int = 4) -> str:
        if value is None:
            return "none"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, int):
            return str(value)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isfinite(value_float):
            return f"{value_float:.{digits}f}"
        return str(value_float)

    @classmethod
    def _format_utilization_for_log(cls, value, digits: int = 4) -> str:
        ratio_text = cls._format_log_value(value, digits)
        try:
            ratio_float = float(value)
        except (TypeError, ValueError):
            return ratio_text
        if not math.isfinite(ratio_float):
            return ratio_text
        return f"{ratio_text} ({ratio_float * 100.0:.2f}%)"

    def _summarize_runtime_config(self) -> str:
        training_cfg = getattr(self, "training_cfg", None)
        checkpoint_cfg = getattr(self, "checkpoint_cfg", None)
        checkpoint_save_cfg = getattr(checkpoint_cfg, "save", None)
        stage_name = training_cfg.stage if training_cfg is not None else "none"
        state_cfg = getattr(self, "state_cfg", None)
        dep_seq_len = getattr(state_cfg, "deployment_seq_len", "na")
        off_seq_len = getattr(state_cfg, "offloading_seq_len", "na")
        return (
            f"mode={getattr(self, 'mode', 'unknown')}, train_stage={stage_name}, "
            f"device={getattr(self, 'device', 'unknown')}, seed={getattr(self, 'seed', 'na')}, "
            f"intervals(dep/off)={self._format_log_value(getattr(self, 'deployment_interval', None), 2)}/"
            f"{self._format_log_value(getattr(self, 'offloading_interval', None), 2)}s, "
            f"stage_step={getattr(self, '_epoch', 'na')}, "
            f"global_step={getattr(self, '_global_update_step', 'na')}, "
            f"rollout(dep/off)={getattr(training_cfg, 'deployment_rollout_len', 'na')}/"
            f"{getattr(training_cfg, 'offloading_rollout_len', 'na')}, "
            f"batch(dep/off)={getattr(training_cfg, 'deployment_batch_size', 'na')}/"
            f"{getattr(training_cfg, 'offloading_batch_size', 'na')}, "
            f"state_seq(dep/off)={dep_seq_len}/{off_seq_len}, "
            f"total_steps={getattr(training_cfg, 'total_updates', 'na')}, "
            f"save_interval={getattr(checkpoint_save_cfg, 'interval_updates', 'na')}"
        )

    def _summarize_topology(self) -> str:
        logical_services = len(self.logical_topology) if self.logical_topology is not None else 0
        logical_edges = len(self.logical_topology.links) if self.logical_topology is not None else 0
        physical_nodes = len(self.physical_topology) if self.physical_topology is not None else 0
        physical_edges = len(self.physical_topology.links) if self.physical_topology is not None else 0
        source_name = (
            self.physical_topology[self.physical_topology.source_idx]
            if self.physical_topology is not None and physical_nodes > 0
            else "unregistered"
        )
        cloud_name = (
            self.physical_topology[self.physical_topology.cloud_idx]
            if self.physical_topology is not None and physical_nodes > 0
            else "unregistered"
        )
        return (
            f"logical_services={logical_services}, logical_edges={logical_edges}, "
            f"physical_nodes={physical_nodes}, physical_edges={physical_edges}, "
            f"source={source_name}, cloud={cloud_name}"
        )

    def _summarize_metrics(self, metrics: Optional[Dict[str, float]]) -> str:
        if not metrics:
            return "none"
        return ", ".join(
            f"{key}={self._format_log_value(value)}"
            for key, value in sorted(metrics.items())
        )

    def _summarize_deploy_mask(self, deploy_mask: Optional[torch.Tensor]) -> str:
        if deploy_mask is None:
            return "services=0, edge_replicas=0, cloud_replicas=0, cloud_only=0"

        mask = deploy_mask.detach().clone().cpu().bool()
        if mask.numel() == 0 or self.physical_topology is None:
            return "services=0, edge_replicas=0, cloud_replicas=0, cloud_only=0"

        cloud_idx = self.physical_topology.cloud_idx
        edge_mask = mask[:, :cloud_idx]
        edge_replicas = int(edge_mask.sum().item()) if edge_mask.numel() else 0
        cloud_replicas = int(mask[:, cloud_idx].sum().item())
        cloud_only = int((~edge_mask.any(dim=1)).sum().item()) if edge_mask.numel() else int(mask.size(0))
        return (
            f"services={mask.size(0)}, edge_replicas={edge_replicas}, "
            f"cloud_replicas={cloud_replicas}, cloud_only={cloud_only}"
        )

    def _summarize_deployment_plan(self, deployment_plan: Optional[dict], sample_size: int = 3) -> str:
        if not deployment_plan:
            return "services=0, replicas=0, edge_replicas=0, sample=[]"

        cloud_name = None
        if self.physical_topology is not None and len(self.physical_topology) > 0:
            cloud_name = self.physical_topology[self.physical_topology.cloud_idx]

        total_replicas = 0
        edge_replicas = 0
        sample_parts = []
        for idx, (service_name, device_names) in enumerate(deployment_plan.items()):
            if isinstance(device_names, (list, tuple, set)):
                devices = list(device_names)
            else:
                devices = [device_names]
            total_replicas += len(devices)
            if cloud_name is not None:
                edge_replicas += sum(1 for device_name in devices if device_name != cloud_name)
            if idx < sample_size:
                sample_parts.append(f"{service_name}->[{','.join(map(str, devices))}]")
        sample_text = "; ".join(sample_parts) if sample_parts else "[]"
        return (
            f"services={len(deployment_plan)}, replicas={total_replicas}, "
            f"edge_replicas={edge_replicas}, sample={sample_text}"
        )

    def _summarize_offloading_plan(self, offloading_plan: Optional[dict], sample_size: int = 3) -> str:
        if not offloading_plan:
            return "services=0, cloud=0, unique_targets=0, sample=[]"

        cloud_name = None
        if self.physical_topology is not None and len(self.physical_topology) > 0:
            cloud_name = self.physical_topology[self.physical_topology.cloud_idx]

        cloud_assignments = 0
        unique_targets = set()
        sample_parts = []
        for idx, (service_name, device_name) in enumerate(offloading_plan.items()):
            if device_name == cloud_name:
                cloud_assignments += 1
            unique_targets.add(device_name)
            if idx < sample_size:
                sample_parts.append(f"{service_name}->{device_name}")
        sample_text = "; ".join(sample_parts) if sample_parts else "[]"
        return (
            f"services={len(offloading_plan)}, cloud={cloud_assignments}, "
            f"unique_targets={len(unique_targets)}, sample={sample_text}"
        )

    def _summarize_state_snapshot(
            self,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            metrics: Optional[Dict[str, float]] = None,
    ) -> str:
        service_count = int(logic_feats["model_flops"].numel()) if "model_flops" in logic_feats else 0
        device_count = int(phys_feats["gpu_flops"].numel()) if "gpu_flops" in phys_feats else 0

        latest_complexity = 0.0
        if "task_complexity_seq" in logic_feats and logic_feats["task_complexity_seq"].numel():
            latest_complexity = float(logic_feats["task_complexity_seq"][:, -1].mean().item())

        latest_latency = 0.0
        if "hist_latency_seq" in logic_feats and logic_feats["hist_latency_seq"].numel():
            latest_latency = float(logic_feats["hist_latency_seq"][:, -1].mean().item())

        latest_gpu_util = 0.0
        if "gpu_util_seq" in phys_feats and phys_feats["gpu_util_seq"].numel():
            latest_gpu_util = float(phys_feats["gpu_util_seq"][:, -1].mean().item())

        latest_mem_util = 0.0
        if "mem_util_seq" in phys_feats and phys_feats["mem_util_seq"].numel():
            latest_mem_util = float(phys_feats["mem_util_seq"][:, -1].mean().item())

        latest_cloud_bw = 0.0
        latest_edge_bw = 0.0
        if "bandwidth_seq" in phys_feats and phys_feats["bandwidth_seq"].numel():
            bandwidth_seq = phys_feats["bandwidth_seq"]
            cloud_idx = self.physical_topology.cloud_idx if self.physical_topology is not None else bandwidth_seq.size(0) - 1
            latest_cloud_bw = float(bandwidth_seq[cloud_idx, -1].item())
            edge_bandwidth = bandwidth_seq[:, -1]
            if edge_bandwidth.numel() > 1:
                edge_bandwidth = torch.cat([edge_bandwidth[:cloud_idx], edge_bandwidth[cloud_idx + 1:]])
            else:
                edge_bandwidth = torch.empty(0, dtype=edge_bandwidth.dtype)
            latest_edge_bw = float(edge_bandwidth.mean().item()) if edge_bandwidth.numel() else 0.0

        base = (
            f"services={service_count}, devices={device_count}, "
            f"latest_complexity={self._format_log_value(latest_complexity)}, "
            f"latest_latency={self._format_log_value(latest_latency)}, "
            f"latest_edge_bw={self._format_log_value(latest_edge_bw)}, "
            f"latest_cloud_bw={self._format_log_value(latest_cloud_bw)}, "
            f"latest_gpu_util={self._format_utilization_for_log(latest_gpu_util)}, "
            f"latest_mem_util={self._format_utilization_for_log(latest_mem_util)}"
        )
        if metrics:
            base += f", metrics=({self._summarize_metrics(metrics)})"
        return base

    def register_topology_encoder(self):
        if self.shared_topology_encoder:
            return

        self.shared_topology_encoder = TopologyEncoders(
            d_model=self.encoder_cfg.embedding_dim,
            heads=self.encoder_cfg.logical_heads,
            num_roles=self.encoder_cfg.physical_role_count,
            role_emb_dim=self.encoder_cfg.physical_role_embedding_dim,
            dropout=self.encoder_cfg.dropout,
        ).to(self.device)

    def register_deployment_agent(self):
        if self.deployment_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before deployment agent.'

        if self.deployment_agent_params.get("update_encoder", False):
            LOGGER.warning(
                "[Hedger][Train] Deployment-side encoder updates are disabled. "
                "The shared topology encoder is owned by the offloading side to avoid "
                "dual-optimizer instability on shared parameters."
            )
        self.deployment_agent_params["update_encoder"] = False

        self.deployment_agent = HedgerDeploymentPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_cfg.embedding_dim,
            actor_lr=self.deployment_agent_params['actor_lr'],
            critic_lr=self.deployment_agent_params['critic_lr'],
            gamma=self.deployment_agent_params['gamma'],
            lamda=self.deployment_agent_params['lamda'],
            clip_eps=self.deployment_agent_params['clip_eps'],
            update_encoder=False,
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
            constraint_cfg=from_partial_dict(DeploymentConstraintCfg, self.deployment_agent_params),
        ).to(self.device)
        self._sync_agent_topology_bindings()

    def register_offloading_agent(self):
        if self.offloading_agent:
            return

        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before offloading agent.'

        self.offloading_agent = HedgerOffloadingPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_cfg.embedding_dim,
            actor_lr=self.offloading_agent_params['actor_lr'],
            critic_lr=self.offloading_agent_params['critic_lr'],
            gamma=self.offloading_agent_params['gamma'],
            lamda=self.offloading_agent_params['lamda'],
            clip_eps=self.offloading_agent_params['clip_eps'],
            update_encoder=self.offloading_agent_params['update_encoder'],
            source_node_idx=self.physical_topology.source_idx if self.physical_topology is not None else 0,
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
            constraint_cfg=from_partial_dict(OffloadingConstraintCfg, self.offloading_agent_params),
        ).to(self.device)
        self._sync_agent_topology_bindings()

    def register_physical_topology(self, edge_nodes, source_device):
        if self.physical_topology:
            return

        self.physical_topology = PhysicalTopology(edge_nodes, source_device)
        self._sync_agent_topology_bindings()

        LOGGER.info(f"[Hedger][Topology] Registered physical topology: {self._summarize_topology()}")
        LOGGER.debug(
            f"[Hedger][Topology] Physical nodes={self.physical_topology.nodes}, "
            f"links={self.physical_topology.links}"
        )

    def register_logical_topology(self, dag):
        if self.logical_topology:
            return

        self.logical_topology = LogicalTopology(dag)

        LOGGER.info(f"[Hedger][Topology] Registered logical topology: {self._summarize_topology()}")
        LOGGER.debug(
            f"[Hedger][Topology] Logical services={self.logical_topology.service_list}, "
            f"links={self.logical_topology.links}"
        )

    def register_state_buffer(self):
        if self.state_buffer:
            return

        assert self.logical_topology, "Logical topology must be registered before registering state buffer."
        assert self.physical_topology, "Physical topology must be registered before registering state buffer."

        self.state_buffer = StateBuffer(self.state_cfg.max_buffer_size,
                                        logical_topology=self.logical_topology,
                                        physical_topology=self.physical_topology)
        LOGGER.info(
            f"[Hedger][StateBuffer] Registered state buffer: capacity={self.state_cfg.max_buffer_size}, "
            f"fixed_lan_bandwidth={self.state_buffer.fixed_lan_bandwidth_mbps:.2f}Mbps"
        )

    def register_initial_deployment(self, deployment_plan):
        assert self.logical_topology, "Logical topology must be registered before registering initial deployment."
        assert self.physical_topology, "Physical topology must be registered before registering initial deployment."

        if self.initial_deployment_plan is not None:
            return
        self.initial_deployment_plan = copy.deepcopy(deployment_plan)
        self.deployment_plan = copy.deepcopy(deployment_plan)

        # Cache the current deployment mask.
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(deployment_plan or {})
        LOGGER.info(
            f"[Hedger][Deployment] Registered initial deployment: "
            f"{self._summarize_deployment_plan(self.initial_deployment_plan)}"
        )
        LOGGER.debug(f"[Hedger][Deployment] Initial deployment detail: {self.initial_deployment_plan}")

    def get_offloading_plan(self):
        return copy.deepcopy(self.offloading_plan)

    def get_initial_deployment_plan(self):
        return copy.deepcopy(self.initial_deployment_plan)

    def get_redeployment_plan(self):
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan)
        return copy.deepcopy(self.deployment_plan)

    def _build_state_wait_cfg(self) -> BufferWaitCfg:
        return BufferWaitCfg(
            min_dynamic_len=self.state_cfg.min_dynamic_len,
            require_full_seq=self.state_cfg.require_full_seq,
            timeout_s=self.state_cfg.wait_timeout_s,
        )

    def _current_deploy_mask(self) -> torch.Tensor:
        if self.cur_deploy_mask is not None:
            return self.cur_deploy_mask.detach().clone().cpu()

        deploy_mask = torch.zeros(
            (len(self.logical_topology), len(self.physical_topology)),
            dtype=torch.bool,
        )
        deploy_mask[:, self.physical_topology.cloud_idx] = True
        return deploy_mask

    def _build_edge_index(self, links) -> torch.Tensor:
        """Build a graph edge-index tensor and keep empty graphs well-formed."""
        if not links:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        return torch.tensor(links, dtype=torch.long, device=self.device).t().contiguous()

    def _prepare_training_stage_runtime(self):
        self._frozen_offloading_agent = None
        if not self.stage_cfg.use_frozen_offloading_rollout:
            return

        with self._model_lock:
            self._frozen_offloading_agent = copy.deepcopy(self.offloading_agent).to(self.device)
        self._frozen_offloading_agent.eval()
        for param in self._frozen_offloading_agent.parameters():
            param.requires_grad_(False)

    def _current_offloading_rollout_agent(self):
        return self._frozen_offloading_agent if self._frozen_offloading_agent is not None else self.offloading_agent

    def _sleep_until_next_tick(self, last_tick: float, interval_s: float) -> float:
        """Sleep long enough to preserve the target loop cadence."""
        interval_s = max(0.0, float(interval_s))
        if interval_s <= 0.0:
            return time.time()

        now = time.time()
        sleep_s = interval_s if last_tick <= 0 else max(0.0, interval_s - (now - last_tick))
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        return time.time()

    def _collect_graph_state(self, seq_len: int):
        assert self.state_buffer is not None, "State buffer must be registered before collecting state."
        logic_feats_raw, phys_feats_raw = self.state_buffer.get_state(
            seq_len=seq_len,
            wait_cfg=self._build_state_wait_cfg(),
            pad_mode="edge",
        )

        logic_feats = {
            "model_flops": torch.tensor(logic_feats_raw["model_flops"], dtype=torch.float32),
            "model_mem": torch.tensor(logic_feats_raw["model_mem"], dtype=torch.float32),
            "task_complexity_seq": torch.tensor(logic_feats_raw["task_complexity_seq"], dtype=torch.float32),
            "hist_latency_seq": torch.tensor(logic_feats_raw["hist_latency_seq"], dtype=torch.float32),
        }
        phys_feats = {
            "gpu_flops": torch.tensor(phys_feats_raw["gpu_flops"], dtype=torch.float32),
            "role_id": torch.tensor(phys_feats_raw["role_id"], dtype=torch.long),
            "mem_capacity": torch.tensor(phys_feats_raw["mem_capacity"], dtype=torch.float32),
            "bandwidth_seq": torch.tensor(phys_feats_raw["bandwidth_seq"], dtype=torch.float32),
            "gpu_util_seq": torch.tensor(phys_feats_raw["gpu_util_seq"], dtype=torch.float32),
            "mem_util_seq": torch.tensor(phys_feats_raw["mem_util_seq"], dtype=torch.float32),
        }
        return logic_feats, phys_feats

    def _compute_slo_violation(self, latest_latency: torch.Tensor) -> float:
        if self.state_cfg.latency_slo is None:
            return 0.0
        if latest_latency.numel() == 0:
            return 0.0
        threshold = float(self.state_cfg.latency_slo)
        return float((latest_latency > threshold).float().mean().item())

    def _compute_cloud_fraction(self) -> float:
        num_services = len(self.logical_topology)
        if num_services == 0:
            return 0.0

        if not self.offloading_plan:
            return 1.0

        cloud_name = self.physical_topology[self.physical_topology.cloud_idx]
        cloud_count = sum(1 for service_name in self.logical_topology.service_list
                          if self.offloading_plan.get(service_name) == cloud_name)
        return float(cloud_count / num_services)

    def _compute_deploy_change_cost(self, prev_deploy_mask: Optional[torch.Tensor]) -> float:
        if prev_deploy_mask is None:
            return 0.0

        current_mask = self._current_deploy_mask().bool()
        prev_mask = prev_deploy_mask.detach().clone().cpu().bool()
        if current_mask.shape != prev_mask.shape:
            return 0.0

        cloud_idx = self.physical_topology.cloud_idx
        current_edge = current_mask[:, :cloud_idx]
        prev_edge = prev_mask[:, :cloud_idx]
        return float(torch.logical_xor(current_edge, prev_edge).sum().item())

    def _collect_deployment_state(self, prev_deploy_mask: Optional[torch.Tensor] = None):
        logic_feats, phys_feats = self._collect_graph_state(self.state_cfg.deployment_seq_len)
        reward_stats = self.state_buffer.get_offloading_reward_stats(last_k=self.state_cfg.deployment_reward_window)
        metrics = {
            "avg_offloading_reward": float(reward_stats["mean"]),
            "deploy_change_cost": self._compute_deploy_change_cost(prev_deploy_mask),
        }
        done = False
        return logic_feats, phys_feats, metrics, done

    def _collect_offloading_state(self):
        logic_feats, phys_feats = self._collect_graph_state(self.state_cfg.offloading_seq_len)
        latest_latency = logic_feats["hist_latency_seq"][:, -1] if logic_feats["hist_latency_seq"].numel() else \
            torch.empty(0, dtype=torch.float32)
        metrics = {
            "latency": float(latest_latency.mean().item()) if latest_latency.numel() else 0.0,
            "slo_violation": self._compute_slo_violation(latest_latency),
            "cloud_fraction": self._compute_cloud_fraction(),
        }
        done = False
        return logic_feats, phys_feats, metrics, done

    def inference_hedger(self):
        assert self.logical_topology is not None, "Logical topology must be registered before inference."
        assert self.physical_topology is not None, "Physical topology must be registered before inference."

        LOGGER.info(f"[Hedger][Inference] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        self.set_seed()

        self.shared_topology_encoder.eval()
        self.deployment_agent.eval()
        self.offloading_agent.eval()

        logic_links = self._build_edge_index(self.logical_topology.links)
        phys_links = self._build_edge_index(self.physical_topology.links)
        LOGGER.debug(
            f"[Hedger][Inference] Graph summary: logical_edges={logic_links.size(1)}, "
            f"physical_edges={phys_links.size(1)}"
        )

        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        if self.cur_deploy_mask is None:
            if self.deployment_plan is not None:
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan).detach().cpu()
            elif self.initial_deployment_plan is not None:
                self.deployment_plan = copy.deepcopy(self.initial_deployment_plan)
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.initial_deployment_plan
                ).detach().cpu()
            else:
                LOGGER.warning('[Hedger][Inference] No previous deployment state found, initialize to pure cloud deployment.')
                self.cur_deploy_mask = torch.zeros(
                    (len(self.logical_topology), len(self.physical_topology)),
                    dtype=torch.bool,
                )
                self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)
        elif self.deployment_plan is None:
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self._current_deploy_mask())

        LOGGER.info(
            f"[Hedger][Inference] Initial deployment state: "
            f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
            f"{self._summarize_deployment_plan(self.deployment_plan)}"
        )
        LOGGER.info(
            f"[Hedger][Inference] Initial offloading state: "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

        deployment_thread = threading.Thread(target=self.inference_deployment_agent, daemon=True)
        offloading_thread = threading.Thread(target=self.inference_offloading_agent, daemon=True)
        deployment_thread.start()
        offloading_thread.start()

        while not (self.deployment_thread_stop_event.is_set() or self.offloading_thread_stop_event.is_set()):
            if not deployment_thread.is_alive():
                LOGGER.warning('[Hedger][Inference] Deployment worker stopped unexpectedly.')
                self.offloading_thread_stop_event.set()
                break
            if not offloading_thread.is_alive():
                LOGGER.warning('[Hedger][Inference] Offloading worker stopped unexpectedly.')
                self.deployment_thread_stop_event.set()
                break
            time.sleep(0.5)

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        LOGGER.info(
            f"[Hedger][Inference] Finished: "
            f"{self._summarize_deployment_plan(self.deployment_plan)}, "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

    def inference_deployment_agent(self):
        LOGGER.info(
            f"[Hedger][Inference][Deployment] Worker started: "
            f"interval={self._format_log_value(self.deployment_interval, 2)}s"
        )

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment inference."

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        deployment_time_ticket = 0
        prev_deploy_mask = self._current_deploy_mask()
        logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)

        step = 0
        while not self.deployment_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

                with self._model_lock, torch.inference_mode():
                    deploy_mask, _, _, _, aux = self.deployment_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        topo_order=None,
                        prev_deploy_mask=prev_deploy_mask_dev,
                    )

                self.cur_deploy_mask = deploy_mask.detach().cpu()
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                deployment_time_ticket = self._sleep_until_next_tick(
                    deployment_time_ticket,
                    self.deployment_interval,
                )

                prev_deploy_mask = self._current_deploy_mask()
                logic_feats, phys_feats, metrics, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
                LOGGER.debug(
                    f"[Hedger][Inference][Deployment] step={step}, "
                    f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
                    f"{self._summarize_deployment_plan(self.deployment_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"capacity_relax_cnt={aux['capacity_relax_cnt']}, "
                    f"capacity_relax_cost={self._format_log_value(aux['capacity_relax_cost'])}"
                )
                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Inference][Deployment] Worker loop error: {e}")
                time.sleep(0.5)

    def inference_offloading_agent(self):
        LOGGER.info(
            f"[Hedger][Inference][Offloading] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s"
        )

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading inference."

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _ = self._collect_offloading_state()

        step = 0
        while not self.offloading_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                static_mask = self._current_deploy_mask()
                static_mask_dev = static_mask.to(self.device)

                with self._model_lock, torch.inference_mode():
                    actions, _, _, _, aux = self.offloading_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        static_mask=static_mask_dev,
                        topo_order=None,
                    )

                self.offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                offloading_time_ticket = self._sleep_until_next_tick(
                    offloading_time_ticket,
                    self.offloading_interval,
                )

                logic_feats, phys_feats, metrics, _ = self._collect_offloading_state()
                LOGGER.debug(
                    f"[Hedger][Inference][Offloading] step={step}, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"switches={aux['switches']}, correction_cnt={aux['correction_cnt']}, "
                    f"correction_cost={self._format_log_value(aux['correction_cost'])}, "
                    f"aux_cost={self._format_log_value(aux['aux_cost'])}"
                )
                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Inference][Offloading] Worker loop error: {e}")
                time.sleep(0.5)

    def train_hedger(self):
        assert self.logical_topology is not None, "Logical topology must be registered before training."
        assert self.physical_topology is not None, "Physical topology must be registered before training."

        LOGGER.info(f"[Hedger][Train] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        self.set_seed()
        self.shared_topology_encoder.train()
        self.deployment_agent.train()
        self.offloading_agent.train()
        self._prepare_training_stage_runtime()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        logic_links = self._build_edge_index(self.logical_topology.links)
        phys_links = self._build_edge_index(self.physical_topology.links)

        LOGGER.debug(
            f"[Hedger][Train] Graph summary: logical_edges={logic_links.size(1)}, "
            f"physical_edges={phys_links.size(1)}"
        )

        if self.cur_deploy_mask is None:
            LOGGER.warning('[Hedger][Train] No previous deployment state found, initialize to pure cloud deployment.')
            self.cur_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)),
                                               dtype=torch.bool, device=self.device)
            self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True

        LOGGER.info(
            f"[Hedger][Train] Initial deployment state: {self._summarize_deploy_mask(self.cur_deploy_mask)}"
        )

        if not self.stage_cfg.run_deployment_worker and not self.stage_cfg.run_offloading_worker:
            LOGGER.warning('[Hedger][Train] All training workers are disabled for the selected stage, skip training run.')
            return

        deployment_thread = None
        offloading_thread = None
        if self.stage_cfg.run_deployment_worker:
            deployment_thread = threading.Thread(target=self.train_deployment_agent, daemon=True)
            deployment_thread.start()
        if self.stage_cfg.run_offloading_worker:
            offloading_thread = threading.Thread(target=self.train_offloading_agent, daemon=True)
            offloading_thread.start()

        update_fieldnames = self._ppo_update_fieldnames()
        if self.stage_cfg.update_deployment_policy:
            self.dep_update_recorder = Recorder(
                self._stage_log_path("deployment_ppo_updates.csv"),
                fmt="csv",
                fieldnames=update_fieldnames,
                overwrite=True,
                flush_every=1,
            )
        if self.stage_cfg.update_offloading_policy:
            self.off_update_recorder = Recorder(
                self._stage_log_path("offloading_ppo_updates.csv"),
                fmt="csv",
                fieldnames=update_fieldnames,
                overwrite=True,
                flush_every=1,
            )

        while True:
            try:
                if deployment_thread is not None and not deployment_thread.is_alive():
                    LOGGER.warning('[Hedger][Train] Deployment worker stopped unexpectedly.')
                    self.offloading_thread_stop_event.set()
                    break
                if offloading_thread is not None and not offloading_thread.is_alive():
                    LOGGER.warning('[Hedger][Train] Offloading worker stopped unexpectedly.')
                    self.deployment_thread_stop_event.set()
                    break

                updates_in_tick = 0

                # Run a PPO update for the offloading agent.
                if self.stage_cfg.update_offloading_policy and \
                        len(self.offloading_transitions) >= self.training_cfg.offloading_rollout_len:
                    with self._data_lock:
                        off_transitions = self.offloading_transitions[:self.training_cfg.offloading_rollout_len]
                        del self.offloading_transitions[:self.training_cfg.offloading_rollout_len]
                        off_remaining = len(self.offloading_transitions)

                    with self._model_lock:
                        off_ppo_stats = self.offloading_agent.ppo_update(
                            off_transitions,
                            epochs=self.training_cfg.ppo_epochs,
                            batch_size=self.training_cfg.offloading_batch_size,
                        )
                    self._offloading_update_steps += 1
                    updates_in_tick += 1
                    self._record_ppo_update(
                        self.off_update_recorder,
                        "offloading",
                        self._offloading_update_steps,
                        used=len(off_transitions),
                        remaining=off_remaining,
                        stats=off_ppo_stats,
                    )
                    LOGGER.info(
                        f"[Hedger][Train][Offloading] PPO update={self._offloading_update_steps}, "
                        f"used={len(off_transitions)}, remaining={off_remaining}, "
                        f"reward_mean={self._format_log_value(off_ppo_stats.get('reward_mean', 0.0))}, "
                        f"policy_loss={self._format_log_value(off_ppo_stats.get('policy_loss', 0.0))}, "
                        f"value_loss={self._format_log_value(off_ppo_stats.get('value_loss', 0.0))}, "
                        f"entropy={self._format_log_value(off_ppo_stats.get('entropy', 0.0))}, "
                        f"approx_kl={self._format_log_value(off_ppo_stats.get('approx_kl', 0.0))}, "
                        f"clip_fraction={self._format_log_value(off_ppo_stats.get('clip_fraction', 0.0))}"
                    )

                # Run a PPO update for the deployment agent.
                if self.stage_cfg.update_deployment_policy and \
                        len(self.deployment_transitions) >= self.training_cfg.deployment_rollout_len:
                    with self._data_lock:
                        dep_transitions = self.deployment_transitions[:self.training_cfg.deployment_rollout_len]
                        del self.deployment_transitions[:self.training_cfg.deployment_rollout_len]
                        dep_remaining = len(self.deployment_transitions)

                    with self._model_lock:
                        dep_ppo_stats = self.deployment_agent.ppo_update(
                            dep_transitions,
                            epochs=self.training_cfg.ppo_epochs,
                            batch_size=self.training_cfg.deployment_batch_size,
                        )
                    self._deployment_update_steps += 1
                    updates_in_tick += 1
                    self._record_ppo_update(
                        self.dep_update_recorder,
                        "deployment",
                        self._deployment_update_steps,
                        used=len(dep_transitions),
                        remaining=dep_remaining,
                        stats=dep_ppo_stats,
                    )
                    LOGGER.info(
                        f"[Hedger][Train][Deployment] PPO update={self._deployment_update_steps}, "
                        f"used={len(dep_transitions)}, remaining={dep_remaining}, "
                        f"reward_mean={self._format_log_value(dep_ppo_stats.get('reward_mean', 0.0))}, "
                        f"policy_loss={self._format_log_value(dep_ppo_stats.get('policy_loss', 0.0))}, "
                        f"value_loss={self._format_log_value(dep_ppo_stats.get('value_loss', 0.0))}, "
                        f"entropy={self._format_log_value(dep_ppo_stats.get('entropy', 0.0))}, "
                        f"approx_kl={self._format_log_value(dep_ppo_stats.get('approx_kl', 0.0))}, "
                        f"clip_fraction={self._format_log_value(dep_ppo_stats.get('clip_fraction', 0.0))}"
                    )

                # Save a checkpoint.
                if updates_in_tick > 0:
                    prev_epoch = self._epoch
                    self._epoch += updates_in_tick
                    self._global_update_step += updates_in_tick
                    # Save once whenever `_epoch` crosses a multiple of `save_interval`.
                    save_interval = self.checkpoint_cfg.save.interval_updates
                    if (prev_epoch // save_interval) != (self._epoch // save_interval):
                        try:
                            self.save_checkpoint(stage_step=self._epoch, is_final=False)
                        except Exception as e:
                            LOGGER.exception(
                                f"[Hedger][Train] Failed to save checkpoint at stage_step={self._epoch}: {e}"
                            )

                if self._epoch >= self.training_cfg.total_updates:
                    LOGGER.info(
                        f"[Hedger][Train] Reached training step budget: "
                        f"stage_step={self._epoch}, limit={self.training_cfg.total_updates}, "
                        f"global_step={self._global_update_step}"
                    )
                    break

                time.sleep(0.5)
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train] Coordinator loop error: {e}")
                continue

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        try:
            self.save_checkpoint(stage_step=self._epoch, is_final=True)
        except Exception as e:
            LOGGER.exception(f"[Hedger][Train] Failed to save final checkpoint at stage_step={self._epoch}: {e}")
        if self.dep_update_recorder is not None:
            self.dep_update_recorder.close()
            self.dep_update_recorder = None
        if self.off_update_recorder is not None:
            self.off_update_recorder.close()
            self.off_update_recorder = None
        LOGGER.info(
            f"[Hedger][Train] Finished: stage_step={self._epoch}, global_step={self._global_update_step}, "
            f"dep_updates={self._deployment_update_steps}, off_updates={self._offloading_update_steps}, "
            f"{self._summarize_deployment_plan(self.deployment_plan)}, "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

    def train_deployment_agent(self):
        """
        Sampling loop for the deployment agent:
            - Collect the current topology and system state periodically
            - Sample a new deployment mask via `deployment_agent.policy`
            - Let the environment run for one deployment interval, then collect metrics and reward
            - Push the transition into `self.deployment_transitions` for PPO updates
        """
        if not self.stage_cfg.run_deployment_worker:
            LOGGER.info("[Hedger][Train][Deployment] Worker disabled by training stage, skip startup.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment training."

        LOGGER.info(
            f"[Hedger][Train][Deployment] Worker started: "
            f"interval={self._format_log_value(self.deployment_interval, 2)}s, "
            f"rollout={self.training_cfg.deployment_rollout_len}, "
            f"batch={self.training_cfg.deployment_batch_size}, "
            f"update_policy={self.stage_cfg.update_deployment_policy}"
        )

        self.dep_recorder = Recorder(
            self._stage_log_path("deployment_train.csv"),
            fmt="csv",
            fieldnames=["step", "epoch", "dep_updates", "dep_reward", "avg_off_reward",
                        "dep_change_cost", "cap_relax_cnt", "cap_relax_cost",
                        "policy_logp", "policy_entropy", "value_estimate", "next_value",
                        "raw_edge_replicas", "edge_replicas", "cloud_replicas", "cloud_only",
                        "transition_buffer",
                        "dep_offload_weight", "dep_change_weight", "cap_relax_weight"],
            overwrite=True,
            flush_every=1,
        )

        # Static graph edge indices can be reused across iterations.
        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        step = 0
        deployment_time_ticket = 0

        prev_deploy_mask = copy.deepcopy(self.cur_deploy_mask)
        logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)

        while not self.deployment_thread_stop_event.is_set():
            try:
                # Move features onto the active device.
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

                with self._model_lock, torch.no_grad():
                    # Sample a raw deployment action, then execute the corrected
                    # deployment mask returned by the policy.
                    deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        topo_order=None,  # Derived internally from the logical graph.
                        prev_deploy_mask=prev_deploy_mask_dev
                    )
                deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                self.deployment_plan = deploy_plan
                self.cur_deploy_mask = deploy_mask.detach().cpu()

                deployment_time_ticket = self._sleep_until_next_tick(
                    deployment_time_ticket,
                    self.deployment_interval,
                )

                new_logic_feats, new_phys_feats, metrics, done = self._collect_deployment_state(
                    prev_deploy_mask=prev_deploy_mask,)
                new_logic_feats_dev = {k: v.to(self.device) for k, v in new_logic_feats.items()}
                new_phys_feats_dev = {k: v.to(self.device) for k, v in new_phys_feats.items()}
                with self._model_lock, torch.no_grad():
                    next_value = (
                        0.0 if done else float(
                            self.deployment_agent.estimate_value(
                                logic_edge_index=logic_edge_index,
                                logic_feats=new_logic_feats_dev,
                                phys_edge_index=phys_edge_index,
                                phys_feats=new_phys_feats_dev,
                                prev_deploy_mask=deploy_mask.detach(),
                            ).detach().cpu().item()
                        )
                    )

                # Compute the reward from environment metrics and policy-side auxiliaries.
                reward = self._compute_deployment_reward(metrics, aux)

                # Keep transition payloads on CPU to avoid cross-thread device issues.
                # PPO is trained against the raw sampled action, while the
                # environment executes the corrected deployment mask.
                if self.stage_cfg.update_deployment_policy:
                    tr = {
                        "logic_edge_index": logic_edge_index.cpu(),
                        "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                        "phys_edge_index": phys_edge_index.cpu(),
                        "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                        "deploy_mask": deploy_mask.cpu(),
                        "raw_deploy_mask": aux["raw_deploy_mask"].detach().cpu(),
                        "topo_order": None,  # Recomputed during evaluation if needed.
                        "prev_deploy_mask": prev_deploy_mask.cpu() if prev_deploy_mask is not None else None,
                        "logp": logp.detach().cpu(),
                        "value": value.detach().cpu(),
                        "next_value": float(next_value),
                        "reward": float(reward),
                        "done": bool(done),
                    }

                    with self._data_lock:
                        self.deployment_transitions.append(tr)
                        transition_count = len(self.deployment_transitions)
                else:
                    with self._data_lock:
                        transition_count = len(self.deployment_transitions)

                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                prev_deploy_mask = deploy_mask.detach().cpu()

                cloud_idx = self.physical_topology.cloud_idx
                raw_deploy_mask = aux["raw_deploy_mask"].detach().cpu().bool()
                exec_deploy_mask = deploy_mask.detach().cpu().bool()
                raw_edge_replicas = int(raw_deploy_mask[:, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
                edge_replicas = int(exec_deploy_mask[:, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
                cloud_replicas = int(exec_deploy_mask[:, cloud_idx].sum().item())
                cloud_only = int((~exec_deploy_mask[:, :cloud_idx].any(dim=1)).sum().item()) if cloud_idx > 0 else \
                    int(exec_deploy_mask.size(0))

                self.dep_recorder.log(
                    step=step,
                    epoch=self._epoch,
                    dep_updates=self._deployment_update_steps,
                    dep_reward=reward,
                    avg_off_reward=metrics["avg_offloading_reward"],
                    dep_change_cost=metrics["deploy_change_cost"],
                    cap_relax_cnt=aux["capacity_relax_cnt"],
                    cap_relax_cost=aux["capacity_relax_cost"],
                    policy_logp=float(logp.detach().cpu().item()),
                    policy_entropy=float(ent.detach().cpu().item()),
                    value_estimate=float(value.detach().cpu().item()),
                    next_value=float(next_value),
                    raw_edge_replicas=raw_edge_replicas,
                    edge_replicas=edge_replicas,
                    cloud_replicas=cloud_replicas,
                    cloud_only=cloud_only,
                    transition_buffer=transition_count,
                    dep_offload_weight=self.deployment_agent_params["reward_dep_offload_weight"],
                    dep_change_weight=self.deployment_agent_params["reward_dep_change_weight"],
                    cap_relax_weight=self.deployment_agent_params["penalty_capacity_relax"],
                )
                LOGGER.debug(
                    f"[Hedger][Train][Deployment] step={step}, reward={self._format_log_value(reward)}, "
                    f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
                    f"{self._summarize_deployment_plan(self.deployment_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"capacity_relax_cnt={aux['capacity_relax_cnt']}, "
                    f"capacity_relax_cost={self._format_log_value(aux['capacity_relax_cost'])}, "
                    f"update_policy={self.stage_cfg.update_deployment_policy}, transitions={transition_count}"
                )

                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train][Deployment] Worker loop error: {e}")
                time.sleep(0.5)

        self.dep_recorder.close()
        LOGGER.info("[Hedger][Train][Deployment] Worker stopped.")

    def train_offloading_agent(self):
        """
        Sampling loop for the offloading agent:
            - Collect topology and system state at a shorter cadence
            - Select the execution device for each service via `offloading_agent.policy`
            - Let the environment execute the policy and report latency/SLO/cloud metrics
            - Push the transition into `self.offloading_transitions`
        """
        if not self.stage_cfg.run_offloading_worker:
            LOGGER.info("[Hedger][Train][Offloading] Worker disabled by training stage, skip startup.")
            return

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading training."

        LOGGER.info(
            f"[Hedger][Train][Offloading] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s, "
            f"rollout={self.training_cfg.offloading_rollout_len}, "
            f"batch={self.training_cfg.offloading_batch_size}, "
            f"update_policy={self.stage_cfg.update_offloading_policy}, "
            f"rollout_agent={'frozen' if self._frozen_offloading_agent is not None else 'live'}"
        )

        self.off_recorder = Recorder(
            self._stage_log_path("offloading_train.csv"),
            fmt="csv",
            fieldnames=["step", "epoch", "off_updates", "off_reward", "latency",
                        "slo_violation", "cloud_fraction", "aux_cost", "off_latency_weight",
                        "off_slo_weight", "off_cloud_weight", "switch_cnt", "correction_cnt",
                        "correction_cost", "policy_logp", "policy_entropy", "value_estimate",
                        "next_value", "raw_cloud_fraction", "executed_cloud_fraction",
                        "unique_targets", "feasible_targets_mean", "feasible_targets_min",
                        "feasible_targets_max", "transition_buffer",
                        "switch_weight", "correction_weight"],
            overwrite=True,
            flush_every=10,
        )

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)
        rollout_agent = self._current_offloading_rollout_agent()

        step = 0
        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _ = self._collect_offloading_state()
        while not self.offloading_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                static_mask = self._current_deploy_mask()
                static_mask_dev = static_mask.to(self.device)

                with self._model_lock, torch.no_grad():
                    actions, logp, ent, value, aux = rollout_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        static_mask=static_mask_dev,
                        topo_order=None,
                    )
                    offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                    self.offloading_plan = offloading_plan

                offloading_time_ticket = self._sleep_until_next_tick(
                    offloading_time_ticket,
                    self.offloading_interval,
                )

                new_logic_feats, new_phys_feats, metrics, done = self._collect_offloading_state()
                new_logic_feats_dev = {k: v.to(self.device) for k, v in new_logic_feats.items()}
                new_phys_feats_dev = {k: v.to(self.device) for k, v in new_phys_feats.items()}
                next_static_mask_dev = self._current_deploy_mask().to(self.device)
                with self._model_lock, torch.no_grad():
                    next_value = (
                        0.0 if done else float(
                            self.offloading_agent.estimate_value(
                                logic_edge_index=logic_edge_index,
                                logic_feats=new_logic_feats_dev,
                                phys_edge_index=phys_edge_index,
                                phys_feats=new_phys_feats_dev,
                                static_mask=next_static_mask_dev,
                            ).detach().cpu().item()
                        )
                    )

                reward = self._compute_offloading_reward(metrics, aux)
                if self.state_buffer is not None:
                    self.state_buffer.add_offloading_reward(reward)

                # PPO is trained against the raw sampled action, while the
                # environment executes the corrected offloading plan.
                if self.stage_cfg.update_offloading_policy:
                    tr = {
                        "logic_edge_index": logic_edge_index.cpu(),
                        "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                        "phys_edge_index": phys_edge_index.cpu(),
                        "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                        "actions": aux["raw_actions"].detach().cpu(),
                        "static_mask": static_mask_dev.cpu(),
                        "topo_order": None,
                        "logp": logp.detach().cpu(),
                        "value": value.detach().cpu(),
                        "next_value": float(next_value),
                        "reward": float(reward),
                        "done": bool(done),
                    }

                    with self._data_lock:
                        self.offloading_transitions.append(tr)
                        transition_count = len(self.offloading_transitions)
                else:
                    with self._data_lock:
                        transition_count = len(self.offloading_transitions)

                logic_feats = new_logic_feats
                phys_feats = new_phys_feats

                cloud_idx = self.physical_topology.cloud_idx
                actions_cpu = actions.detach().cpu()
                raw_actions_cpu = aux["raw_actions"].detach().cpu()
                raw_cloud_fraction = float((raw_actions_cpu == cloud_idx).float().mean().item())
                executed_cloud_fraction = float((actions_cpu == cloud_idx).float().mean().item())
                unique_targets = int(actions_cpu.unique().numel())
                feasible_counts = static_mask.detach().cpu().float().sum(dim=1)
                feasible_targets_mean = float(feasible_counts.mean().item()) if feasible_counts.numel() else 0.0
                feasible_targets_min = float(feasible_counts.min().item()) if feasible_counts.numel() else 0.0
                feasible_targets_max = float(feasible_counts.max().item()) if feasible_counts.numel() else 0.0

                self.off_recorder.log(
                    step=step,
                    epoch=self._epoch,
                    off_updates=self._offloading_update_steps,
                    off_reward=reward,
                    latency=metrics["latency"],
                    slo_violation=metrics["slo_violation"],
                    cloud_fraction=metrics["cloud_fraction"],
                    aux_cost=aux["aux_cost"],
                    off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                    off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                    off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                    switch_cnt=aux["switches"],
                    correction_cnt=aux["correction_cnt"],
                    correction_cost=aux["correction_cost"],
                    policy_logp=float(logp.detach().cpu().item()),
                    policy_entropy=float(ent.detach().cpu().item()),
                    value_estimate=float(value.detach().cpu().item()),
                    next_value=float(next_value),
                    raw_cloud_fraction=raw_cloud_fraction,
                    executed_cloud_fraction=executed_cloud_fraction,
                    unique_targets=unique_targets,
                    feasible_targets_mean=feasible_targets_mean,
                    feasible_targets_min=feasible_targets_min,
                    feasible_targets_max=feasible_targets_max,
                    transition_buffer=transition_count,
                    switch_weight=self.offloading_agent_params["penalty_switch"],
                    correction_weight=self.offloading_agent_params["penalty_relax"],
                )
                LOGGER.debug(
                    f"[Hedger][Train][Offloading] step={step}, reward={self._format_log_value(reward)}, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"switches={aux['switches']}, correction_cnt={aux['correction_cnt']}, "
                    f"correction_cost={self._format_log_value(aux['correction_cost'])}, "
                    f"aux_cost={self._format_log_value(aux['aux_cost'])}, "
                    f"update_policy={self.stage_cfg.update_offloading_policy}, transitions={transition_count}"
                )

                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train][Offloading] Worker loop error: {e}")
                time.sleep(0.5)

        self.off_recorder.close()
        LOGGER.info("[Hedger][Train][Offloading] Worker stopped.")

    def _compute_offloading_reward(self, metrics, aux) -> float:
        """
        Compute the offloading-agent reward.

        The reward is defined as negative performance cost plus policy-side
        penalties. `metrics` contains environment indicators, while `aux`
        contains switch/correction diagnostics gathered inside the policy.
        """
        metrics = metrics or {}

        # Extract the core metrics.
        latency = float(metrics["latency"])
        slo_v = float(metrics["slo_violation"])
        cloud_frac = float(metrics["cloud_fraction"])

        # Reward weights from hyper-parameters.
        w_lat = float(self.offloading_agent_params["reward_off_latency_weight"])
        w_slo = float(self.offloading_agent_params["reward_off_slo_weight"])
        w_cloud = float(self.offloading_agent_params["reward_off_cloud_weight"])

        # Base reward: lower latency, fewer SLO violations, and lower cloud usage.
        reward = 0.0
        reward -= w_lat * latency
        reward -= w_slo * slo_v
        reward -= w_cloud * cloud_frac

        # Additional constraint cost: device switches and post-sampling
        # correction severity. `correction_cnt` is still logged for debugging,
        # but the reward uses a continuous correction cost.
        aux_cost = float(aux["aux_cost"])
        reward -= aux_cost

        return reward

    def _compute_deployment_reward(self, metrics, aux) -> float:
        """
        Compute the deployment-agent reward.

        The reward combines:
            - aggregated feedback from offloading (`avg_offloading_reward`)
            - deployment change cost
            - capacity-projection penalty
        """
        metrics = metrics or {}

        avg_off_r = float(metrics["avg_offloading_reward"])
        deploy_change_cost = float(metrics["deploy_change_cost"])

        w_change = float(self.deployment_agent_params["reward_dep_change_weight"])
        w_off = float(self.deployment_agent_params["reward_dep_offload_weight"])

        reward = 0.0
        reward -= w_change * deploy_change_cost

        # Treat the lower-level offloading reward as bottom-up feedback.
        reward += w_off * avg_off_r

        # Penalize projection severity rather than just projection count.
        # `capacity_relax_cnt` is still logged for diagnostics, but the reward
        # uses a continuous correction cost that reflects how strongly the raw
        # policy preferred the removed replicas.
        cap_relax_cost = float(aux.get("capacity_relax_cost", 0.0))
        penalty_capacity_relax = float(self.deployment_agent_params["penalty_capacity_relax"])
        reward -= penalty_capacity_relax * cap_relax_cost

        return reward

    def _checkpoint_stage_dir(self, stage_name: Optional[str] = None) -> str:
        resolved_stage = stage_name if stage_name is not None else (self.training_cfg.stage if self.training_cfg else None)
        if not resolved_stage:
            raise ValueError("Checkpoint stage directory requires an explicit training stage.")
        return os.path.join(self.checkpoint_cfg.root_dir, resolved_stage)

    def _checkpoint_snapshot_dir(self, stage_name: Optional[str] = None) -> str:
        return os.path.join(self._checkpoint_stage_dir(stage_name), "snapshots")

    def _checkpoint_snapshot_path(self, stage_step: int, stage_name: Optional[str] = None) -> str:
        return os.path.join(self._checkpoint_snapshot_dir(stage_name), f"step_{int(stage_step):08d}.pt")

    def _checkpoint_alias_path(self, tag: str, stage_name: Optional[str] = None) -> str:
        if tag not in {"latest", "final"}:
            raise ValueError(f"Unsupported Hedger checkpoint tag {tag!r}.")
        return os.path.join(self._checkpoint_stage_dir(stage_name), f"{tag}.pt")

    def _find_latest_stage_snapshot_path(self, stage_name: Optional[str] = None) -> Optional[str]:
        pattern = os.path.join(self._checkpoint_snapshot_dir(stage_name), "step_*.pt")
        files = glob.glob(pattern)
        if not files:
            return None

        def parse_stage_step(path):
            try:
                return int(os.path.basename(path).replace("step_", "").replace(".pt", ""))
            except Exception:
                return -1

        files = sorted(files, key=parse_stage_step, reverse=True)
        return files[0] if files else None

    def _resolve_checkpoint_source_stage(self, stage_ref: Optional[str]) -> Optional[str]:
        if stage_ref is not None:
            stage_ref = str(stage_ref).strip()
            return stage_ref or None
        if self.training_cfg is not None:
            return self.training_cfg.stage
        return None

    def _resolve_checkpoint_load_path(
        self,
        stage: Optional[str] = None,
        which: Optional[str] = None,
        step: Optional[int] = None,
        path: Optional[str] = None,
    ) -> Optional[str]:
        if path:
            resolved_path = self._resolve_path(path)
            return resolved_path if os.path.exists(resolved_path) else None

        source_stage = self._resolve_checkpoint_source_stage(stage)
        if source_stage is None:
            return None

        if which == "step":
            snapshot_path = self._checkpoint_snapshot_path(int(step), source_stage)
            return snapshot_path if os.path.exists(snapshot_path) else None

        target_tag = (which or "latest").strip().lower()
        alias_path = self._checkpoint_alias_path(target_tag, source_stage)
        if os.path.exists(alias_path):
            return alias_path

        if target_tag == "latest":
            return self._find_latest_stage_snapshot_path(source_stage)
        if target_tag == "final":
            LOGGER.warning(
                f"[Hedger][Checkpoint] Final checkpoint for stage={source_stage} is missing, "
                f"fall back to latest snapshot."
            )
            return self._find_latest_stage_snapshot_path(source_stage)
        return None

    def _prune_old_stage_snapshots(self, stage_name: Optional[str] = None):
        keep_last = self.checkpoint_cfg.save.keep_last_snapshots
        if keep_last is None:
            return

        pattern = os.path.join(self._checkpoint_snapshot_dir(stage_name), "step_*.pt")
        files = glob.glob(pattern)
        if len(files) <= keep_last:
            return

        def parse_stage_step(path):
            try:
                return int(os.path.basename(path).replace("step_", "").replace(".pt", ""))
            except Exception:
                return -1

        files = sorted(files, key=parse_stage_step, reverse=True)
        for old_path in files[keep_last:]:
            if os.path.exists(old_path):
                os.remove(old_path)

    def _build_checkpoint_payload(self, stage_step: int) -> dict:
        return {
            'encoder': self.shared_topology_encoder.state_dict(),
            'deployment_agent': self.deployment_agent.state_dict(),
            'offloading_agent': self.offloading_agent.state_dict(),
            'deployment_actor_opt': self.deployment_agent.actor_opt.state_dict(),
            'deployment_critic_opt': self.deployment_agent.critic_opt.state_dict(),
            'offloading_actor_opt': self.offloading_agent.actor_opt.state_dict(),
            'offloading_critic_opt': self.offloading_agent.critic_opt.state_dict(),
            'meta': {
                'schema_version': 2,
                'time': time.time(),
                'seed': self.seed,
                'mode': self.mode,
                'training_stage': self.training_cfg.stage if self.training_cfg is not None else None,
                'stage_step': int(stage_step),
                'global_step': int(self._global_update_step),
                'deployment_updates': int(self._deployment_update_steps),
                'offloading_updates': int(self._offloading_update_steps),
                'device': str(self.device),
                'source_checkpoint': self._loaded_checkpoint_path,
            }
        }

    def save_checkpoint(self, stage_step: Optional[int] = None, is_final: bool = False):
        """
        Save a stage-aware checkpoint.

        Layout:
            {root_dir}/{stage}/latest.pt
            {root_dir}/{stage}/final.pt
            {root_dir}/{stage}/snapshots/step_{stage_step}.pt
        """
        if stage_step is None:
            stage_step = self._epoch

        stage_dir = self._checkpoint_stage_dir()
        snapshot_dir = self._checkpoint_snapshot_dir()
        FileOps.create_directory(stage_dir)
        FileOps.create_directory(snapshot_dir)

        with self._model_lock:
            ckpt = self._build_checkpoint_payload(stage_step)
        saved_paths = []

        if self.checkpoint_cfg.save.save_history:
            snapshot_path = self._checkpoint_snapshot_path(stage_step)
            torch.save(ckpt, snapshot_path)
            saved_paths.append(snapshot_path)
            self._prune_old_stage_snapshots()

        if self.checkpoint_cfg.save.save_latest:
            latest_path = self._checkpoint_alias_path("latest")
            torch.save(ckpt, latest_path)
            saved_paths.append(latest_path)

        if is_final and self.checkpoint_cfg.save.save_final:
            final_path = self._checkpoint_alias_path("final")
            torch.save(ckpt, final_path)
            saved_paths.append(final_path)

        if saved_paths:
            LOGGER.info(
                f"[Hedger][Checkpoint] Saved stage={self.training_cfg.stage if self.training_cfg is not None else 'none'}, "
                f"stage_step={stage_step}, global_step={self._global_update_step}, "
                f"paths={saved_paths}"
            )
        else:
            LOGGER.warning(
                f"[Hedger][Checkpoint] Save skipped because all checkpoint outputs are disabled: "
                f"stage={self.training_cfg.stage if self.training_cfg is not None else 'none'}, stage_step={stage_step}"
            )

    def load_checkpoint(self):
        """
        Load a stage-aware checkpoint.

        Resolution order:
            1. explicit `checkpoint.load.path`
            2. `{checkpoint.load.from_stage}/snapshots/step_x.pt`
            3. `{checkpoint.load.from_stage}/{which}.pt`
            4. latest snapshot fallback within that stage

        When loading from a different stage, stage-local counters are reset while
        the global update step is preserved to keep lineage across staged training.
        """
        if not self.checkpoint_cfg.load.enabled:
            return

        target_path = self._resolve_checkpoint_load_path(
            stage=self.checkpoint_cfg.load.from_stage,
            which=self.checkpoint_cfg.load.which,
            step=self.checkpoint_cfg.load.step,
            path=self.checkpoint_cfg.load.load_path,
        )

        current_stage = self.training_cfg.stage if self.training_cfg is not None else None
        if not target_path or not os.path.exists(target_path):
            requested_stage = self.checkpoint_cfg.load.from_stage or current_stage
            LOGGER.warning(
                f"[Hedger][Checkpoint] No checkpoint found under root={self.checkpoint_cfg.root_dir}, "
                f"requested_stage={requested_stage}, which={self.checkpoint_cfg.load.which}, "
                f"step={self.checkpoint_cfg.load.step}."
            )
            return

        with self._model_lock:
            ckpt = torch.load(target_path, map_location=self.device)
            meta = ckpt.get('meta', {})
            loaded_stage = meta.get('training_stage')
            same_stage_resume = loaded_stage == current_stage
            self._loaded_checkpoint_path = target_path
            LOGGER.info(
                f"[Hedger][Checkpoint] Loading from {target_path} "
                f"(loaded_stage={loaded_stage}, current_stage={current_stage})"
            )

            if self.checkpoint_cfg.load.restore_encoder:
                enc_state = ckpt.get('encoder')
                if enc_state is not None:
                    self.shared_topology_encoder.load_state_dict(enc_state)
                    LOGGER.info('[Hedger][Checkpoint] Loaded encoder state.')
                else:
                    LOGGER.warning('[Hedger][Checkpoint] Missing encoder state in checkpoint.')
            else:
                LOGGER.info('[Hedger][Checkpoint] Skip encoder loading per config.')

            if self.checkpoint_cfg.load.restore_deployment_agent:
                dep_state = ckpt.get('deployment_agent')
                if dep_state is not None:
                    self._load_state_dict_compatible(self.deployment_agent, dep_state, "deployment_agent")
                    LOGGER.info('[Hedger][Checkpoint] Loaded deployment agent state.')
                else:
                    LOGGER.warning('[Hedger][Checkpoint] Missing deployment_agent state in checkpoint.')
            else:
                LOGGER.info('[Hedger][Checkpoint] Skip deployment agent loading per config.')

            if self.checkpoint_cfg.load.restore_offloading_agent:
                off_state = ckpt.get('offloading_agent')
                if off_state is not None:
                    self._load_state_dict_compatible(self.offloading_agent, off_state, "offloading_agent")
                    LOGGER.info('[Hedger][Checkpoint] Loaded offloading agent state.')
                else:
                    LOGGER.warning('[Hedger][Checkpoint] Missing offloading_agent state in checkpoint.')
            else:
                LOGGER.info('[Hedger][Checkpoint] Skip offloading agent loading per config.')

            if self.checkpoint_cfg.load.restore_optimizer:
                if self.checkpoint_cfg.load.restore_deployment_agent and 'deployment_actor_opt' in ckpt:
                    try:
                        self.deployment_agent.actor_opt.load_state_dict(ckpt['deployment_actor_opt'])
                        self._move_optimizer_state(self.deployment_agent.actor_opt, self.device)
                        LOGGER.info('[Hedger][Checkpoint] Loaded deployment actor optimizer state.')
                    except ValueError as exc:
                        LOGGER.warning(
                            f"[Hedger][Checkpoint] Skip incompatible deployment actor optimizer state: {exc}"
                        )
                if self.checkpoint_cfg.load.restore_deployment_agent and 'deployment_critic_opt' in ckpt:
                    try:
                        self.deployment_agent.critic_opt.load_state_dict(ckpt['deployment_critic_opt'])
                        self._move_optimizer_state(self.deployment_agent.critic_opt, self.device)
                        LOGGER.info('[Hedger][Checkpoint] Loaded deployment critic optimizer state.')
                    except ValueError as exc:
                        LOGGER.warning(
                            f"[Hedger][Checkpoint] Skip incompatible deployment critic optimizer state: {exc}"
                        )
                if self.checkpoint_cfg.load.restore_offloading_agent and 'offloading_actor_opt' in ckpt:
                    try:
                        self.offloading_agent.actor_opt.load_state_dict(ckpt['offloading_actor_opt'])
                        self._move_optimizer_state(self.offloading_agent.actor_opt, self.device)
                        LOGGER.info('[Hedger][Checkpoint] Loaded offloading actor optimizer state.')
                    except ValueError as exc:
                        LOGGER.warning(
                            f"[Hedger][Checkpoint] Skip incompatible offloading actor optimizer state: {exc}"
                        )
                if self.checkpoint_cfg.load.restore_offloading_agent and 'offloading_critic_opt' in ckpt:
                    try:
                        self.offloading_agent.critic_opt.load_state_dict(ckpt['offloading_critic_opt'])
                        self._move_optimizer_state(self.offloading_agent.critic_opt, self.device)
                        LOGGER.info('[Hedger][Checkpoint] Loaded offloading critic optimizer state.')
                    except ValueError as exc:
                        LOGGER.warning(
                            f"[Hedger][Checkpoint] Skip incompatible offloading critic optimizer state: {exc}"
                        )
            else:
                LOGGER.info('[Hedger][Checkpoint] Skip optimizer state loading per config.')

        restored_global_step = int(meta.get('global_step', meta.get('stage_step', meta.get('epoch', 0))))
        self._global_update_step = restored_global_step

        reset_stage_counters = self.checkpoint_cfg.load.reset_stage_counters or not same_stage_resume
        if reset_stage_counters:
            self._deployment_update_steps = 0
            self._offloading_update_steps = 0
            self._epoch = 0
            LOGGER.info(
                f"[Hedger][Checkpoint] Stage-local counters reset: "
                f"same_stage_resume={same_stage_resume}, "
                f"reset_stage_counters={self.checkpoint_cfg.load.reset_stage_counters}, "
                f"global_step={self._global_update_step}"
            )
        else:
            self._deployment_update_steps = int(meta.get('deployment_updates', 0))
            self._offloading_update_steps = int(meta.get('offloading_updates', 0))
            self._epoch = int(meta.get('stage_step', meta.get('epoch', self._epoch)))
            LOGGER.info(
                f"[Hedger][Checkpoint] Restored stage-local counters: "
                f"dep_updates={self._deployment_update_steps}, "
                f"off_updates={self._offloading_update_steps}, "
                f"stage_step={self._epoch}, global_step={self._global_update_step}"
            )

        LOGGER.info(f"[Hedger][Checkpoint] Loaded successfully from {target_path}")

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device):
        """Ensure all optimizer state tensors live on the same device as the model."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    @staticmethod
    def _load_state_dict_compatible(module: torch.nn.Module, state_dict: dict, module_name: str) -> None:
        """
        Load only keys whose shapes still match the current module.

        This keeps staged training usable after small architectural changes such
        as critic input expansion, while still restoring all compatible weights.
        """
        current_state = module.state_dict()
        filtered_state = {}
        skipped_keys = []
        for key, value in (state_dict or {}).items():
            current_value = current_state.get(key)
            if current_value is None or current_value.shape != value.shape:
                skipped_keys.append(key)
                continue
            filtered_state[key] = value

        missing_keys, unexpected_keys = module.load_state_dict(filtered_state, strict=False)
        if skipped_keys:
            LOGGER.warning(
                f"[Hedger][Checkpoint] Skip incompatible {module_name} keys: {sorted(skipped_keys)}"
            )
        if missing_keys:
            LOGGER.warning(
                f"[Hedger][Checkpoint] Missing {module_name} keys after compatible load: {sorted(missing_keys)}"
            )
        if unexpected_keys:
            LOGGER.warning(
                f"[Hedger][Checkpoint] Unexpected {module_name} keys after compatible load: {sorted(unexpected_keys)}"
            )

    def _map_deployment_plan_to_deployment_mask(self, deployment_plan: dict):
        """
        Convert a deployment-plan dictionary into a deployment-mask tensor.

        `deployment_plan`: `service_name -> [device_name, ...]`

        Returns:
            `deploy_mask`: boolean tensor of shape `(num_services, num_devices)`.
            The cloud replica is always kept enabled for every service.
        """
        num_services = len(self.logical_topology)
        num_devices = len(self.physical_topology)
        deploy_mask = torch.zeros((num_services, num_devices), dtype=torch.bool, device=self.device)

        for service_name, device_names in (deployment_plan or {}).items():
            try:
                s_idx = self.logical_topology.index(service_name)
            except ValueError:
                LOGGER.debug(f"[Hedger][Deployment] Ignore unknown service in plan: {service_name}")
                continue
            if isinstance(device_names, (list, tuple, set)):
                iterable = device_names
            else:
                iterable = [device_names]
            for device_name in iterable:
                try:
                    d_idx = self.physical_topology.index(device_name)
                except ValueError:
                    LOGGER.debug(f"[Hedger][Deployment] Ignore unknown device in plan: {device_name}")
                    continue
                deploy_mask[s_idx, d_idx] = True

        deploy_mask[:, self.physical_topology.cloud_idx] = True

        return deploy_mask

    def _map_deployment_mask_to_deployment_plan(self, deploy_mask: torch.Tensor):
        """
        Convert a deployment-mask tensor into a deployment-plan dictionary.

        `deploy_mask`: boolean tensor of shape `(num_services, num_devices)`

        Returns:
            `deployment_plan`: `service_name -> [device_name, ...]`
        """
        deployment_plan = {}
        num_services = deploy_mask.size(0)
        num_devices = deploy_mask.size(1)

        for s_idx in range(num_services):
            for d_idx in range(num_devices):
                if deploy_mask[s_idx, d_idx]:
                    service_name = self.logical_topology[s_idx]
                    device_name = self.physical_topology[d_idx]
                    if service_name not in deployment_plan:
                        deployment_plan[service_name] = [device_name]
                    else:
                        deployment_plan[service_name].append(device_name)

        return deployment_plan

    def _map_offloading_mask_to_offloading_plan(self, offloading_mask: torch.Tensor):
        """
        Convert an offloading-action tensor into an offloading-plan dictionary.

        `offloading_mask`: integer tensor of shape `(num_services,)`, where
        each value is a device index.

        Returns:
            `offloading_plan`: `service_name -> device_name`
        """
        offloading_plan = {}
        num_services = offloading_mask.size(0)

        for s_idx in range(num_services):
            d_idx = offloading_mask[s_idx].item()
            service_name = self.logical_topology[s_idx]
            device_name = self.physical_topology[d_idx]
            offloading_plan[service_name] = device_name

        return offloading_plan

    @property
    def _ready_for_run(self):
        return self.physical_topology and self.logical_topology

    def run(self):
        wait_logged = False
        while not self._ready_for_run:
            if not wait_logged:
                LOGGER.debug(
                    f"[Hedger][Lifecycle] Waiting for topology registration: "
                    f"logical_ready={self.logical_topology is not None}, "
                    f"physical_ready={self.physical_topology is not None}"
                )
                wait_logged = True
            time.sleep(0.5)

        if self.mode == 'train':
            self.train_hedger()
        elif self.mode == 'inference':
            self.inference_hedger()
        else:
            raise ValueError(f'Unsupported mode {self.mode} for Hedger, only "train" and "inference" are supported.')
