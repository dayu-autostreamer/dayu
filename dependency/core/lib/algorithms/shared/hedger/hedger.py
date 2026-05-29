import copy
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import threading
import random
import math
from dataclasses import dataclass, field
import torch
import time
import os
import glob
import json

from core.lib.common import LOGGER, FileOps, Context, Recorder, KubeConfig
from core.lib.estimation import OverheadEstimator

from .topology_encoder import TopologyEncoders
from .ppo_agent import (
    HedgerOffloadingPPO,
    HedgerDeploymentPPO,
    SERVICE_DEMAND_FEATURE_NAMES,
    DEVICE_CAPABILITY_FEATURE_NAMES,
    RUNTIME_PAIR_FEATURE_NAMES,
)
from .hedger_config import from_partial_dict, DeploymentConstraintCfg, LogicalTopology, \
    PhysicalTopology
from .state_buffer import StateBuffer, BufferWaitCfg
from .deployment_dataset import (
    DeploymentTransitionDataset,
    DeploymentTransitionWriter,
    summarize_transition_quality,
)

__all__ = ('Hedger',)

TRAINING_STAGE_NAMES = {
    "offloading_warmup",
    "deployment_collect",
    "deployment_offline",
    "deployment_online",
    "joint_finetune",
}


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
    deployment_reward_min_samples: int
    deployment_feedback_timeout_s: Optional[float]


@dataclass(frozen=True)
class HedgerEncoderCfg:
    embedding_dim: int
    dropout: float


@dataclass(frozen=True)
class HedgerDeploymentDefaultWarmupCfg:
    enabled: bool = False
    min_intervals: int = 0
    min_feedback_samples: int = 0
    timeout_s: Optional[float] = None
    clear_feedback_window: bool = True


@dataclass(frozen=True)
class HedgerDeploymentDatasetCfg:
    enabled: bool = True
    root_dir: str = "deployment_dataset"
    shard_size: int = 32
    clear_on_collect_start: bool = True


@dataclass(frozen=True)
class HedgerDeploymentCollectCfg:
    keep_prob: float = 0.75
    actor_prob: float = 0.0
    perturb_prob: float = 0.25
    actor_logit_noise_std: float = 0.25
    safe_add_prob: float = 0.65
    safe_swap_prob: float = 0.30
    safe_remove_prob: float = 0.05
    max_perturb_flips: int = 1
    max_action_attempts: int = 6
    reset_on_bad_streak: bool = True
    bad_streak_threshold: int = 2
    reset_to_anchor_prob: float = 1.0
    max_queue_pressure: float = 0.65
    max_hotspot_cost: float = 0.08
    max_runtime_risk: float = 0.65
    min_pair_quality: float = 0.15
    max_queue_pressure_increase: float = 0.15
    max_hotspot_cost_increase: float = 0.04
    max_runtime_risk_increase: float = 0.20
    max_pair_quality_drop: float = 0.10
    max_capacity_relax_cost: float = 0.15
    allow_remove_last_edge_replica: bool = False
    fallback_to_best_candidate: bool = True


@dataclass(frozen=True)
class HedgerDeploymentOfflineRLCfg:
    batch_size: int = 32
    action_target: str = "executed"
    advantage_temperature: float = 1.0
    min_advantage_weight: float = 0.0
    max_advantage_weight: float = 20.0
    actor_bc_coef: float = 1.0
    negative_bc_coef: float = 0.2
    raw_removed_negative_coef: float = 0.0
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    bootstrap_current_value: bool = True
    offline_replay_ratio: float = 0.5
    online_replay_capacity: int = 512
    online_min_new_transitions: int = 1


@dataclass(frozen=True)
class HedgerDeploymentEventTriggerCfg:
    enabled: bool = False
    min_interval_s: float = 20.0
    cooldown_s: float = 30.0
    queue_pressure_threshold: float = 0.70
    hotspot_pressure_threshold: float = 0.45
    inference_warmup_grace_s: float = 45.0
    inference_min_feedback_samples: int = 5
    inference_require_feedback_before_trigger: bool = True
    e2e_slo_threshold: float = 0.18
    e2e_p95_threshold_s: float = 6.0
    e2e_min_feedback_samples: int = 12


@dataclass(frozen=True)
class HedgerTrainingCfg:
    stage: str
    total_updates: int
    ppo_epochs: int
    deployment_rollout_len: int
    offloading_rollout_len: int
    deployment_batch_size: int
    offloading_batch_size: int
    deployment_rollout_deterministic: bool = False
    offloading_rollout_deterministic: bool = False
    deployment_default_warmup: HedgerDeploymentDefaultWarmupCfg = field(
        default_factory=HedgerDeploymentDefaultWarmupCfg
    )
    deployment_dataset: HedgerDeploymentDatasetCfg = field(default_factory=HedgerDeploymentDatasetCfg)
    deployment_collect: HedgerDeploymentCollectCfg = field(default_factory=HedgerDeploymentCollectCfg)
    deployment_offline_rl: HedgerDeploymentOfflineRLCfg = field(default_factory=HedgerDeploymentOfflineRLCfg)


@dataclass(frozen=True)
class HedgerInferenceCfg:
    run_deployment_worker: bool = True
    run_offloading_worker: bool = True
    deployment_deterministic: bool = True
    offloading_deterministic: bool = True
    wait_for_initial_task_feedback: bool = True
    initial_task_feedback_min_samples: int = 2
    initial_task_feedback_timeout_s: Optional[float] = 60.0
    deployment_wait_until_served: bool = True
    deployment_require_version_matched_feedback: bool = True
    deployment_min_version_matched_samples: int = 0
    deployment_feedback_timeout_s: Optional[float] = None
    record_offloading_feedback: bool = True


@dataclass(frozen=True)
class HedgerRecordCfg:
    state_summary: bool = True
    debug_mode: bool = False
    state_snapshot_debug: bool = False
    actor_snapshot_debug: bool = False
    decision_candidate_features_debug: bool = False
    decision_actor_debug: bool = False
    normal_json_max_chars: int = 12000
    debug_json_max_chars: int = 200000


@dataclass(frozen=True)
class HedgerLatencyGuardCfg:
    enabled: bool = False
    latency_threshold_s: float = 3.0
    window_size: int = 32
    min_samples: int = 16
    trigger_violation_ratio: float = 0.5
    trigger_consecutive_windows: int = 2
    recover_violation_ratio: float = 0.2
    recover_consecutive_windows: int = 3
    poll_interval_s: float = 1.0
    clear_transition_buffers: bool = True
    force_default_decisions: bool = True
    pause_generation: bool = False
    queue_recovery_enabled: bool = True
    queue_recovery_threshold: float = 0.0
    queue_recovery_consecutive_updates: int = 3
    queue_recovery_stable_s: float = 10.0
    queue_recovery_stale_timeout_s: float = 20.0
    min_pause_s: float = 10.0
    log_interval_s: float = 10.0
    queue_flush_enabled: bool = False
    queue_flush_on_trigger: bool = True
    queue_flush_scope: str = "observed"
    queue_flush_threshold: float = 0.0
    queue_flush_max_count_per_service: Optional[int] = None
    queue_flush_min_interval_s: float = 30.0
    queue_flush_timeout_s: float = 5.0
    queue_flush_dry_run: bool = False
    task_feedback_quarantine_enabled: bool = True
    task_feedback_quarantine_s: float = 5.0
    clear_latency_window_on_recovery: bool = True


@dataclass(frozen=True)
class HedgerDeploymentFeedbackWaitResult:
    ok: bool
    count: int = 0
    timed_out: bool = False
    timeout_s: Optional[float] = None
    guard_interrupted: bool = False
    guard_event_seq: int = 0
    guard_stats: Dict[str, Any] = field(default_factory=dict)


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
class HedgerPPOUpdateCfg:
    entropy_coef: float
    entropy_coef_final: float
    entropy_decay_updates: int
    value_coef: float


@dataclass(frozen=True)
class HedgerTrainingStageCfg:
    name: str
    run_deployment_worker: bool
    update_deployment_policy: bool
    run_offloading_worker: bool
    update_offloading_policy: bool
    use_frozen_offloading_rollout: bool = False
    deployment_train_mode: str = "none"


class Hedger:
    def __init__(self, config: dict):
        config = copy.deepcopy(config or {})

        self.mode = self._require_choice(config, "mode", {"train", "inference"})
        self.agent_id = int(config.get("agent_id", 0))
        self.device = torch.device(self._require_str(config, "device"))
        self.seed = int(config.get("seed", 0))

        timing_cfg = self._require_mapping(config, "timing")
        self.deployment_interval = float(timing_cfg["deployment_interval_s"])
        self.offloading_interval = float(timing_cfg["offloading_interval_s"])

        self.encoder_cfg = self._build_encoder_cfg(config)
        self.state_cfg = self._build_state_cfg(config)
        self.training_cfg = self._build_training_cfg(config) if self.mode == "train" else None
        self.stage_cfg = self._build_training_stage_cfg(self.training_cfg.stage) if self.training_cfg is not None else None
        self.inference_cfg = self._build_inference_cfg(config)
        self.record_cfg = self._build_record_cfg(config)
        self.latency_guard_cfg = self._build_latency_guard_cfg(config)
        self.deployment_event_trigger_cfg = self._build_deployment_event_trigger_cfg(config)
        self.checkpoint_cfg = self._build_checkpoint_cfg(config)

        agents_cfg = self._require_mapping(config, "agents")
        self.deployment_agent_params = self._build_deployment_agent_params(agents_cfg)
        self.offloading_agent_params = self._build_offloading_agent_params(agents_cfg)

        self.deployment_thread_stop_event = threading.Event()
        self.offloading_thread_stop_event = threading.Event()
        self._latency_guard_trigger_event = threading.Event()
        self._deployment_event_trigger_event = threading.Event()

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
        self._latency_guard_lock = threading.Lock()
        self._deployment_version_cond = threading.Condition(threading.Lock())
        self._deployment_decision_version = 0
        self._deployment_served_version = 0
        self._last_processor_pod_cleanup_t = 0.0
        self._processor_pod_cleanup_cooldown_s = 30.0
        self._run_thread = None
        self._run_started = False

        FileOps.create_directory(self.checkpoint_cfg.root_dir)
        if self.checkpoint_cfg.load.enabled:
            self.load_checkpoint()
        if self.mode == "inference" and self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "[Hedger][Inference] No checkpoint was loaded. "
                "Set checkpoint.load.enabled=true and provide a valid from_stage/which or path."
            )

        self.initial_deployment_plan = None
        self.deployment_plan = None
        self.offloading_plan = None

        self.deployment_transitions: List[dict] = []
        self.offloading_transitions: List[dict] = []
        self.deployment_dataset_writer: Optional[DeploymentTransitionWriter] = None
        self.deployment_offline_dataset: Optional[DeploymentTransitionDataset] = None
        self._deployment_collected_transition_count = 0
        self._deployment_last_online_update_transition_count = 0
        self._deployment_last_collect_checkpoint_step = 0
        self._deployment_collect_anchor_mask: Optional[torch.Tensor] = None
        self._deployment_collect_bad_streak = 0
        self._deployment_collect_good_count = 0
        self._deployment_collect_bad_count = 0
        self._deployment_collect_last_quality = "unknown"
        self._init_deployment_dataset_runtime()

        self.dep_recorder = None
        self.off_recorder = None
        self.dep_update_recorder = None
        self.off_update_recorder = None
        self.dep_decision_recorder = None
        self.off_decision_recorder = None
        self.deployment_overhead_estimator = None
        self.offloading_overhead_estimator = None
        if self.mode == "inference":
            self.deployment_overhead_estimator = OverheadEstimator(
                "Hedger-Deployment",
                "scheduler/hedger",
                agent_id=self.agent_id,
            )
            self.offloading_overhead_estimator = OverheadEstimator(
                "Hedger-Offloading",
                "scheduler/hedger",
                agent_id=self.agent_id,
            )

        self.cur_deploy_mask = None
        self.pending_deployment_plan = None
        self.pending_deploy_mask = None
        self._pending_deployment_force_serve = False
        self._pending_deployment_reason = None
        self._last_deployment_served_monotonic = 0.0
        self._last_deployment_event_trigger_monotonic = 0.0
        self._last_deployment_event_trigger_record: Dict[str, Any] = {}
        self._last_deployment_event_suppressed_record: Dict[str, Any] = {}
        self._last_redeployment_throttle_log_t = 0.0
        self._frozen_offloading_agent = None
        self._latency_guard_samples = deque(maxlen=self.latency_guard_cfg.window_size)
        self._latency_guard_active = False
        self._latency_guard_trigger_streak = 0
        self._latency_guard_recover_streak = 0
        self._latency_guard_queue_recover_streak = 0
        self._latency_guard_queue_drained_since_t = 0.0
        self._latency_guard_activated_t = 0.0
        self._latency_guard_activated_wall_t = 0.0
        self._latency_guard_recovered_wall_t = 0.0
        self._latency_guard_feedback_quarantine_until_t = 0.0
        self._latency_guard_queue_observations = {}
        self._latency_guard_last_queue_flush_t = 0.0
        self._latency_guard_queue_flush_command_seq = 0
        self._latency_guard_queue_flush_command = None
        self._latency_guard_last_log_t = 0.0
        self._latency_guard_last_stats = self._empty_latency_guard_stats()
        self._latency_guard_trigger_seq = 0
        self._latency_guard_trigger_deployment_version = None
        self._latency_guard_trigger_task_version = None
        self._latency_guard_trigger_stats = {}

        if self.mode == "inference":
            self.ensure_started("inference initialization")
        elif self.stage_cfg is not None and self.stage_cfg.deployment_train_mode == "offline":
            LOGGER.info(
                "[Hedger][Lifecycle] Deployment offline training will start once topology, "
                "state buffer, and initial deployment are registered."
            )
        else:
            LOGGER.info("[Hedger][Lifecycle] Training start is deferred until the first task update arrives.")

    def _init_deployment_dataset_runtime(self) -> None:
        if self.mode != "train" or self.training_cfg is None:
            return
        if not self.training_cfg.deployment_dataset.enabled:
            return
        mode = self.stage_cfg.deployment_train_mode if self.stage_cfg is not None else "none"
        if mode in {"collect", "online"}:
            self.deployment_dataset_writer = DeploymentTransitionWriter(
                self.training_cfg.deployment_dataset.root_dir,
                shard_size=self.training_cfg.deployment_dataset.shard_size,
                clear_existing=(
                    mode == "collect"
                    and self.training_cfg.deployment_dataset.clear_on_collect_start
                ),
            )
            self._deployment_collected_transition_count = self.deployment_dataset_writer.count
            LOGGER.info(
                f"[Hedger][DeploymentDataset] Writer ready: "
                f"mode={mode}, root={self.training_cfg.deployment_dataset.root_dir}, "
                f"count={self._deployment_collected_transition_count}"
            )
        if mode in {"offline", "online"}:
            self.deployment_offline_dataset = DeploymentTransitionDataset(
                self.training_cfg.deployment_dataset.root_dir
            )
            LOGGER.info(
                f"[Hedger][DeploymentDataset] Loaded offline replay: "
                f"root={self.training_cfg.deployment_dataset.root_dir}, "
                f"samples={len(self.deployment_offline_dataset)}"
            )

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

    def _maybe_autostart_deployment_offline(self, reason: str) -> bool:
        if self.mode != "train" or self.stage_cfg is None:
            return False
        if self.stage_cfg.deployment_train_mode != "offline":
            return False
        if self.initial_deployment_plan is None:
            LOGGER.debug(
                f"[Hedger][Lifecycle] Deployment offline autostart waits for initial deployment: "
                f"reason={reason}"
            )
            return False
        if not self._ready_for_run:
            LOGGER.debug(
                f"[Hedger][Lifecycle] Deployment offline autostart waits for topology/state buffer: "
                f"reason={reason}, logical_ready={self.logical_topology is not None}, "
                f"physical_ready={self.physical_topology is not None}, "
                f"state_buffer_ready={self.state_buffer is not None}"
            )
            return False
        return self.ensure_started(reason)

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
                "'offloading_warmup', 'deployment_collect', 'deployment_offline', "
                "'deployment_online', 'joint_finetune'."
            )
        default_warmup = training.get("deployment_default_warmup") or {}
        if not isinstance(default_warmup, dict):
            raise ValueError("Hedger config `training.deployment_default_warmup` must be a mapping when provided.")
        default_warmup_enabled = bool(default_warmup.get("enabled", False))
        if default_warmup_enabled:
            default_min_intervals = 2
            default_min_feedback = self.state_cfg.deployment_reward_min_samples
        else:
            default_min_intervals = 0
            default_min_feedback = 0
        default_warmup_timeout = default_warmup.get("timeout_s")
        default_warmup_timeout = (
            None if default_warmup_timeout is None else max(0.0, float(default_warmup_timeout))
        )
        deployment_default_warmup = HedgerDeploymentDefaultWarmupCfg(
            enabled=default_warmup_enabled,
            min_intervals=max(0, int(default_warmup.get("min_intervals", default_min_intervals))),
            min_feedback_samples=max(0, int(default_warmup.get("min_feedback_samples", default_min_feedback))),
            timeout_s=default_warmup_timeout,
            clear_feedback_window=bool(default_warmup.get("clear_feedback_window", True)),
        )

        dataset_cfg = training.get("deployment_dataset") or {}
        if not isinstance(dataset_cfg, dict):
            raise ValueError("Hedger config `training.deployment_dataset` must be a mapping when provided.")
        deployment_dataset = HedgerDeploymentDatasetCfg(
            enabled=bool(dataset_cfg.get("enabled", True)),
            root_dir=self._resolve_path(str(dataset_cfg.get("root_dir", "deployment_dataset"))),
            shard_size=max(1, int(dataset_cfg.get("shard_size", 32))),
            clear_on_collect_start=bool(dataset_cfg.get("clear_on_collect_start", True)),
        )

        collect_cfg = training.get("deployment_collect") or {}
        if not isinstance(collect_cfg, dict):
            raise ValueError("Hedger config `training.deployment_collect` must be a mapping when provided.")
        keep_prob = max(0.0, float(collect_cfg.get("keep_prob", 0.75)))
        actor_prob = max(0.0, float(collect_cfg.get("actor_prob", 0.0)))
        perturb_prob = max(0.0, float(collect_cfg.get("perturb_prob", 0.25)))
        total_prob = keep_prob + actor_prob + perturb_prob
        if total_prob <= 0.0:
            keep_prob, actor_prob, perturb_prob = 1.0, 0.0, 0.0
            total_prob = 1.0
        safe_add_prob = max(0.0, float(collect_cfg.get("safe_add_prob", 0.65)))
        safe_swap_prob = max(0.0, float(collect_cfg.get("safe_swap_prob", 0.30)))
        safe_remove_prob = max(0.0, float(collect_cfg.get("safe_remove_prob", 0.05)))
        safe_total_prob = safe_add_prob + safe_swap_prob + safe_remove_prob
        if safe_total_prob <= 0.0:
            safe_add_prob, safe_swap_prob, safe_remove_prob = 1.0, 0.0, 0.0
            safe_total_prob = 1.0
        deployment_collect = HedgerDeploymentCollectCfg(
            keep_prob=keep_prob / total_prob,
            actor_prob=actor_prob / total_prob,
            perturb_prob=perturb_prob / total_prob,
            actor_logit_noise_std=max(0.0, float(collect_cfg.get("actor_logit_noise_std", 0.25))),
            safe_add_prob=safe_add_prob / safe_total_prob,
            safe_swap_prob=safe_swap_prob / safe_total_prob,
            safe_remove_prob=safe_remove_prob / safe_total_prob,
            max_perturb_flips=max(1, int(collect_cfg.get("max_perturb_flips", 1))),
            max_action_attempts=max(1, int(collect_cfg.get("max_action_attempts", 6))),
            reset_on_bad_streak=bool(collect_cfg.get("reset_on_bad_streak", True)),
            bad_streak_threshold=max(1, int(collect_cfg.get("bad_streak_threshold", 2))),
            reset_to_anchor_prob=min(
                1.0,
                max(0.0, float(collect_cfg.get("reset_to_anchor_prob", 1.0))),
            ),
            max_queue_pressure=max(0.0, float(collect_cfg.get("max_queue_pressure", 0.65))),
            max_hotspot_cost=max(0.0, float(collect_cfg.get("max_hotspot_cost", 0.08))),
            max_runtime_risk=max(0.0, float(collect_cfg.get("max_runtime_risk", 0.65))),
            min_pair_quality=max(0.0, float(collect_cfg.get("min_pair_quality", 0.15))),
            max_queue_pressure_increase=max(0.0, float(collect_cfg.get("max_queue_pressure_increase", 0.15))),
            max_hotspot_cost_increase=max(0.0, float(collect_cfg.get("max_hotspot_cost_increase", 0.04))),
            max_runtime_risk_increase=max(0.0, float(collect_cfg.get("max_runtime_risk_increase", 0.20))),
            max_pair_quality_drop=max(0.0, float(collect_cfg.get("max_pair_quality_drop", 0.10))),
            max_capacity_relax_cost=max(0.0, float(collect_cfg.get("max_capacity_relax_cost", 0.15))),
            allow_remove_last_edge_replica=bool(collect_cfg.get("allow_remove_last_edge_replica", False)),
            fallback_to_best_candidate=bool(collect_cfg.get("fallback_to_best_candidate", True)),
        )

        offline_rl_cfg = training.get("deployment_offline_rl") or {}
        if not isinstance(offline_rl_cfg, dict):
            raise ValueError("Hedger config `training.deployment_offline_rl` must be a mapping when provided.")
        action_target = str(offline_rl_cfg.get("action_target", "executed")).strip().lower()
        if action_target not in {"executed", "raw"}:
            raise ValueError("Hedger training.deployment_offline_rl.action_target must be 'executed' or 'raw'.")
        deployment_offline_rl = HedgerDeploymentOfflineRLCfg(
            batch_size=max(1, int(offline_rl_cfg.get("batch_size", batch_size["deployment"]))),
            action_target=action_target,
            advantage_temperature=max(1e-6, float(offline_rl_cfg.get("advantage_temperature", 1.0))),
            min_advantage_weight=max(0.0, float(offline_rl_cfg.get("min_advantage_weight", 0.0))),
            max_advantage_weight=max(1.0, float(offline_rl_cfg.get("max_advantage_weight", 20.0))),
            actor_bc_coef=max(0.0, float(offline_rl_cfg.get("actor_bc_coef", 1.0))),
            negative_bc_coef=max(0.0, float(offline_rl_cfg.get("negative_bc_coef", 0.2))),
            raw_removed_negative_coef=max(0.0, float(offline_rl_cfg.get("raw_removed_negative_coef", 0.0))),
            value_coef=max(0.0, float(offline_rl_cfg.get("value_coef", 0.5))),
            entropy_coef=max(0.0, float(offline_rl_cfg.get("entropy_coef", 0.0))),
            bootstrap_current_value=bool(offline_rl_cfg.get("bootstrap_current_value", True)),
            offline_replay_ratio=min(1.0, max(0.0, float(offline_rl_cfg.get("offline_replay_ratio", 0.5)))),
            online_replay_capacity=max(1, int(offline_rl_cfg.get("online_replay_capacity", 512))),
            online_min_new_transitions=max(1, int(offline_rl_cfg.get("online_min_new_transitions", 1))),
        )

        return HedgerTrainingCfg(
            stage=stage,
            total_updates=max(0, int(training["total_updates"])),
            ppo_epochs=max(1, int(training["ppo_epochs"])),
            deployment_rollout_len=max(1, int(rollout["deployment"])),
            offloading_rollout_len=max(1, int(rollout["offloading"])),
            deployment_batch_size=max(1, int(batch_size["deployment"])),
            offloading_batch_size=max(1, int(batch_size["offloading"])),
            deployment_rollout_deterministic=bool(training.get("deployment_rollout_deterministic", False)),
            offloading_rollout_deterministic=bool(training.get("offloading_rollout_deterministic", False)),
            deployment_default_warmup=deployment_default_warmup,
            deployment_dataset=deployment_dataset,
            deployment_collect=deployment_collect,
            deployment_offline_rl=deployment_offline_rl,
        )

    def _build_inference_cfg(self, config: dict) -> HedgerInferenceCfg:
        inference = config.get("inference") or {}
        if not isinstance(inference, dict):
            raise ValueError("Hedger config `inference` must be a mapping when provided.")

        initial_timeout = inference.get("initial_task_feedback_timeout_s", 60.0)
        initial_timeout = None if initial_timeout is None else max(0.0, float(initial_timeout))

        feedback_timeout = inference.get(
            "deployment_feedback_timeout_s",
            self.state_cfg.deployment_feedback_timeout_s,
        )
        feedback_timeout = None if feedback_timeout is None else max(0.0, float(feedback_timeout))

        default_min_samples = self.state_cfg.deployment_reward_min_samples
        configured_min_samples = int(inference.get(
            "deployment_min_version_matched_samples",
            default_min_samples,
        ))

        run_deployment_worker = bool(inference.get("run_deployment_worker", True))
        run_offloading_worker = bool(inference.get("run_offloading_worker", True))
        if not run_deployment_worker and not run_offloading_worker:
            raise ValueError(
                "Hedger inference config requires at least one enabled worker: "
                "`inference.run_deployment_worker` or `inference.run_offloading_worker`."
            )

        return HedgerInferenceCfg(
            run_deployment_worker=run_deployment_worker,
            run_offloading_worker=run_offloading_worker,
            deployment_deterministic=bool(inference.get("deployment_deterministic", True)),
            offloading_deterministic=bool(inference.get("offloading_deterministic", True)),
            wait_for_initial_task_feedback=bool(inference.get("wait_for_initial_task_feedback", True)),
            initial_task_feedback_min_samples=max(
                1,
                int(inference.get(
                    "initial_task_feedback_min_samples",
                    max(1, self.state_cfg.min_dynamic_len),
                )),
            ),
            initial_task_feedback_timeout_s=initial_timeout,
            deployment_wait_until_served=bool(inference.get("deployment_wait_until_served", True)),
            deployment_require_version_matched_feedback=bool(
                inference.get("deployment_require_version_matched_feedback", True)
            ),
            deployment_min_version_matched_samples=max(1, configured_min_samples),
            deployment_feedback_timeout_s=feedback_timeout,
            record_offloading_feedback=bool(inference.get("record_offloading_feedback", True)),
        )

    def _build_record_cfg(self, config: dict) -> HedgerRecordCfg:
        record = config.get("record") or {}
        if not isinstance(record, dict):
            raise ValueError("Hedger config `record` must be a mapping when provided.")
        debug_mode = bool(record.get("debug_mode", False))
        return HedgerRecordCfg(
            state_summary=bool(record.get("state_summary", True)),
            debug_mode=debug_mode,
            state_snapshot_debug=debug_mode or bool(record.get("state_snapshot_debug", False)),
            actor_snapshot_debug=debug_mode or bool(record.get("actor_snapshot_debug", False)),
            decision_candidate_features_debug=debug_mode or bool(
                record.get("decision_candidate_features_debug", False)
            ),
            decision_actor_debug=debug_mode or bool(record.get("decision_actor_debug", False)),
            normal_json_max_chars=max(0, int(record.get("normal_json_max_chars", 12000))),
            debug_json_max_chars=max(0, int(record.get("debug_json_max_chars", 200000))),
        )

    def _build_state_cfg(self, config: dict) -> HedgerStateCfg:
        state = self._require_mapping(config, "state")
        sequence_length = self._require_mapping(state, "sequence_length")
        reward_window_default = max(1, int(math.ceil(self.deployment_interval / max(self.offloading_interval, 1e-6))))
        wait_timeout_s = state.get("wait_timeout_s", 1.0)
        wait_timeout_s = None if wait_timeout_s is None else max(0.0, float(wait_timeout_s))
        feedback_timeout_s = state.get("deployment_feedback_timeout_s", self.deployment_interval)
        feedback_timeout_s = None if feedback_timeout_s is None else max(0.0, float(feedback_timeout_s))
        return HedgerStateCfg(
            max_buffer_size=max(1, int(state["max_buffer_size"])),
            offloading_seq_len=max(1, int(sequence_length["offloading"])),
            deployment_seq_len=max(1, int(sequence_length["deployment"])),
            min_dynamic_len=max(0, int(state.get("min_dynamic_length", 1))),
            wait_timeout_s=wait_timeout_s,
            require_full_seq=bool(state.get("require_full_sequence", False)),
            latency_slo=float(state["latency_slo_s"]),
            deployment_reward_window=max(1, int(state.get("deployment_reward_window", reward_window_default))),
            deployment_reward_min_samples=max(1, int(state.get("deployment_reward_min_samples", 1))),
            deployment_feedback_timeout_s=feedback_timeout_s,
        )

    def _build_latency_guard_cfg(self, config: dict) -> HedgerLatencyGuardCfg:
        guard = config.get("latency_guard") or {}
        if not isinstance(guard, dict):
            raise ValueError("Hedger latency_guard must be a mapping when provided.")

        slo = self.state_cfg.latency_slo
        default_threshold = 3.0
        if slo is not None and math.isfinite(float(slo)) and float(slo) > 0.0:
            default_threshold = max(float(slo) * 1.5, float(slo) + 0.5)

        window_size = max(1, int(guard.get("window_size", 32)))
        min_samples = max(1, int(guard.get("min_samples", min(16, window_size))))
        min_samples = min(min_samples, window_size)

        trigger_ratio = min(1.0, max(0.0, float(guard.get("trigger_violation_ratio", 0.5))))
        recover_ratio = min(1.0, max(0.0, float(guard.get("recover_violation_ratio", 0.2))))
        if recover_ratio > trigger_ratio:
            raise ValueError(
                "Hedger latency_guard.recover_violation_ratio must be <= trigger_violation_ratio."
            )

        queue_flush_scope = str(guard.get("queue_flush_scope", "observed")).strip().lower()
        if queue_flush_scope not in {"observed", "all_devices"}:
            raise ValueError("Hedger latency_guard.queue_flush_scope must be one of: observed, all_devices.")

        queue_flush_max_count = guard.get("queue_flush_max_count_per_service")
        if queue_flush_max_count is not None:
            queue_flush_max_count = max(1, int(queue_flush_max_count))

        return HedgerLatencyGuardCfg(
            enabled=bool(guard.get("enabled", False)),
            latency_threshold_s=max(0.0, float(guard.get("latency_threshold_s", default_threshold))),
            window_size=window_size,
            min_samples=min_samples,
            trigger_violation_ratio=trigger_ratio,
            trigger_consecutive_windows=max(1, int(guard.get("trigger_consecutive_windows", 2))),
            recover_violation_ratio=recover_ratio,
            recover_consecutive_windows=max(1, int(guard.get("recover_consecutive_windows", 3))),
            poll_interval_s=max(0.1, float(guard.get("poll_interval_s", 1.0))),
            clear_transition_buffers=bool(guard.get("clear_transition_buffers", True)),
            force_default_decisions=bool(guard.get("force_default_decisions", True)),
            pause_generation=bool(guard.get("pause_generation", False)),
            queue_recovery_enabled=bool(guard.get("queue_recovery_enabled", True)),
            queue_recovery_threshold=max(0.0, float(guard.get("queue_recovery_threshold", 0.0))),
            queue_recovery_consecutive_updates=max(
                1,
                int(guard.get("queue_recovery_consecutive_updates", 3)),
            ),
            queue_recovery_stable_s=max(0.0, float(guard.get("queue_recovery_stable_s", 10.0))),
            queue_recovery_stale_timeout_s=max(
                0.1,
                float(guard.get("queue_recovery_stale_timeout_s", 20.0)),
            ),
            min_pause_s=max(0.0, float(guard.get("min_pause_s", 10.0))),
            log_interval_s=max(1.0, float(guard.get("log_interval_s", 10.0))),
            queue_flush_enabled=bool(guard.get("queue_flush_enabled", False)),
            queue_flush_on_trigger=bool(guard.get("queue_flush_on_trigger", True)),
            queue_flush_scope=queue_flush_scope,
            queue_flush_threshold=max(0.0, float(guard.get(
                "queue_flush_threshold",
                guard.get("queue_recovery_threshold", 0.0),
            ))),
            queue_flush_max_count_per_service=queue_flush_max_count,
            queue_flush_min_interval_s=max(0.0, float(guard.get("queue_flush_min_interval_s", 30.0))),
            queue_flush_timeout_s=max(0.1, float(guard.get("queue_flush_timeout_s", 5.0))),
            queue_flush_dry_run=bool(guard.get("queue_flush_dry_run", False)),
            task_feedback_quarantine_enabled=bool(guard.get("task_feedback_quarantine_enabled", True)),
            task_feedback_quarantine_s=max(0.0, float(guard.get("task_feedback_quarantine_s", 5.0))),
            clear_latency_window_on_recovery=bool(guard.get("clear_latency_window_on_recovery", True)),
        )

    def _build_deployment_event_trigger_cfg(self, config: dict) -> HedgerDeploymentEventTriggerCfg:
        event = config.get("deployment_event_trigger") or {}
        if not isinstance(event, dict):
            raise ValueError("Hedger deployment_event_trigger must be a mapping when provided.")
        return HedgerDeploymentEventTriggerCfg(
            enabled=bool(event.get("enabled", False)),
            min_interval_s=max(0.0, float(event.get("min_interval_s", 20.0))),
            cooldown_s=max(0.0, float(event.get("cooldown_s", 30.0))),
            queue_pressure_threshold=min(1.0, max(0.0, float(event.get("queue_pressure_threshold", 0.70)))),
            hotspot_pressure_threshold=min(
                1.0,
                max(0.0, float(event.get("hotspot_pressure_threshold", 0.45))),
            ),
            inference_warmup_grace_s=max(
                0.0,
                float(event.get("inference_warmup_grace_s", 45.0)),
            ),
            inference_min_feedback_samples=max(
                0,
                int(event.get("inference_min_feedback_samples", 5)),
            ),
            inference_require_feedback_before_trigger=bool(
                event.get("inference_require_feedback_before_trigger", True)
            ),
            e2e_slo_threshold=min(
                1.0,
                max(0.0, float(event.get("e2e_slo_threshold", 0.18))),
            ),
            e2e_p95_threshold_s=max(0.0, float(event.get("e2e_p95_threshold_s", 6.0))),
            e2e_min_feedback_samples=max(1, int(event.get("e2e_min_feedback_samples", 12))),
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
                    "'offloading_warmup', 'deployment_collect', 'deployment_offline', "
                    "'deployment_online', 'joint_finetune'."
                )
        load_enabled = bool(load_cfg.get("enabled", False))
        if self.mode == "inference" and not load_enabled:
            raise ValueError(
                "Hedger inference mode requires checkpoint.load.enabled=true "
                "to avoid serving randomly initialized policies."
            )
        if self.mode == "inference" and load_enabled and which != "path" and from_stage is None:
            raise ValueError(
                "Hedger inference mode requires checkpoint.load.from_stage unless checkpoint.load.which='path'."
            )

        keep_last_snapshots = save_cfg.get("keep_last")
        if keep_last_snapshots is not None:
            keep_last_snapshots = max(1, int(keep_last_snapshots))

        return HedgerCheckpointCfg(
            root_dir=self._resolve_path(checkpoint["root_dir"]),
            load=HedgerCheckpointLoadCfg(
                enabled=load_enabled,
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

    def _inference_log_path(self, filename: str) -> str:
        return os.path.join(self._checkpoint_stage_dir("inference"), filename)

    @staticmethod
    def _ppo_update_fieldnames(include_offline_batch: bool = False) -> List[str]:
        fieldnames = [
            "agent", "update", "epoch", "used", "remaining",
            "samples", "epochs", "batch_size", "minibatches",
            "reward_mean", "reward_std", "reward_min", "reward_max",
            "value_old_mean", "value_old_std", "value_new_mean",
            "return_mean", "return_std", "adv_mean", "adv_std",
            "last_value", "done_fraction",
            "policy_loss", "value_loss", "entropy", "entropy_coef", "value_coef", "approx_kl",
            "clip_fraction", "ratio_mean", "ratio_std",
            "actor_grad_norm", "critic_grad_norm",
            "negative_loss", "raw_removed_negative_loss",
            "actor_positive_weight_mean", "actor_negative_weight_mean",
            "actor_raw_removed_weight_mean",
            "actor_positive_samples", "actor_negative_samples", "actor_raw_removed_samples",
            "actor_selected_risky_samples", "actor_selected_low_quality_samples",
            "actor_selected_runtime_risky_samples", "actor_selected_unknown_samples",
            "actor_selected_stale_samples",
            "bad_actor_masked",
            "positive_logp_mean", "negative_logp_mean", "raw_removed_logp_mean",
        ]
        if include_offline_batch:
            fieldnames.extend([
                "offline_batch_good", "offline_batch_mid", "offline_batch_bad", "offline_batch_unknown",
                "offline_batch_bad_ratio", "offline_batch_reward_mean",
                "offline_batch_slo_violation_mean", "offline_batch_latency_p95_mean",
                "offline_batch_guard_interrupted", "offline_batch_collect_risk_mean",
                "offline_batch_queue_pressure_mean", "offline_batch_hotspot_cost_mean",
                "offline_batch_runtime_risk_mean", "offline_batch_min_pair_quality_mean",
            ])
        return fieldnames

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
        if stage_name == "deployment_collect":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=True,
                update_deployment_policy=False,
                run_offloading_worker=True,
                update_offloading_policy=False,
                use_frozen_offloading_rollout=True,
                deployment_train_mode="collect",
            )
        if stage_name == "deployment_offline":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=False,
                update_deployment_policy=True,
                run_offloading_worker=False,
                update_offloading_policy=False,
                use_frozen_offloading_rollout=False,
                deployment_train_mode="offline",
            )
        if stage_name == "deployment_online":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=True,
                update_deployment_policy=True,
                run_offloading_worker=True,
                update_offloading_policy=False,
                use_frozen_offloading_rollout=True,
                deployment_train_mode="online",
            )
        if stage_name == "joint_finetune":
            return HedgerTrainingStageCfg(
                name=stage_name,
                run_deployment_worker=True,
                update_deployment_policy=True,
                run_offloading_worker=True,
                update_offloading_policy=True,
                deployment_train_mode="ppo",
            )

        raise ValueError(
            f"Unsupported training stage {stage_name!r}. "
            f"Expected one of: offloading_warmup, deployment_collect, "
            f"deployment_offline, deployment_online, joint_finetune."
        )

    def _build_deployment_agent_params(self, agents_cfg: dict) -> dict:
        deployment = self._require_mapping(agents_cfg, "deployment")
        reward = self._require_mapping(deployment, "reward")
        penalty = self._require_mapping(deployment, "penalty")
        constraints = deployment.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ValueError("Hedger config `agents.deployment.constraints` must be a mapping when provided.")
        hotspot = deployment.get("hotspot") or {}
        if not isinstance(hotspot, dict):
            raise ValueError("Hedger config `agents.deployment.hotspot` must be a mapping when provided.")
        matrix_policy = deployment.get("matrix_policy") or {}
        if not isinstance(matrix_policy, dict):
            raise ValueError("Hedger config `agents.deployment.matrix_policy` must be a mapping when provided.")
        max_edge_replicas = constraints.get("max_edge_replicas_per_device")
        if max_edge_replicas is not None:
            max_edge_replicas = int(max_edge_replicas)
            if max_edge_replicas <= 0:
                max_edge_replicas = None
        edge_memory_budget_ratio = float(constraints.get("edge_memory_budget_ratio", 1.0))
        if not math.isfinite(edge_memory_budget_ratio) or not (0.0 < edge_memory_budget_ratio <= 1.0):
            raise ValueError("Hedger config `agents.deployment.constraints.edge_memory_budget_ratio` must be in (0, 1].")
        queue_normalizer = float(hotspot.get("queue_normalizer", 8.0))
        if not math.isfinite(queue_normalizer) or queue_normalizer <= 0.0:
            raise ValueError("Hedger config `agents.deployment.hotspot.queue_normalizer` must be > 0.")
        def _matrix_float(name: str, default: float, *, probability: bool = False) -> float:
            value = float(matrix_policy.get(name, default))
            if not math.isfinite(value):
                raise ValueError(f"Hedger config `agents.deployment.matrix_policy.{name}` must be finite.")
            if probability and not (0.0 <= value <= 1.0):
                raise ValueError(f"Hedger config `agents.deployment.matrix_policy.{name}` must be in [0, 1].")
            return value
        dep_latency_cfg = self._parse_latency_reward_cfg(
            reward,
            scope="deployment",
            default_clip=6.0,
        )
        ppo = self._build_ppo_update_cfg(deployment)
        return {
            "actor_lr": float(deployment["actor_lr"]),
            "critic_lr": float(deployment["critic_lr"]),
            "gamma": float(deployment["gamma"]),
            "lamda": float(deployment["lamda"]),
            "clip_eps": float(deployment["clip_eps"]),
            "update_encoder": bool(deployment.get("update_encoder", True)),
            "reward_dep_offload_weight": float(reward["offloading_weight"]),
            "reward_dep_latency_weight": float(reward.get("latency_weight", 0.0)),
            "reward_dep_slo_weight": float(reward.get("slo_weight", 0.0)),
            "reward_dep_change_weight": float(reward["change_cost_weight"]),
            "reward_dep_cloud_only_weight": float(reward.get("cloud_only_weight", 0.0)),
            "reward_dep_runtime_risk_weight": float(reward.get("runtime_risk_weight", 0.0)),
            "reward_dep_unknown_option_weight": float(reward.get("unknown_option_weight", 0.0)),
            "reward_dep_stale_option_weight": float(reward.get("stale_option_weight", 0.0)),
            "reward_dep_low_quality_weight": float(reward.get("low_quality_weight", 0.0)),
            "reward_dep_latency_transform": dep_latency_cfg["transform"],
            "reward_dep_latency_normalizer": dep_latency_cfg["normalizer"],
            "reward_dep_latency_clip": dep_latency_cfg["clip"],
            "penalty_capacity_relax": float(penalty["capacity_relax"]),
            "penalty_edge_cover_repair": float(penalty.get("edge_cover_repair", 0.0)),
            "penalty_latency_guard_trigger": float(penalty.get("latency_guard_trigger", 0.0)),
            "penalty_feedback_timeout": float(penalty.get("feedback_timeout", 0.0)),
            "reward_dep_hotspot_weight": float(hotspot.get("hotspot_weight", 0.0)),
            "max_edge_replicas_per_device": max_edge_replicas,
            "edge_memory_budget_ratio": edge_memory_budget_ratio,
            "queue_normalizer": queue_normalizer,
            "select_threshold": _matrix_float("select_threshold", 0.55, probability=True),
            "negative_queue_threshold": _matrix_float("negative_queue_threshold", 0.65, probability=True),
            "negative_hotspot_threshold": _matrix_float("negative_hotspot_threshold", 0.08, probability=True),
            "negative_runtime_risk_threshold": _matrix_float(
                "negative_runtime_risk_threshold",
                0.50,
                probability=True,
            ),
            "negative_unknown_threshold": _matrix_float("negative_unknown_threshold", 0.50, probability=True),
            "negative_stale_threshold": _matrix_float("negative_stale_threshold", 0.85, probability=True),
            "positive_quality_threshold": _matrix_float("positive_quality_threshold", 0.30, probability=True),
            "pair_adjustment_scale": _matrix_float("pair_adjustment_scale", 0.80),
            "qk_weight": _matrix_float("qk_weight", 0.20),
            "quality_weight": _matrix_float("quality_weight", 0.90),
            "confidence_weight": _matrix_float("confidence_weight", 0.20),
            "service_pressure_weight": _matrix_float("service_pressure_weight", 0.10),
            "inertia_weight": _matrix_float("inertia_weight", 0.15),
            "unknown_penalty": _matrix_float("unknown_penalty", 0.90),
            "stale_penalty": _matrix_float("stale_penalty", 0.75),
            "runtime_risk_penalty": _matrix_float("runtime_risk_penalty", 0.60),
            "low_quality_penalty": _matrix_float("low_quality_penalty", 1.20),
            "queue_penalty": _matrix_float("queue_penalty", 0.25),
            "memory_penalty": _matrix_float("memory_penalty", 0.25),
            "device_load_penalty": _matrix_float("device_load_penalty", 0.10),
            "hotspot_penalty": _matrix_float("hotspot_penalty", 0.35),
            "ppo": ppo,
        }

    def _build_offloading_agent_params(self, agents_cfg: dict) -> dict:
        offloading = self._require_mapping(agents_cfg, "offloading")
        reward = self._require_mapping(offloading, "reward")
        ppo = self._build_ppo_update_cfg(offloading)
        scoring = offloading.get("scoring") or {}
        if not isinstance(scoring, dict):
            raise ValueError("Hedger agents.offloading.scoring must be a mapping when provided.")
        unknown_exploration = offloading.get("unknown_exploration") or {}
        if not isinstance(unknown_exploration, dict):
            raise ValueError("Hedger agents.offloading.unknown_exploration must be a mapping when provided.")
        unknown_exploration_enabled = bool(unknown_exploration.get("enabled", False))
        unknown_exploration_prob = (
            float(unknown_exploration.get("prob", 0.0))
            if unknown_exploration_enabled else 0.0
        )
        risk_exploration = offloading.get("risk_exploration") or {}
        if not isinstance(risk_exploration, dict):
            raise ValueError("Hedger agents.offloading.risk_exploration must be a mapping when provided.")
        risk_exploration_enabled = bool(risk_exploration.get("enabled", False))
        risk_exploration_prob = (
            float(risk_exploration.get("prob", 0.0))
            if risk_exploration_enabled else 0.0
        )
        off_latency_cfg = self._parse_latency_reward_cfg(
            reward,
            scope="offloading",
            default_clip=6.0,
        )
        return {
            "actor_lr": float(offloading["actor_lr"]),
            "critic_lr": float(offloading["critic_lr"]),
            "gamma": float(offloading["gamma"]),
            "lamda": float(offloading["lamda"]),
            "clip_eps": float(offloading["clip_eps"]),
            "update_encoder": bool(offloading.get("update_encoder", True)),
            "unknown_exploration_prob": max(0.0, min(1.0, unknown_exploration_prob)),
            "risk_exploration_prob": max(0.0, min(1.0, risk_exploration_prob)),
            "risk_exploration_temperature": float(risk_exploration.get("temperature", 0.25)),
            "risk_exploration_min_gap": float(risk_exploration.get("min_risk_gap", 0.05)),
            "reward_off_latency_weight": float(reward["latency_weight"]),
            "reward_off_slo_weight": float(reward["slo_weight"]),
            "reward_off_cloud_weight": float(reward["cloud_weight"]),
            "reward_off_projection_weight": float(reward.get("projection_weight", 0.0)),
            "reward_off_queue_weight": float(reward.get("queue_weight", 0.0)),
            "reward_off_queue_clip": float(reward.get("queue_clip", 3.0)),
            "reward_off_latency_transform": off_latency_cfg["transform"],
            "reward_off_latency_normalizer": off_latency_cfg["normalizer"],
            "reward_off_latency_clip": off_latency_cfg["clip"],
            "score_static_prior_scale": float(scoring.get("static_prior_scale", 0.45)),
            "score_runtime_weight": float(scoring.get("runtime_weight", 0.35)),
            "score_runtime_clip": float(scoring.get("runtime_clip", 3.0)),
            "score_absolute_queue_weight": float(scoring.get("absolute_queue_weight", 0.35)),
            "score_relative_queue_weight": float(scoring.get("relative_queue_weight", 1.8)),
            "score_overload_weight": float(scoring.get("overload_weight", 3.0)),
            "score_planned_load_weight": float(scoring.get("planned_load_weight", 0.8)),
            "score_relative_planned_load_weight": float(scoring.get("relative_planned_load_weight", 0.8)),
            "score_offered_load_weight": float(scoring.get("offered_load_weight", 0.45)),
            "score_offered_load_clip": float(scoring.get("offered_load_clip", 1.0)),
            "score_weak_replica_weight": float(scoring.get("weak_replica_weight", 1.2)),
            "score_weak_gap_clip": float(scoring.get("weak_gap_clip", 1.0)),
            "score_runtime_weak_gap_clip": float(scoring.get("runtime_weak_gap_clip", 0.35)),
            "score_runtime_weakness_min_confidence": float(scoring.get("runtime_weakness_min_confidence", 0.2)),
            "score_runtime_recency_floor": float(scoring.get("runtime_recency_floor", 0.7)),
            "score_weak_service_time_weight": float(scoring.get("weak_service_time_weight", 0.5)),
            "score_weak_capacity_weight": float(scoring.get("weak_capacity_weight", 0.35)),
            "score_weak_queue_amplifier": float(scoring.get("weak_queue_amplifier", 1.0)),
            "score_cloud_fallback_penalty": float(scoring.get("cloud_fallback_penalty", 1.2)),
            "score_cross_tier_weight": float(scoring.get("cross_tier_weight", 0.2)),
            "score_planned_load_clip": float(scoring.get("planned_load_clip", 3.0)),
            "score_load_clip": float(scoring.get("load_clip", 3.0)),
            "score_risk_clip": float(scoring.get("risk_clip", 3.0)),
            "ppo": ppo,
        }

    def _parse_latency_reward_cfg(
            self,
            reward_cfg: dict,
            *,
            scope: str,
            default_clip: Optional[float],
    ) -> Dict[str, Optional[float]]:
        latency_transform = str(reward_cfg.get("latency_transform", "clipped_ratio")).strip().lower()
        if latency_transform not in {"raw", "clipped_ratio", "log_ratio"}:
            raise ValueError(
                f"Hedger {scope}.reward.latency_transform must be one of: raw, clipped_ratio, log_ratio."
            )
        latency_normalizer = reward_cfg.get("latency_normalizer")
        if latency_normalizer is None:
            latency_normalizer = self.state_cfg.latency_slo if self.state_cfg.latency_slo is not None else 1.0
        latency_clip = reward_cfg.get("latency_clip", default_clip)
        return {
            "transform": latency_transform,
            "normalizer": max(1e-6, float(latency_normalizer)),
            "clip": None if latency_clip is None else max(0.0, float(latency_clip)),
        }

    def _build_ppo_update_cfg(self, agent_cfg: dict) -> HedgerPPOUpdateCfg:
        ppo = agent_cfg.get("ppo") or {}
        if not isinstance(ppo, dict):
            raise ValueError("Hedger agent ppo section must be a mapping when provided.")
        entropy_coef = max(0.0, float(ppo.get("entropy_coef", 0.003)))
        entropy_coef_final = max(0.0, float(ppo.get("entropy_coef_final", entropy_coef)))
        default_decay = self.training_cfg.total_updates if self.training_cfg is not None else 0
        entropy_decay_updates = max(0, int(ppo.get("entropy_decay_updates", default_decay)))
        value_coef = max(0.0, float(ppo.get("value_coef", 0.5)))
        return HedgerPPOUpdateCfg(
            entropy_coef=entropy_coef,
            entropy_coef_final=entropy_coef_final,
            entropy_decay_updates=entropy_decay_updates,
            value_coef=value_coef,
        )

    @staticmethod
    def _scheduled_entropy_coef(ppo_cfg: HedgerPPOUpdateCfg, update_step: int) -> float:
        if ppo_cfg.entropy_decay_updates <= 0:
            return float(ppo_cfg.entropy_coef)
        progress = min(1.0, max(0.0, float(update_step) / float(ppo_cfg.entropy_decay_updates)))
        return float(
            ppo_cfg.entropy_coef
            + (ppo_cfg.entropy_coef_final - ppo_cfg.entropy_coef) * progress
        )

    def _sync_agent_topology_bindings(self):
        """Synchronize agent-side cloud indices with the registered physical topology."""
        if self.physical_topology is None:
            return
        if self.deployment_agent is not None:
            self.deployment_agent.cloud_idx = self.physical_topology.cloud_idx
        if self.offloading_agent is not None:
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

    def _empty_latency_guard_stats(self) -> Dict[str, float]:
        return {
            "active": False,
            "trigger_seq": 0,
            "count": 0,
            "bad_count": 0,
            "bad_ratio": 0.0,
            "threshold_s": float(getattr(self, "latency_guard_cfg", HedgerLatencyGuardCfg()).latency_threshold_s),
            "trigger_streak": 0,
            "recover_streak": 0,
            "queue_recover_streak": 0,
            "max_queue_length": 0.0,
            "queue_observed": False,
            "queue_drained_s": 0.0,
        }

    def _latency_guard_enabled(self) -> bool:
        cfg = getattr(self, "latency_guard_cfg", HedgerLatencyGuardCfg())
        return (
            getattr(self, "mode", None) in {"train", "inference"}
            and cfg.enabled
            and cfg.window_size > 0
        )

    def _compute_latency_guard_stats_locked(self) -> Dict[str, float]:
        cfg = self.latency_guard_cfg
        samples = [float(value) for value in self._latency_guard_samples]
        count = len(samples)
        if count <= 0:
            return self._empty_latency_guard_stats()
        bad_count = sum(1 for value in samples if value > cfg.latency_threshold_s)
        bad_ratio = float(bad_count / max(1, count))
        queue_stats = self._latency_guard_queue_stats_locked(time.monotonic())
        return {
            "active": bool(self._latency_guard_active),
            "trigger_seq": int(getattr(self, "_latency_guard_trigger_seq", 0)),
            "count": int(count),
            "bad_count": int(bad_count),
            "bad_ratio": bad_ratio,
            "threshold_s": float(cfg.latency_threshold_s),
            "trigger_streak": int(self._latency_guard_trigger_streak),
            "recover_streak": int(self._latency_guard_recover_streak),
            "queue_recover_streak": int(getattr(self, "_latency_guard_queue_recover_streak", 0)),
            "max_queue_length": float(queue_stats["max_queue_length"]),
            "queue_observed": bool(queue_stats["queue_observed"]),
            "queue_drained_s": float(queue_stats["queue_drained_s"]),
        }

    def latency_guard_status(self) -> Dict[str, float]:
        if not self._latency_guard_enabled():
            return self._empty_latency_guard_stats()
        with self._latency_guard_lock:
            return copy.deepcopy(self._latency_guard_last_stats)

    def _record_latency_guard_trigger_event_locked(self, task_version: Optional[int],
                                                   deployment_version: Optional[int],
                                                   stats: Dict[str, Any]) -> None:
        self._latency_guard_trigger_seq = int(getattr(self, "_latency_guard_trigger_seq", 0)) + 1
        self._latency_guard_trigger_task_version = (
            int(task_version) if task_version is not None else None
        )
        self._latency_guard_trigger_deployment_version = (
            int(deployment_version) if deployment_version is not None else None
        )
        event_stats = copy.deepcopy(stats)
        event_stats["trigger_seq"] = int(self._latency_guard_trigger_seq)
        event_stats["trigger_task_version"] = self._latency_guard_trigger_task_version
        event_stats["trigger_deployment_version"] = self._latency_guard_trigger_deployment_version
        self._latency_guard_trigger_stats = event_stats

    def _latency_guard_trigger_event_for_deployment(self, deployment_version: Optional[int]) -> Optional[dict]:
        if not self._latency_guard_enabled() or deployment_version is None:
            return None
        try:
            target_version = int(deployment_version)
        except (TypeError, ValueError):
            return None
        with self._latency_guard_lock:
            trigger_version = getattr(self, "_latency_guard_trigger_deployment_version", None)
            if trigger_version is None or int(trigger_version) != target_version:
                return None
            seq = int(getattr(self, "_latency_guard_trigger_seq", 0))
            if seq <= 0:
                return None
            return {
                "seq": seq,
                "deployment_version": int(trigger_version),
                "task_version": getattr(self, "_latency_guard_trigger_task_version", None),
                "stats": copy.deepcopy(getattr(self, "_latency_guard_trigger_stats", {}) or {}),
            }

    def is_latency_guard_active(self) -> bool:
        if not self._latency_guard_enabled():
            return False
        with self._latency_guard_lock:
            return bool(self._latency_guard_active)

    def should_force_default_decisions(self) -> bool:
        cfg = getattr(self, "latency_guard_cfg", HedgerLatencyGuardCfg())
        return bool(cfg.force_default_decisions and self.is_latency_guard_active())

    def should_pause_generation(self) -> bool:
        cfg = getattr(self, "latency_guard_cfg", HedgerLatencyGuardCfg())
        return bool(cfg.pause_generation and self.is_latency_guard_active())

    def _latency_guard_trigger_seq_value(self) -> int:
        if not self._latency_guard_enabled():
            return 0
        with self._latency_guard_lock:
            return int(getattr(self, "_latency_guard_trigger_seq", 0))

    def _notify_inference_latency_guard_trigger(self) -> None:
        if getattr(self, "mode", None) != "inference":
            return
        self._latency_guard_trigger_event.set()
        with self._deployment_version_cond:
            self._deployment_version_cond.notify_all()

    @staticmethod
    def _task_total_start_wall_time(task) -> Optional[float]:
        try:
            tmp_data = task.get_tmp_data()
        except Exception:
            return None
        if not isinstance(tmp_data, dict):
            return None
        values = []
        for key, value in tmp_data.items():
            if not str(key).endswith(":total_start_time"):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return min(values) if values else None

    def should_quarantine_task_feedback(self, task) -> bool:
        cfg = getattr(self, "latency_guard_cfg", HedgerLatencyGuardCfg())
        if getattr(self, "mode", None) != "train":
            return False
        if not self._latency_guard_enabled() or not cfg.task_feedback_quarantine_enabled:
            return False

        with self._latency_guard_lock:
            active = bool(self._latency_guard_active)
            quarantine_until_t = float(getattr(self, "_latency_guard_feedback_quarantine_until_t", 0.0))
            recovered_wall_t = float(getattr(self, "_latency_guard_recovered_wall_t", 0.0))
            activated_wall_t = float(getattr(self, "_latency_guard_activated_wall_t", 0.0))

        if active:
            return True
        if time.monotonic() > quarantine_until_t:
            return False

        cutoff_wall_t = recovered_wall_t if recovered_wall_t > 0.0 else activated_wall_t
        task_start_wall_t = self._task_total_start_wall_time(task)
        if task_start_wall_t is None or cutoff_wall_t <= 0.0:
            return True
        return task_start_wall_t <= cutoff_wall_t

    def _latency_guard_queue_flush_devices(self) -> List[str]:
        cfg = self.latency_guard_cfg
        if cfg.queue_flush_scope == "all_devices" and self.physical_topology is not None:
            return list(dict.fromkeys(str(device) for device in self.physical_topology.nodes))

        now = time.monotonic()
        with self._latency_guard_lock:
            observations = copy.deepcopy(getattr(self, "_latency_guard_queue_observations", {}))

        devices = []
        for device, record in observations.items():
            if not isinstance(record, dict):
                continue
            try:
                ts = float(record.get("ts", 0.0))
            except (TypeError, ValueError):
                continue
            if now - ts > cfg.queue_recovery_stale_timeout_s:
                continue
            values = record.get("values") or {}
            if not isinstance(values, dict):
                continue
            max_queue = 0.0
            for value in values.values():
                try:
                    max_queue = max(max_queue, float(value))
                except (TypeError, ValueError):
                    continue
            if max_queue > cfg.queue_flush_threshold:
                devices.append(str(device))
        return list(dict.fromkeys(devices))

    def _flush_processor_queues_for_latency_guard(self, reason: str) -> None:
        cfg = self.latency_guard_cfg
        if not (cfg.queue_flush_enabled and cfg.queue_flush_on_trigger):
            return

        now = time.monotonic()
        last_flush_t = float(getattr(self, "_latency_guard_last_queue_flush_t", 0.0))
        if last_flush_t > 0.0 and now - last_flush_t < cfg.queue_flush_min_interval_s:
            LOGGER.warning(
                f"[Hedger][LatencyGuard] Skip processor queue flush: reason={reason}, "
                f"elapsed={self._format_log_value(now - last_flush_t, 2)}s, "
                f"min_interval={self._format_log_value(cfg.queue_flush_min_interval_s, 2)}s."
            )
            return

        devices = self._latency_guard_queue_flush_devices()
        if not devices:
            LOGGER.warning(
                f"[Hedger][LatencyGuard] No processor queue flush command target found: "
                f"reason={reason}, scope={cfg.queue_flush_scope}."
            )
            return

        self._latency_guard_last_queue_flush_t = now
        self._latency_guard_queue_flush_command_seq += 1
        command_id = (
            f"latency_guard_queue_flush_"
            f"{int(getattr(self, '_latency_guard_activated_wall_t', time.time()) * 1000)}_"
            f"{self._latency_guard_queue_flush_command_seq}"
        )
        request = {
            "reason": reason,
            "max_count": cfg.queue_flush_max_count_per_service,
            "dry_run": cfg.queue_flush_dry_run,
            "timeout_s": cfg.queue_flush_timeout_s,
        }
        command = {
            "type": "clear_processor_queues",
            "command_id": command_id,
            "target_devices": devices,
            "request": request,
        }

        with self._latency_guard_lock:
            self._latency_guard_queue_flush_command = command

        LOGGER.warning(
            f"[Hedger][LatencyGuard] Published processor queue flush command: "
            f"command_id={command_id}, reason={reason}, devices={devices}, "
            f"dry_run={cfg.queue_flush_dry_run}."
        )

    @staticmethod
    def _normalize_completed_action_targets(info: Optional[dict]) -> set:
        if not isinstance(info, dict):
            return set()
        completed = info.get("completed_action_targets") or []
        if isinstance(completed, str):
            completed = [completed]
        if not isinstance(completed, (list, tuple, set)):
            return set()
        return {str(item) for item in completed}

    @staticmethod
    def _filter_completed_action_targets(command: dict, completed_targets: set) -> Optional[dict]:
        if not completed_targets:
            return command
        command_id = str(command.get("command_id") or "")
        target_devices = command.get("target_devices") or command.get("devices") or []
        if isinstance(target_devices, str):
            target_devices = [target_devices]
        if not isinstance(target_devices, (list, tuple, set)):
            return command

        pending_devices = []
        for device in target_devices:
            device = str(device)
            keys = {device, f"{command_id}:{device}"} if command_id else {device}
            if keys.intersection(completed_targets):
                continue
            pending_devices.append(device)
        if not pending_devices:
            return None

        command = copy.deepcopy(command)
        command["target_devices"] = pending_devices
        return command

    def get_generation_admission_actions(self, info: Optional[dict] = None) -> List[dict]:
        if not self._latency_guard_enabled():
            return []
        with self._latency_guard_lock:
            active = bool(self._latency_guard_active)
            command = copy.deepcopy(getattr(self, "_latency_guard_queue_flush_command", None))
        if not active or not command:
            return []
        command = self._filter_completed_action_targets(
            command,
            self._normalize_completed_action_targets(info),
        )
        return [command] if command else []

    def _clear_latency_guard_window_after_recovery(self, reason: str) -> None:
        if not self.latency_guard_cfg.clear_latency_window_on_recovery:
            return
        with self._latency_guard_lock:
            sample_count = len(self._latency_guard_samples)
            self._latency_guard_samples.clear()
            self._latency_guard_last_stats = self._compute_latency_guard_stats_locked()
        if sample_count > 0:
            LOGGER.warning(
                f"[Hedger][LatencyGuard] Cleared latency window after recovery: "
                f"reason={reason}, samples={sample_count}."
            )

    def _latency_guard_queue_stats_locked(self, now: Optional[float] = None) -> Dict[str, float]:
        cfg = self.latency_guard_cfg
        now = time.monotonic() if now is None else float(now)
        max_queue_length = 0.0
        observed = False
        observations = getattr(self, "_latency_guard_queue_observations", {})
        for record in observations.values():
            ts = float(record.get("ts", 0.0))
            if now - ts > cfg.queue_recovery_stale_timeout_s:
                continue
            values = record.get("values") or {}
            if not isinstance(values, dict):
                continue
            for value in values.values():
                try:
                    queue_length = float(value)
                except (TypeError, ValueError):
                    continue
                max_queue_length = max(max_queue_length, queue_length)
                observed = True
        drained_since_t = getattr(self, "_latency_guard_queue_drained_since_t", 0.0)
        queue_drained_s = max(0.0, now - drained_since_t) if drained_since_t > 0.0 else 0.0
        return {
            "max_queue_length": float(max_queue_length),
            "queue_observed": bool(observed),
            "queue_drained_s": float(queue_drained_s),
        }

    def observe_task_end_to_end_latency(
            self,
            latency: float,
            task_version: Optional[int] = None,
            deployment_version: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Feed completed-task end-to-end latency into a circuit-breaker style guard.

        The guard watches a sliding window of recent completed tasks. If too
        many samples breach the configured latency threshold for several
        consecutive evaluations, PPO sampling/updating is paused and scheduler
        policies are forced back to the template defaults until the window has
        recovered for several consecutive evaluations.
        """
        if not self._latency_guard_enabled():
            return self._empty_latency_guard_stats()

        try:
            latency_value = float(latency)
        except (TypeError, ValueError):
            return self.latency_guard_status()
        if not math.isfinite(latency_value) or latency_value < 0.0:
            return self.latency_guard_status()

        state_change = None
        with self._latency_guard_lock:
            self._latency_guard_samples.append(latency_value)
            cfg = self.latency_guard_cfg
            stats = self._compute_latency_guard_stats_locked()

            if stats["count"] >= cfg.min_samples:
                if self._latency_guard_active:
                    if stats["bad_ratio"] <= cfg.recover_violation_ratio:
                        self._latency_guard_recover_streak += 1
                    else:
                        self._latency_guard_recover_streak = 0
                    self._latency_guard_trigger_streak = 0

                    if self._latency_guard_recover_streak >= cfg.recover_consecutive_windows:
                        self._latency_guard_active = False
                        self._latency_guard_recover_streak = 0
                        self._latency_guard_trigger_streak = 0
                        self._latency_guard_recovered_wall_t = time.time()
                        self._latency_guard_feedback_quarantine_until_t = (
                            time.monotonic() + cfg.task_feedback_quarantine_s
                        )
                        state_change = "recovered"
                else:
                    if stats["bad_ratio"] >= cfg.trigger_violation_ratio:
                        self._latency_guard_trigger_streak += 1
                    else:
                        self._latency_guard_trigger_streak = 0
                    self._latency_guard_recover_streak = 0

                    if self._latency_guard_trigger_streak >= cfg.trigger_consecutive_windows:
                        self._latency_guard_active = True
                        self._latency_guard_activated_t = time.monotonic()
                        self._latency_guard_activated_wall_t = time.time()
                        self._latency_guard_trigger_streak = 0
                        self._latency_guard_recover_streak = 0
                        self._latency_guard_queue_recover_streak = 0
                        self._latency_guard_queue_drained_since_t = 0.0
                        state_change = "triggered"

            self._latency_guard_last_stats = self._compute_latency_guard_stats_locked()
            stats = copy.deepcopy(self._latency_guard_last_stats)
            if state_change == "triggered":
                self._record_latency_guard_trigger_event_locked(task_version, deployment_version, stats)

        if state_change == "triggered":
            self._handle_latency_guard_triggered(stats, task_version, deployment_version)
        elif state_change == "recovered":
            self._handle_latency_guard_recovered(stats, task_version, deployment_version, reason="latency_window")
        else:
            self._log_latency_guard_if_active(stats)
        return stats

    def update_latency_guard_queue_lengths(self, device: str, queue_lengths) -> Dict[str, float]:
        """
        Feed processor queue lengths into the latency guard.

        This is used as the preferred recovery signal when generation has been
        paused. Once monitored queues stay below the low watermark for a few
        consecutive monitor updates, the guard can recover even if no new task
        completions arrive.
        """
        guard_queue_enabled = self._latency_guard_enabled() and self.latency_guard_cfg.queue_recovery_enabled
        deployment_event_enabled = self._deployment_event_trigger_cfg_enabled()
        if not guard_queue_enabled and not deployment_event_enabled:
            return self._empty_latency_guard_stats()
        if not isinstance(queue_lengths, dict):
            return self.latency_guard_status() if guard_queue_enabled else self._empty_latency_guard_stats()

        normalized = {}
        for service_name, value in queue_lengths.items():
            try:
                normalized[str(service_name)] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        if not normalized:
            return self.latency_guard_status() if guard_queue_enabled else self._empty_latency_guard_stats()

        recovered = False
        now = time.monotonic()
        with self._latency_guard_lock:
            self._latency_guard_queue_observations[str(device)] = {
                "values": normalized,
                "ts": now,
            }
            if guard_queue_enabled and self._latency_guard_active:
                queue_stats = self._latency_guard_queue_stats_locked(now)
                activated_t = getattr(self, "_latency_guard_activated_t", 0.0)
                pause_elapsed = (
                    activated_t <= 0.0
                    or now - activated_t >= self.latency_guard_cfg.min_pause_s
                )
                if (
                        pause_elapsed
                        and queue_stats["queue_observed"]
                        and queue_stats["max_queue_length"] <= self.latency_guard_cfg.queue_recovery_threshold
                ):
                    if getattr(self, "_latency_guard_queue_drained_since_t", 0.0) <= 0.0:
                        self._latency_guard_queue_drained_since_t = now
                    self._latency_guard_queue_recover_streak = (
                        getattr(self, "_latency_guard_queue_recover_streak", 0) + 1
                    )
                else:
                    self._latency_guard_queue_drained_since_t = 0.0
                    self._latency_guard_queue_recover_streak = 0

                queue_stats = self._latency_guard_queue_stats_locked(now)
                if (
                        self._latency_guard_queue_recover_streak >=
                        self.latency_guard_cfg.queue_recovery_consecutive_updates
                        and queue_stats["queue_drained_s"] >= self.latency_guard_cfg.queue_recovery_stable_s
                ):
                    self._latency_guard_active = False
                    self._latency_guard_trigger_streak = 0
                    self._latency_guard_recover_streak = 0
                    self._latency_guard_queue_recover_streak = 0
                    self._latency_guard_queue_drained_since_t = 0.0
                    self._latency_guard_recovered_wall_t = time.time()
                    self._latency_guard_feedback_quarantine_until_t = (
                        time.monotonic() + self.latency_guard_cfg.task_feedback_quarantine_s
                    )
                    recovered = True
            if guard_queue_enabled:
                self._latency_guard_last_stats = self._compute_latency_guard_stats_locked()
                stats = copy.deepcopy(self._latency_guard_last_stats)
            else:
                stats = self._empty_latency_guard_stats()

        if guard_queue_enabled:
            if recovered:
                self._handle_latency_guard_recovered(
                    stats,
                    task_version=None,
                    deployment_version=None,
                    reason="queue_drain",
                )
            else:
                self._log_latency_guard_if_active(stats)
        if deployment_event_enabled:
            self._deployment_event_trigger_event.set()
        return stats

    def _deployment_event_trigger_cfg_enabled(self) -> bool:
        cfg = getattr(self, "deployment_event_trigger_cfg", HedgerDeploymentEventTriggerCfg())
        return bool(cfg.enabled)

    @staticmethod
    def _deployment_event_suppression_defaults() -> Dict[str, Any]:
        return {
            "deployment_event_suppressed": False,
            "deployment_event_suppress_reason": "",
            "deployment_event_suppressed_event_reason": "",
            "deployment_event_warmup_remaining_s": 0.0,
            "deployment_event_feedback_count": 0,
            "deployment_event_feedback_required": 0,
        }

    def _record_deployment_event_suppressed(self, status: Dict[str, Any]) -> None:
        self._last_deployment_event_suppressed_record = copy.deepcopy(status)

    def _consume_deployment_event_suppressed_record(self) -> Dict[str, Any]:
        record = copy.deepcopy(getattr(self, "_last_deployment_event_suppressed_record", {}) or {})
        self._last_deployment_event_suppressed_record = {}
        return record

    def _deployment_event_wait_status(self, reason: str) -> Dict[str, Any]:
        status = {
            "triggered": False,
            "reason": reason,
            "queue_pressure": 0.0,
            "hotspot_pressure": 0.0,
            "max_queue": 0.0,
            "hotspot_service": "",
            "hotspot_device": "",
            **self._deployment_event_suppression_defaults(),
        }
        suppressed = self._consume_deployment_event_suppressed_record()
        if suppressed:
            status.update(suppressed)
            status["triggered"] = False
            status["reason"] = reason
        return status

    def _deployment_event_trigger_status(self) -> Dict[str, Any]:
        cfg = getattr(self, "deployment_event_trigger_cfg", HedgerDeploymentEventTriggerCfg())
        if not cfg.enabled:
            return {"triggered": False, "reason": ""}
        now = time.monotonic()
        if now - float(getattr(self, "_last_deployment_event_trigger_monotonic", 0.0)) < cfg.cooldown_s:
            return {"triggered": False, "reason": "cooldown"}
        last_served_t = float(getattr(self, "_last_deployment_served_monotonic", 0.0))
        if last_served_t > 0.0 and now - last_served_t < cfg.min_interval_s:
            return {"triggered": False, "reason": "min_interval"}

        queue_normalizer = max(1e-6, float(self.deployment_agent_params.get("queue_normalizer", 8.0)))
        with self._latency_guard_lock:
            observations = copy.deepcopy(getattr(self, "_latency_guard_queue_observations", {}))

        max_queue = 0.0
        hotspot_pressure = 0.0
        hotspot_service = ""
        hotspot_device = ""
        with self._data_lock:
            deploy_mask = self.cur_deploy_mask.detach().clone().cpu() if self.cur_deploy_mask is not None else None

        for device_name, record in observations.items():
            if not isinstance(record, dict):
                continue
            try:
                ts = float(record.get("ts", 0.0))
            except (TypeError, ValueError):
                continue
            if now - ts > self.latency_guard_cfg.queue_recovery_stale_timeout_s:
                continue
            values = record.get("values") or {}
            if not isinstance(values, dict):
                continue
            for service_name, value in values.items():
                try:
                    queue_length = max(0.0, float(value))
                except (TypeError, ValueError):
                    continue
                max_queue = max(max_queue, queue_length)
                if deploy_mask is None or self.logical_topology is None or self.physical_topology is None:
                    continue
                try:
                    service_idx = self.logical_topology.index(str(service_name))
                    device_idx = self.physical_topology.index(str(device_name))
                except ValueError:
                    continue
                cloud_idx = self.physical_topology.cloud_idx
                if device_idx == cloud_idx or device_idx < 0 or device_idx >= deploy_mask.size(1):
                    continue
                if service_idx < 0 or service_idx >= deploy_mask.size(0):
                    continue
                if not bool(deploy_mask[service_idx, device_idx].item()):
                    continue
                pressure = queue_length / queue_normalizer
                if pressure > hotspot_pressure:
                    hotspot_pressure = pressure
                    hotspot_service = str(service_name)
                    hotspot_device = str(device_name)

        queue_pressure = max_queue / queue_normalizer
        status = {
            "triggered": False,
            "reason": "",
            "queue_pressure": float(queue_pressure),
            "hotspot_pressure": float(hotspot_pressure),
            "max_queue": float(max_queue),
            "hotspot_service": hotspot_service,
            "hotspot_device": hotspot_device,
            "e2e_slo_violation": 0.0,
            "e2e_p95": 0.0,
            "e2e_feedback_count": 0,
            **self._deployment_event_suppression_defaults(),
        }
        served_version = self.get_active_deployment_version()
        e2e_feedback_count = self._deployment_feedback_count(served_version)
        if e2e_feedback_count >= int(cfg.e2e_min_feedback_samples):
            try:
                latency_stats = self.state_buffer.get_task_end_to_end_latency_stats(
                    deployment_version=served_version,
                    last_k=self.state_cfg.deployment_reward_window,
                )
                latency_values = latency_stats.get("latencies", [])
                e2e_p95 = self._percentile(latency_values, 95)
                e2e_slo = self._compute_slo_violation(latency_values)
                status["e2e_slo_violation"] = float(e2e_slo)
                status["e2e_p95"] = float(e2e_p95)
                status["e2e_feedback_count"] = int(latency_stats.get("count", e2e_feedback_count) or 0)
                if e2e_slo >= cfg.e2e_slo_threshold:
                    status["triggered"] = True
                    status["reason"] = "event_e2e_slo"
                elif e2e_p95 >= cfg.e2e_p95_threshold_s:
                    status["triggered"] = True
                    status["reason"] = "event_e2e_p95"
            except Exception as exc:
                LOGGER.debug(f"[Hedger][DeploymentEvent] Failed to evaluate e2e trigger: {exc}")
        if hotspot_pressure >= cfg.hotspot_pressure_threshold:
            status["triggered"] = True
            status["reason"] = "event_pair_hotspot"
        elif queue_pressure >= cfg.queue_pressure_threshold:
            status["triggered"] = True
            status["reason"] = "event_queue_pressure"
        if not status["triggered"] or self.mode != "inference":
            return status

        feedback_required = max(0, int(cfg.inference_min_feedback_samples))
        feedback_count = e2e_feedback_count
        warmup_remaining_s = 0.0
        if last_served_t > 0.0 and cfg.inference_warmup_grace_s > 0.0:
            warmup_remaining_s = max(0.0, cfg.inference_warmup_grace_s - (now - last_served_t))

        suppress_reason = ""
        if warmup_remaining_s > 0.0:
            suppress_reason = "inference_warmup_grace"
        elif (
                cfg.inference_require_feedback_before_trigger
                and feedback_required > 0
                and feedback_count < feedback_required
        ):
            suppress_reason = "inference_feedback_shortfall"

        if suppress_reason:
            suppressed_status = copy.deepcopy(status)
            suppressed_status.update({
                "triggered": False,
                "reason": suppress_reason,
                "deployment_event_suppressed": True,
                "deployment_event_suppress_reason": suppress_reason,
                "deployment_event_suppressed_event_reason": str(status.get("reason", "") or ""),
                "deployment_event_warmup_remaining_s": float(warmup_remaining_s),
                "deployment_event_feedback_count": int(feedback_count),
                "deployment_event_feedback_required": int(feedback_required),
            })
            self._record_deployment_event_suppressed(suppressed_status)
            return suppressed_status
        return status

    def _mark_deployment_event_triggered(self, status: Dict[str, Any]) -> None:
        if not status.get("triggered"):
            return
        self._last_deployment_event_trigger_monotonic = time.monotonic()
        self._last_deployment_event_trigger_record = copy.deepcopy(status)
        self._last_deployment_event_suppressed_record = {}
        self._deployment_event_trigger_event.clear()

    def _handle_latency_guard_triggered(
            self,
            stats: Dict[str, float],
            task_version: Optional[int],
            deployment_version: Optional[int],
    ) -> None:
        if self.latency_guard_cfg.clear_transition_buffers:
            self._clear_training_transition_buffers("latency guard triggered")
        if self.latency_guard_cfg.force_default_decisions:
            self._apply_default_decisions_for_latency_guard("latency guard triggered")
        self._flush_processor_queues_for_latency_guard("latency_guard_triggered")
        self._notify_inference_latency_guard_trigger()
        LOGGER.warning(
            f"[Hedger][LatencyGuard] Triggered: task_version={task_version}, "
            f"deployment_version={deployment_version}, "
            f"bad={stats['bad_count']}/{stats['count']} "
            f"({self._format_log_value(stats['bad_ratio'])}), "
            f"threshold={self._format_log_value(stats['threshold_s'], 2)}s. "
            f"Pause generation/training and "
            f"{'force default decisions' if self.latency_guard_cfg.force_default_decisions else 'preserve deployment feedback'}."
        )

    def _handle_latency_guard_recovered(
            self,
            stats: Dict[str, float],
            task_version: Optional[int],
            deployment_version: Optional[int],
            reason: str = "latency_window",
    ) -> None:
        if self.latency_guard_cfg.clear_transition_buffers:
            self._clear_training_transition_buffers("latency guard recovered")
        LOGGER.warning(
            f"[Hedger][LatencyGuard] Recovered: task_version={task_version}, "
            f"deployment_version={deployment_version}, "
            f"reason={reason}, "
            f"bad={stats['bad_count']}/{stats['count']} "
            f"({self._format_log_value(stats['bad_ratio'])}), "
            f"max_queue={self._format_log_value(stats.get('max_queue_length', 0.0))}, "
            f"threshold={self._format_log_value(stats['threshold_s'], 2)}s. "
            f"Resume PPO sampling."
        )
        with self._latency_guard_lock:
            self._latency_guard_queue_flush_command = None
        self._clear_latency_guard_window_after_recovery(reason)

    def _log_latency_guard_if_active(self, stats: Dict[str, float]) -> None:
        if not stats.get("active"):
            return
        now = time.monotonic()
        if now - self._latency_guard_last_log_t < self.latency_guard_cfg.log_interval_s:
            return
        self._latency_guard_last_log_t = now
        LOGGER.warning(
            f"[Hedger][LatencyGuard] Active: bad={stats['bad_count']}/{stats['count']} "
            f"({self._format_log_value(stats['bad_ratio'])}), "
            f"threshold={self._format_log_value(stats['threshold_s'], 2)}s, "
            f"max_queue={self._format_log_value(stats.get('max_queue_length', 0.0))}, "
            f"queue_drained_s={self._format_log_value(stats.get('queue_drained_s', 0.0), 2)}, "
            f"recover_ratio={self._format_log_value(self.latency_guard_cfg.recover_violation_ratio)}, "
            f"recover_streak={stats['recover_streak']}/"
            f"{self.latency_guard_cfg.recover_consecutive_windows}, "
            f"queue_recover_streak={stats.get('queue_recover_streak', 0)}/"
            f"{self.latency_guard_cfg.queue_recovery_consecutive_updates}."
        )

    def _clear_training_transition_buffers(self, reason: str) -> None:
        with self._data_lock:
            preserved_dep = [
                tr for tr in self.deployment_transitions
                if bool(tr.get("feedback_guard_interrupted", False))
            ]
            dep_count = len(self.deployment_transitions) - len(preserved_dep)
            off_count = len(self.offloading_transitions)
            self.deployment_transitions[:] = preserved_dep
            self.offloading_transitions.clear()
        if dep_count > 0 or off_count > 0:
            LOGGER.warning(
                f"[Hedger][LatencyGuard] Cleared transition buffers while {reason}: "
                f"deployment={dep_count}, offloading={off_count}, "
                f"preserved_guard_deployment={len(preserved_dep)}."
            )

    def _apply_default_decisions_for_latency_guard(self, reason: str) -> None:
        with self._data_lock:
            self.offloading_plan = None
            self.pending_deployment_plan = None
            self.pending_deploy_mask = None
            self._pending_deployment_force_serve = False
            self._pending_deployment_reason = None

            if self.initial_deployment_plan is not None and self.logical_topology is not None \
                    and self.physical_topology is not None:
                self.deployment_plan = copy.deepcopy(self.initial_deployment_plan)
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.initial_deployment_plan
                ).detach().cpu()

        LOGGER.debug(f"[Hedger][LatencyGuard] Applied default decisions: reason={reason}.")

    def _sleep_while_latency_guard_active(self, worker_name: str) -> bool:
        if not self.is_latency_guard_active():
            return False
        if self.latency_guard_cfg.force_default_decisions:
            self._apply_default_decisions_for_latency_guard(f"{worker_name} paused")
        self._log_latency_guard_if_active(self.latency_guard_status())
        time.sleep(self.latency_guard_cfg.poll_interval_s)
        return True

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
        stage_name = getattr(training_cfg, "stage", "none")
        state_cfg = getattr(self, "state_cfg", None)
        latency_guard_cfg = getattr(self, "latency_guard_cfg", None)
        dep_seq_len = getattr(state_cfg, "deployment_seq_len", "na")
        off_seq_len = getattr(state_cfg, "offloading_seq_len", "na")
        dep_params = getattr(self, "deployment_agent_params", {}) or {}
        default_warmup_cfg = getattr(training_cfg, "deployment_default_warmup", None)
        return (
            f"mode={getattr(self, 'mode', 'unknown')}, train_stage={stage_name}, "
            f"device={getattr(self, 'device', 'unknown')}, seed={getattr(self, 'seed', 'na')}, "
            f"intervals(dep/off)={self._format_log_value(getattr(self, 'deployment_interval', None), 2)}/"
            f"{self._format_log_value(getattr(self, 'offloading_interval', None), 2)}s, "
            f"stage_step={getattr(self, '_epoch', 'na')}, "
            f"global_step={getattr(self, '_global_update_step', 'na')}, "
            f"rollout(dep/off)={getattr(training_cfg, 'deployment_rollout_len', 'na')}/"
            f"{getattr(training_cfg, 'offloading_rollout_len', 'na')}, "
            f"rollout_deterministic(dep/off)="
            f"{getattr(training_cfg, 'deployment_rollout_deterministic', 'na')}/"
            f"{getattr(training_cfg, 'offloading_rollout_deterministic', 'na')}, "
            f"batch(dep/off)={getattr(training_cfg, 'deployment_batch_size', 'na')}/"
            f"{getattr(training_cfg, 'offloading_batch_size', 'na')}, "
            f"state_seq(dep/off)={dep_seq_len}/{off_seq_len}, "
            f"total_steps={getattr(training_cfg, 'total_updates', 'na')}, "
            f"save_interval={getattr(checkpoint_save_cfg, 'interval_updates', 'na')}, "
            f"deployment_default_warmup={getattr(default_warmup_cfg, 'enabled', False)}, "
            f"deployment_default_warmup_intervals={getattr(default_warmup_cfg, 'min_intervals', 'na')}, "
            f"deployment_default_warmup_feedback={getattr(default_warmup_cfg, 'min_feedback_samples', 'na')}, "
            f"dep_latency_weight={dep_params.get('reward_dep_latency_weight', 'na')}, "
            f"dep_slo_weight={dep_params.get('reward_dep_slo_weight', 'na')}, "
            f"dep_cloud_only_weight={dep_params.get('reward_dep_cloud_only_weight', 'na')}, "
            f"edge_cover_repair_weight={dep_params.get('penalty_edge_cover_repair', 'na')}, "
            f"latency_guard_penalty_weight={dep_params.get('penalty_latency_guard_trigger', 'na')}, "
            f"feedback_timeout_penalty_weight={dep_params.get('penalty_feedback_timeout', 'na')}, "
            f"max_edge_replicas_per_device={dep_params.get('max_edge_replicas_per_device', 'na')}, "
            f"edge_memory_budget_ratio={dep_params.get('edge_memory_budget_ratio', 'na')}, "
            f"hotspot_weight={dep_params.get('reward_dep_hotspot_weight', 'na')}, "
            f"select_threshold={dep_params.get('select_threshold', 'na')}, "
            f"negative_runtime_risk_threshold={dep_params.get('negative_runtime_risk_threshold', 'na')}, "
            f"positive_quality_threshold={dep_params.get('positive_quality_threshold', 'na')}, "
            f"latency_guard={getattr(latency_guard_cfg, 'enabled', False)}"
        )

    def _summarize_topology(self) -> str:
        logical_services = len(self.logical_topology) if self.logical_topology is not None else 0
        logical_edges = len(self.logical_topology.links) if self.logical_topology is not None else 0
        physical_nodes = len(self.physical_topology) if self.physical_topology is not None else 0
        physical_edges = len(self.physical_topology.links) if self.physical_topology is not None else 0
        cloud_name = (
            self.physical_topology[self.physical_topology.cloud_idx]
            if self.physical_topology is not None and physical_nodes > 0
            else "unregistered"
        )
        return (
            f"logical_services={logical_services}, logical_edges={logical_edges}, "
            f"physical_nodes={physical_nodes}, physical_edges={physical_edges}, "
            f"cloud={cloud_name}"
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

        latest_arrival_rate = 0.0
        if "task_arrival_rate_seq" in logic_feats and logic_feats["task_arrival_rate_seq"].numel():
            latest_arrival_rate = float(logic_feats["task_arrival_rate_seq"][:, -1].mean().item())

        latest_cloud_bw = 0.0
        latest_edge_bw = 0.0
        if "bandwidth_latest" in phys_feats and phys_feats["bandwidth_latest"].numel():
            bandwidth_latest = phys_feats["bandwidth_latest"]
            cloud_idx = (
                self.physical_topology.cloud_idx
                if self.physical_topology is not None
                else bandwidth_latest.size(0) - 1
            )
            latest_cloud_bw = float(bandwidth_latest[cloud_idx].item())
            edge_bandwidth = bandwidth_latest
            if edge_bandwidth.numel() > 1:
                edge_bandwidth = torch.cat([edge_bandwidth[:cloud_idx], edge_bandwidth[cloud_idx + 1:]])
            else:
                edge_bandwidth = torch.empty(0, dtype=edge_bandwidth.dtype)
            latest_edge_bw = float(edge_bandwidth.mean().item()) if edge_bandwidth.numel() else 0.0

        base = (
            f"services={service_count}, devices={device_count}, "
            f"latest_complexity={self._format_log_value(latest_complexity)}, "
            f"latest_arrival_rate={self._format_log_value(latest_arrival_rate)}, "
            f"latest_edge_bw={self._format_log_value(latest_edge_bw)}, "
            f"latest_cloud_bw={self._format_log_value(latest_cloud_bw)}"
        )
        if metrics:
            base += f", metrics=({self._summarize_metrics(metrics)})"
        return base

    def _json_for_record(self, value) -> str:
        def _normalize(obj):
            if obj is None:
                return None
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            if isinstance(obj, dict):
                return {str(key): _normalize(val) for key, val in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_normalize(item) for item in obj]
            try:
                return float(obj) if hasattr(obj, "item") else obj
            except (TypeError, ValueError):
                return obj

        normalized = {} if value is None else _normalize(value)
        encoded = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        debug_enabled = (
                self.record_cfg.debug_mode
                or self.record_cfg.state_snapshot_debug
                or self.record_cfg.actor_snapshot_debug
                or self.record_cfg.decision_candidate_features_debug
                or self.record_cfg.decision_actor_debug
        )
        limit = (
            self.record_cfg.debug_json_max_chars
            if debug_enabled else self.record_cfg.normal_json_max_chars
        )
        if limit > 0 and len(encoded) > limit:
            preview = encoded[:limit]
            return json.dumps(
                {
                    "truncated": True,
                    "original_chars": len(encoded),
                    "preview": preview,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        return encoded

    @staticmethod
    def _nested_float_mean(value) -> float:
        values = []

        def _visit(obj):
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    _visit(item)
                return
            try:
                values.append(float(obj))
            except (TypeError, ValueError):
                return

        _visit(value)
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _pair_feature_map(
            self,
            logic_feats: Dict[str, torch.Tensor],
            key: str,
            service_idx: int,
            names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        value = logic_feats.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        value = value.detach().float().cpu()
        if value.dim() != 3 or service_idx < 0 or service_idx >= value.size(0):
            return {}
        result = {}
        for device_idx in range(value.size(1)):
            row = [float(item) for item in value[service_idx, device_idx].tolist()]
            if names and len(names) == len(row):
                result[self._device_name(device_idx)] = dict(zip(names, row))
            else:
                result[self._device_name(device_idx)] = row
        return result

    @staticmethod
    def _service_feature_vector(
            logic_feats: Dict[str, torch.Tensor],
            key: str,
            service_idx: int,
            names: Optional[List[str]] = None,
    ) -> Any:
        value = logic_feats.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return []
        value = value.detach().float().cpu()
        if value.dim() != 2 or service_idx < 0 or service_idx >= value.size(0):
            return []
        row = [float(item) for item in value[service_idx].tolist()]
        if names and len(names) == len(row):
            return dict(zip(names, row))
        return row

    def _device_feature_map(
            self,
            phys_feats: Dict[str, torch.Tensor],
            key: str,
            names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        value = phys_feats.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        value = value.detach().float().cpu()
        if value.dim() != 2:
            return {}
        result = {}
        for device_idx in range(value.size(0)):
            row = [float(item) for item in value[device_idx].tolist()]
            if names and len(names) == len(row):
                result[self._device_name(device_idx)] = dict(zip(names, row))
            else:
                result[self._device_name(device_idx)] = row
        return result

    def _actor_debug_row_map(
            self,
            actor_debug: Optional[Dict[str, Any]],
            key: str,
            service_idx: int,
    ) -> Dict[str, Any]:
        if not isinstance(actor_debug, dict):
            return {}
        value = actor_debug.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        value = value.detach().float().cpu()
        if value.dim() != 2 or service_idx < 0 or service_idx >= value.size(0):
            return {}
        return {
            self._device_name(device_idx): float(value[service_idx, device_idx].item())
            for device_idx in range(value.size(1))
        }

    @staticmethod
    def _actor_debug_vector_value(
            actor_debug: Optional[Dict[str, Any]],
            key: str,
            service_idx: int,
    ) -> float:
        if not isinstance(actor_debug, dict):
            return 0.0
        value = actor_debug.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        value = value.detach().float().cpu()
        if value.dim() != 1 or service_idx < 0 or service_idx >= value.size(0):
            return 0.0
        return float(value[service_idx].item())

    def _actor_debug_device_vector_map(
            self,
            actor_debug: Optional[Dict[str, Any]],
            key: str,
    ) -> Dict[str, float]:
        if not isinstance(actor_debug, dict):
            return {}
        value = actor_debug.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        value = value.detach().float().cpu()
        if value.dim() != 1:
            return {}
        return {
            self._device_name(device_idx): float(value[device_idx].item())
            for device_idx in range(value.size(0))
        }

    @staticmethod
    def _actor_debug_matrix_value(
            actor_debug: Optional[Dict[str, Any]],
            key: str,
            service_idx: int,
            device_idx: int,
    ) -> float:
        if not isinstance(actor_debug, dict):
            return 0.0
        value = actor_debug.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        value = value.detach().float().cpu()
        if value.dim() != 2 or service_idx < 0 or service_idx >= value.size(0):
            return 0.0
        if device_idx < 0 or device_idx >= value.size(1):
            return 0.0
        return float(value[service_idx, device_idx].item())

    def _actor_debug_pair_feature_map(
            self,
            actor_debug: Optional[Dict[str, Any]],
            key: str,
            service_idx: int,
    ) -> Dict[str, List[float]]:
        if not isinstance(actor_debug, dict):
            return {}
        value = actor_debug.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        value = value.detach().float().cpu()
        if value.dim() != 3 or service_idx < 0 or service_idx >= value.size(0):
            return {}
        names = actor_debug.get(f"{key}_names")
        if key == "candidate_feature":
            names = actor_debug.get("candidate_feature_names", names)
        elif key == "runtime_pair_feature":
            names = actor_debug.get("runtime_pair_feature_names", names)
        result = {}
        for device_idx in range(value.size(1)):
            row = [float(item) for item in value[service_idx, device_idx].tolist()]
            if isinstance(names, list) and len(names) == len(row):
                result[self._device_name(device_idx)] = dict(zip(names, row))
            else:
                result[self._device_name(device_idx)] = row
        return result

    @staticmethod
    def _actor_debug_candidate_value(
            actor_debug: Optional[Dict[str, Any]],
            service_idx: int,
            device_idx: int,
            feature_name: str,
    ) -> float:
        if not isinstance(actor_debug, dict):
            return 0.0
        value = actor_debug.get("candidate_feature")
        names = actor_debug.get("candidate_feature_names")
        if not isinstance(value, torch.Tensor) or value.numel() == 0 or not isinstance(names, list):
            return 0.0
        if feature_name not in names:
            return 0.0
        value = value.detach().float().cpu()
        if value.dim() != 3 or service_idx < 0 or service_idx >= value.size(0):
            return 0.0
        if device_idx < 0 or device_idx >= value.size(1):
            return 0.0
        return float(value[service_idx, device_idx, names.index(feature_name)].item())

    @staticmethod
    def _actor_debug_service_scalar(
            actor_debug: Optional[Dict[str, Any]],
            key: str,
            service_idx: int,
    ) -> float:
        if not isinstance(actor_debug, dict):
            return 0.0
        value = actor_debug.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        value = value.detach().float().cpu()
        if value.dim() != 1 or service_idx < 0 or service_idx >= value.size(0):
            return 0.0
        return float(value[service_idx].item())

    @staticmethod
    def _tensor_mean(value: Optional[torch.Tensor]) -> float:
        if value is None or not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())

    @staticmethod
    def _scalar_value(value, default: float = 0.0) -> float:
        if value is None:
            return float(default)
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().float().mean().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _sync_decision_device(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _latest_overhead(estimator) -> float:
        if estimator is None:
            return 0.0
        return float(estimator.get_latest_overhead())

    def get_deployment_decision_overhead(self) -> float:
        return self._latest_overhead(self.deployment_overhead_estimator)

    def get_offloading_decision_overhead(self) -> float:
        return self._latest_overhead(self.offloading_overhead_estimator)

    def get_schedule_overhead(self) -> float:
        deployment_overhead = self.get_deployment_decision_overhead()
        offloading_overhead = self.get_offloading_decision_overhead()
        if self.deployment_interval > 0.0:
            deployment_weight = self.offloading_interval / self.deployment_interval
        else:
            deployment_weight = 1.0
        return offloading_overhead + deployment_weight * deployment_overhead

    def _state_record_metrics(
            self,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            state_debug: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build record metrics for the state used to sample an action.

        Compact scalar summaries are suitable for long training runs.
        Raw JSON snapshots are intended for short debug runs and are gated by
        `record.state_snapshot_debug`.
        """
        state_debug = state_debug or {}
        runtime_pair_snapshot = state_debug.get("runtime_pair_snapshot", {}) or {}
        queue_pair_snapshot = state_debug.get("queue_pair_snapshot", {}) or {}
        row: Dict[str, Any] = {}
        if self.record_cfg.state_summary:
            service_count = int(logic_feats["model_flops"].numel()) if "model_flops" in logic_feats else 0
            device_count = int(phys_feats["gpu_flops"].numel()) if "gpu_flops" in phys_feats else 0

            latest_complexity = 0.0
            complexity_seq = logic_feats.get("task_complexity_seq")
            if isinstance(complexity_seq, torch.Tensor) and complexity_seq.numel():
                latest_complexity = float(complexity_seq[:, -1].float().mean().item())

            latest_arrival_rate = 0.0
            latest_arrival_rate_max = 0.0
            arrival_rate_seq = logic_feats.get("task_arrival_rate_seq")
            if isinstance(arrival_rate_seq, torch.Tensor) and arrival_rate_seq.numel():
                latest_arrival = arrival_rate_seq[:, -1].float()
                latest_arrival_rate = float(latest_arrival.mean().item())
                latest_arrival_rate_max = float(latest_arrival.max().item())

            latest_cloud_bw = 0.0
            latest_edge_bw = 0.0
            bandwidth_latest = phys_feats.get("bandwidth_latest")
            if isinstance(bandwidth_latest, torch.Tensor) and bandwidth_latest.numel():
                cloud_idx = (
                    self.physical_topology.cloud_idx
                    if self.physical_topology is not None
                    else bandwidth_latest.size(0) - 1
                )
                latest_cloud_bw = float(bandwidth_latest[cloud_idx].float().item())
                edge_bandwidth = bandwidth_latest
                if edge_bandwidth.numel() > 1:
                    edge_bandwidth = torch.cat([edge_bandwidth[:cloud_idx], edge_bandwidth[cloud_idx + 1:]])
                else:
                    edge_bandwidth = torch.empty(0, dtype=edge_bandwidth.dtype)
                latest_edge_bw = float(edge_bandwidth.float().mean().item()) if edge_bandwidth.numel() else 0.0

            runtime_means = [0.0] * len(RUNTIME_PAIR_FEATURE_NAMES)
            runtime_pair = logic_feats.get("runtime_pair_feat")
            if isinstance(runtime_pair, torch.Tensor) and runtime_pair.dim() == 3 \
                    and runtime_pair.size(-1) >= len(RUNTIME_PAIR_FEATURE_NAMES) and runtime_pair.numel():
                runtime_means = [
                    float(runtime_pair[..., idx].detach().float().mean().cpu().item())
                    for idx in range(len(RUNTIME_PAIR_FEATURE_NAMES))
                ]

            row.update({
                "state_services": service_count,
                "state_devices": device_count,
                "state_complexity": latest_complexity,
                "state_arrival_rate_mean": latest_arrival_rate,
                "state_arrival_rate_max": latest_arrival_rate_max,
                "state_edge_bw": latest_edge_bw,
                "state_cloud_bw": latest_cloud_bw,
                "state_model_flops_mean": self._tensor_mean(logic_feats.get("model_flops")),
                "state_model_mem_mean": self._tensor_mean(logic_feats.get("model_mem")),
                "state_gpu_flops_mean": self._tensor_mean(phys_feats.get("gpu_flops")),
                "state_mem_capacity_mean": self._tensor_mean(phys_feats.get("mem_capacity")),
                "state_runtime_pair_obs_count": self._nested_float_mean(runtime_pair_snapshot.get("pair_count")),
                "state_queue_pair_obs_count": self._nested_float_mean(queue_pair_snapshot.get("pair_count")),
                "state_runtime_queue_short_mean": runtime_means[0],
                "state_runtime_busy_ratio_mean": runtime_means[1],
                "state_runtime_real_time_per_complexity_mean": runtime_means[2],
                "state_runtime_confidence_mean": runtime_means[3],
                "state_runtime_recency_mean": runtime_means[4],
                "state_queue_freshness_mean": runtime_means[5],
            })

        if self.record_cfg.state_snapshot_debug:
            row.update({
                "state_logic_snapshot": self._json_for_record(state_debug.get("logic_snapshot")),
                "state_phys_snapshot": self._json_for_record(state_debug.get("phys_snapshot")),
                "state_runtime_source_snapshot": self._json_for_record(runtime_pair_snapshot),
                "state_queue_pair_snapshot": self._json_for_record(queue_pair_snapshot),
                "state_service_demand_snapshot": self._json_for_record(logic_feats.get("service_demand_feat")),
                "state_device_capability_snapshot": self._json_for_record(phys_feats.get("device_capability_feat")),
                "state_runtime_pair_snapshot": self._json_for_record(logic_feats.get("runtime_pair_feat")),
            })
        return row

    @staticmethod
    def _state_record_summary_fieldnames() -> List[str]:
        return [
            "state_services", "state_devices", "state_complexity",
            "state_arrival_rate_mean", "state_arrival_rate_max",
            "state_edge_bw", "state_cloud_bw",
            "state_model_flops_mean", "state_model_mem_mean",
            "state_gpu_flops_mean", "state_mem_capacity_mean",
            "state_runtime_pair_obs_count", "state_queue_pair_obs_count",
            "state_runtime_queue_short_mean", "state_runtime_busy_ratio_mean",
            "state_runtime_real_time_per_complexity_mean", "state_runtime_confidence_mean",
            "state_runtime_recency_mean", "state_queue_freshness_mean",
        ]

    @staticmethod
    def _state_record_debug_fieldnames() -> List[str]:
        return [
            "state_logic_snapshot", "state_phys_snapshot",
            "state_runtime_source_snapshot", "state_queue_pair_snapshot",
            "state_service_demand_snapshot", "state_device_capability_snapshot", "state_runtime_pair_snapshot",
        ]

    def _state_record_fieldnames(self) -> List[str]:
        fieldnames: List[str] = []
        if self.record_cfg.state_summary:
            fieldnames.extend(self._state_record_summary_fieldnames())
        if self.record_cfg.state_snapshot_debug:
            fieldnames.extend(self._state_record_debug_fieldnames())
        return fieldnames

    @staticmethod
    def _latency_guard_record_fieldnames() -> List[str]:
        return [
            "latency_guard_active", "latency_guard_trigger_seq", "latency_guard_bad_ratio",
            "latency_guard_bad_count", "latency_guard_sample_count",
            "latency_guard_max_queue",
        ]

    def _latency_guard_record_metrics(self) -> Dict[str, Any]:
        stats = self.latency_guard_status()
        return {
            "latency_guard_active": int(bool(stats.get("active", False))),
            "latency_guard_trigger_seq": int(stats.get("trigger_seq", 0) or 0),
            "latency_guard_bad_ratio": float(stats.get("bad_ratio", 0.0) or 0.0),
            "latency_guard_bad_count": int(stats.get("bad_count", 0) or 0),
            "latency_guard_sample_count": int(stats.get("count", 0) or 0),
            "latency_guard_max_queue": float(stats.get("max_queue_length", 0.0) or 0.0),
        }

    def _inference_deployment_fieldnames(self) -> List[str]:
        fieldnames = [
            "step", "epoch", "decision_version", "served_deployment_version", "decision_reason", "interval_s",
            "deployment_decision_overhead_s", "dep_reward_estimate",
            "avg_off_reward", "off_reward_std", "off_reward_count",
            "dep_change_cost", "dep_latency_cost",
            "dep_offload_term", "dep_latency_term", "dep_slo_term", "dep_change_term",
            "dep_cloud_only_term", "dep_capacity_relax_term", "dep_edge_cover_repair_term",
            "dep_hotspot_term", "dep_runtime_risk_term", "dep_unknown_option_term",
            "dep_stale_option_term", "dep_low_quality_term",
            "dep_latency_guard_penalty_term", "dep_feedback_timeout_term",
            "active_pair_hotspot_cost", "executed_active_pair_hotspot_cost",
            "e2e_latency_count", "e2e_latency_mean", "e2e_latency_latest",
            "e2e_latency_p50", "e2e_latency_p90", "e2e_latency_p95", "e2e_latency_p99",
            "e2e_slo_violation", "feedback_gate_enabled", "feedback_gate_required_samples",
            "feedback_gate_collected_samples", "feedback_gate_sample_shortfall",
            "feedback_gate_shortfall_ratio", "feedback_gate_timed_out", "feedback_gate_guard_truncated",
            "feedback_timeout_penalty_cost", "deployment_event_triggered",
            "deployment_event_reason", "deployment_event_queue_pressure",
            "deployment_event_hotspot_pressure", "deployment_event_max_queue",
            "deployment_event_e2e_slo_violation", "deployment_event_e2e_p95",
            "deployment_event_e2e_feedback_count",
            "deployment_event_suppressed", "deployment_event_suppress_reason",
            "deployment_event_suppressed_event_reason", "deployment_event_warmup_remaining_s",
            "deployment_event_feedback_count", "deployment_event_feedback_required",
            "deployment_deterministic",
            "cap_relax_cnt", "cap_relax_cost", "edge_cover_repair_cnt",
            "edge_cover_repair_cost", "edge_cover_unmet", "hotspot_repair_cnt",
            "hotspot_repair_cost", "hotspot_unmet", "policy_logp", "policy_entropy",
            "value_estimate", "raw_edge_replicas", "decoded_edge_replicas", "edge_replicas", "cloud_replicas",
            "raw_zero_edge_services", "decoded_zero_edge_services",
            "matrix_added_cnt", "matrix_kept_cnt", "matrix_removed_cnt",
            "decode_added_cnt", "decode_pruned_cnt",
            "capacity_removed_cnt", "selected_queue_pressure_cost", "selected_runtime_risk_cost",
            "selected_unknown_option_cost", "selected_stale_option_cost",
            "selected_runtime_weakness_cost", "selected_low_quality_option_cost",
            "selected_evidence_untrusted_cost", "selected_risky_pair_count",
            "selected_low_quality_pair_count",
            "cloud_only", "cloud_only_ratio", "empty_edge_devices", "empty_edge_device_ratio",
            "raw_deployment_plan", "deployment_plan", "active_deployment_plan",
            *self._state_record_fieldnames(),
            *Hedger._latency_guard_record_fieldnames(),
            "dep_offload_weight", "dep_latency_weight", "dep_latency_transform",
            "dep_latency_normalizer", "dep_latency_clip", "dep_slo_weight",
            "dep_change_weight", "dep_cloud_only_weight", "cap_relax_weight", "edge_cover_repair_weight",
            "hotspot_weight", "runtime_risk_weight", "unknown_option_weight",
            "stale_option_weight", "low_quality_weight",
            "latency_guard_penalty_weight", "feedback_timeout_penalty_weight", "max_edge_replicas_per_device",
            "edge_memory_budget_ratio", "select_threshold",
            "negative_queue_threshold", "negative_hotspot_threshold",
            "negative_runtime_risk_threshold", "negative_unknown_threshold",
            "negative_stale_threshold", "positive_quality_threshold",
            "queue_normalizer", "loaded_checkpoint",
        ]
        if self.record_cfg.actor_snapshot_debug:
            insert_at = fieldnames.index("latency_guard_active")
            fieldnames.insert(insert_at, "state_deployment_actor_snapshot")
        return fieldnames

    def _inference_offloading_fieldnames(self) -> List[str]:
        fieldnames = [
            "step", "epoch", "served_deployment_version", "interval_s",
            "offloading_decision_overhead_s", "off_reward_estimate",
            "latency", "latency_cost", "off_latency_term", "slo_violation", "off_slo_term",
            "cloud_fraction", "task_latency_count", "latest_task_latency",
            "off_cloud_term", "off_projection_term", "off_queue_term",
            "off_queue_cost", "off_queue_risk_cost",
            "policy_logp", "policy_entropy", "value_estimate",
            "proposal_cloud_fraction", "projected_cloud_fraction",
            "offloading_projection_cnt", "offloading_dependency_projection_cnt",
            "offloading_infeasible_projection_cnt", "offloading_projection_cost",
            "off_selected_runtime_ratio", "off_selected_runtime_recency",
            "off_selected_queue_freshness", "off_selected_speed_evidence",
            "off_selected_capacity_pressure", "off_selected_pair_load",
            "off_selected_device_load", "off_selected_load_pressure",
            "off_selected_service_time_factor", "off_selected_base_queue_risk",
            "off_selected_relative_queue_risk", "off_selected_overload_risk",
            "off_selected_queue_risk_total", "off_selected_planned_load_risk",
            "off_selected_relative_planned_load_risk", "off_selected_dynamic_risk",
            "off_selected_offered_load_pressure", "off_selected_offered_load_risk",
            "off_selected_compute_relative_weakness", "off_selected_runtime_relative_weakness",
            "off_selected_relative_weakness", "off_selected_weak_replica_risk",
            "off_selected_weak_pressure",
            "off_selected_runtime_confidence",
            "unique_targets", "feasible_targets_mean",
            "offloading_deterministic",
            "feasible_targets_min", "feasible_targets_max",
            "offloading_plan", "active_deployment_plan", *self._state_record_fieldnames(),
            *Hedger._latency_guard_record_fieldnames(),
            "feedback_task_observations", "feedback_deployment_version",
            "feedback_deployment_versions", "feedback_recorded", "off_latency_weight",
            "off_latency_transform", "off_latency_normalizer", "off_latency_clip",
            "off_slo_weight", "off_cloud_weight", "off_projection_weight",
            "off_queue_weight", "off_queue_clip",
            "loaded_checkpoint",
        ]
        if self.record_cfg.actor_snapshot_debug:
            insert_at = fieldnames.index("latency_guard_active")
            fieldnames.insert(insert_at, "state_offloading_actor_snapshot")
        return fieldnames

    @staticmethod
    def _latest_feature_value(feats: Dict[str, torch.Tensor], key: str, idx: int) -> float:
        value = feats.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return 0.0
        value = value.detach().float().cpu()
        if value.dim() == 0:
            return float(value.item())
        if idx < 0 or idx >= value.size(0):
            return 0.0
        if value.dim() == 1:
            return float(value[idx].item())
        return float(value[idx, -1].item())

    @staticmethod
    def _decision_common_fieldnames() -> List[str]:
        return [
            "step", "epoch", "update_steps", "service_idx", "service",
            "service_model_flops", "service_model_mem", "state_complexity",
        ]

    def _deployment_decision_fieldnames(self) -> List[str]:
        fieldnames = [
            *Hedger._decision_common_fieldnames(),
            "raw_nodes", "executed_nodes", "removed_nodes", "added_nodes",
            "raw_edge_replicas", "executed_edge_replicas", "cloud_replica",
            "raw_zero_edge", "decoded_zero_edge", "matrix_added_nodes", "matrix_kept_nodes",
            "matrix_removed_nodes", "decode_added_nodes", "decode_pruned_nodes",
            "decode_added_reason", "decode_pruned_reason", "capacity_removed_nodes",
            "service_pressure", "edge_feasible_count", "edge_replica_count",
            "device_replica_count", "active_pair_hotspot",
            "collect_behavior", "collect_operation",
            "collect_fallback_selected_best",
            "collect_selected_service", "collect_selected_device",
            "collect_removed_service", "collect_removed_device",
            "collect_base_queue_pressure", "collect_base_hotspot_cost",
            "collect_base_runtime_risk", "collect_base_min_pair_quality",
            "collect_candidate_queue_pressure", "collect_candidate_hotspot_cost",
            "collect_candidate_runtime_risk", "collect_candidate_min_pair_quality",
            "collect_delta_queue_pressure", "collect_delta_hotspot_cost",
            "collect_delta_runtime_risk", "collect_delta_min_pair_quality",
            "collect_new_edge_count", "collect_removed_edge_count",
            "collect_new_edge_runtime_risk", "collect_new_edge_min_pair_quality",
            "collect_base_low_quality_count", "collect_candidate_low_quality_count",
            "collect_delta_low_quality_count", "collect_base_risky_pair_count",
            "collect_candidate_risky_pair_count",
            "capacity_corrected", "deployment_policy_deterministic",
        ]
        if self.record_cfg.decision_candidate_features_debug:
            fieldnames.extend([
                "deployment_runtime_pair_features",
                "deployment_device_capability_features",
                "deployment_candidate_features",
            ])
        if self.record_cfg.decision_actor_debug:
            fieldnames.extend([
                "deployment_qk_scores", "deployment_qk_features",
                "deployment_pair_adjustments", "deployment_base_scores", "deployment_centered_scores",
                "deployment_final_scores", "deployment_select_logits",
                "deployment_select_probs", "deployment_decode_scores",
                "deployment_safety_prior",
                "deployment_static_option_score", "deployment_runtime_risk_score",
                "deployment_pair_quality", "deployment_evidence_confidence",
                "deployment_evidence_untrusted", "deployment_low_quality_gap",
                "deployment_queue_pressure", "deployment_runtime_unknown_risk",
                "deployment_runtime_stale_risk", "deployment_runtime_relative_weakness",
                "deployment_qk_term", "deployment_pair_adjustment_term",
                "deployment_quality_term", "deployment_confidence_term",
                "deployment_service_pressure_term", "deployment_inertia_term",
                "deployment_unknown_penalty_term", "deployment_stale_penalty_term",
                "deployment_runtime_risk_penalty_term", "deployment_low_quality_penalty_term",
                "deployment_queue_penalty_term", "deployment_memory_penalty_term",
                "deployment_device_load_penalty_term", "deployment_hotspot_penalty_term",
                "deployment_policy_probs", "deployment_raw_threshold_nodes",
                "deployment_decoded_nodes", "deployment_positive_mask", "deployment_negative_mask",
                "deployment_service_pressure",
                "deployment_edge_replica_counts", "deployment_device_replica_counts",
                "deployment_active_pair_hotspots",
                "deployment_static_mask",
            ])
        return fieldnames

    def _offloading_decision_fieldnames(self) -> List[str]:
        fieldnames = [
            *Hedger._decision_common_fieldnames(),
            "proposal_target", "target", "projected", "projection_reason", "is_cloud",
            "feasible_targets", "feasible_target_count", "parent_targets",
            "target_gpu_flops", "target_bandwidth", "target_role",
            "target_qk_feature", "target_compute_gap", "target_arrival_rate_short",
            "target_runtime_ratio", "target_runtime_confidence", "target_runtime_recency",
            "target_queue_freshness", "target_speed_evidence", "target_capacity_pressure",
            "target_pair_load", "target_device_load", "target_service_time_factor",
            "target_cross_tier_penalty", "target_static_prior", "target_runtime_risk",
            "target_load_pressure", "target_base_queue_risk", "target_relative_queue_risk",
            "target_overload_risk", "target_queue_risk_total", "target_planned_pressure",
            "target_planned_load_risk", "target_relative_planned_load_risk",
            "target_offered_load_pressure", "target_offered_load_risk",
            "target_compute_relative_weakness", "target_runtime_relative_weakness",
            "target_relative_weakness", "target_weak_replica_risk",
            "target_weak_pressure",
            "target_cloud_penalty", "target_cross_tier_penalty_term",
            "target_dynamic_risk", "target_planned_device_load", "target_final_score",
            "offloading_policy_deterministic",
        ]
        if self.record_cfg.decision_candidate_features_debug:
            fieldnames.extend([
                "offloading_service_demand_features",
                "offloading_device_capability_features",
                "offloading_runtime_pair_features",
                "offloading_candidate_features",
            ])
        if self.record_cfg.decision_actor_debug:
            fieldnames.extend([
                "offloading_qk_scores", "offloading_qk_features",
                "offloading_static_priors",
                "offloading_runtime_risks",
                "offloading_service_time_factors",
                "offloading_runtime_recencies",
                "offloading_queue_freshnesses",
                "offloading_speed_evidences",
                "offloading_capacity_pressures",
                "offloading_load_pressures",
                "offloading_base_queue_risks",
                "offloading_relative_queue_risks",
                "offloading_overload_risks",
                "offloading_queue_risk_totals",
                "offloading_planned_pressures",
                "offloading_planned_load_risks",
                "offloading_relative_planned_load_risks",
                "offloading_offered_load_pressures",
                "offloading_offered_load_risks",
                "offloading_compute_relative_weaknesses",
                "offloading_runtime_relative_weaknesses",
                "offloading_relative_weaknesses",
                "offloading_weak_pressures",
                "offloading_weak_replica_risks",
                "offloading_cloud_penalties",
                "offloading_cross_tier_penalty_terms",
                "offloading_dynamic_risks",
                "offloading_planned_device_loads",
                "offloading_final_scores",
                "offloading_base_policy_probs",
                "offloading_unknown_policy_probs",
                "offloading_unknown_exploration_weights",
                "offloading_unknown_exploration_eps",
                "offloading_risk_policy_probs",
                "offloading_risk_exploration_weights",
                "offloading_risk_exploration_eps",
                "offloading_policy_probs", "offloading_effective_mask",
            ])
        return fieldnames

    def _service_name(self, service_idx: int) -> str:
        if self.logical_topology is None:
            return str(service_idx)
        return self.logical_topology[service_idx]

    def _device_name(self, device_idx: int) -> str:
        if self.physical_topology is None:
            return str(device_idx)
        return self.physical_topology[device_idx]

    def _device_names_from_indices(self, indices: List[int]) -> List[str]:
        return [self._device_name(int(idx)) for idx in indices]

    def _device_names_from_mask(self, mask_row: torch.Tensor) -> List[str]:
        indices = torch.nonzero(mask_row.detach().cpu().bool(), as_tuple=False).flatten().tolist()
        return self._device_names_from_indices(indices)

    @staticmethod
    def _cpu_tensor_dict(feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in feats.items()
        }

    def _deployment_decision_debug_fallback(
            self,
            logic_edge_index: Optional[torch.Tensor],
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            exec_deploy_mask: torch.Tensor,
            actor_debug: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(actor_debug, dict):
            debug: Dict[str, Any] = {}
        else:
            debug = dict(actor_debug)

        required_keys = {
            "service_pressure",
            "edge_feasible_count",
            "edge_replica_count",
            "device_replica_count",
            "active_pair_hotspot",
            "static_mask",
        }
        if all(isinstance(debug.get(key), torch.Tensor) and debug[key].numel() > 0 for key in required_keys):
            return debug
        if self.deployment_agent is None:
            return debug

        try:
            with torch.no_grad():
                logic_feats_cpu = self._cpu_tensor_dict(logic_feats)
                phys_feats_cpu = self._cpu_tensor_dict(phys_feats)
                edge_index_cpu = (
                    logic_edge_index.detach().cpu()
                    if isinstance(logic_edge_index, torch.Tensor)
                    else logic_edge_index
                )
                deploy_mask_cpu = exec_deploy_mask.detach().cpu().bool()
                static_allowed = self.deployment_agent._static_allowed_mask(
                    phys_feats_cpu,
                    logic_feats_cpu,
                ).detach().cpu().bool()
                pair_ctx = self.deployment_agent._deployment_pair_context(
                    edge_index_cpu,
                    logic_feats_cpu,
                    static_allowed,
                    deploy_mask=deploy_mask_cpu,
                )
                for key in (
                        "service_pressure",
                        "edge_feasible_count",
                        "edge_replica_count",
                        "device_replica_count",
                        "active_pair_hotspot",
                ):
                    if not isinstance(debug.get(key), torch.Tensor) or debug[key].numel() == 0:
                        debug[key] = pair_ctx[key].detach().cpu()
                if not isinstance(debug.get("static_mask"), torch.Tensor) or debug["static_mask"].numel() == 0:
                    debug["static_mask"] = static_allowed.detach().cpu()
        except Exception as exc:
            LOGGER.debug(f"[Hedger][Record] Failed to build deployment decision debug fallback: {exc}")
        return debug

    def _log_deployment_decisions(
            self,
            *,
            step: int,
            raw_deploy_mask: torch.Tensor,
            exec_deploy_mask: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            logic_edge_index: Optional[torch.Tensor] = None,
            actor_debug: Optional[Dict[str, Any]] = None,
            collect_debug: Optional[Dict[str, Any]] = None,
            policy_deterministic: Optional[bool] = None,
    ) -> None:
        if self.dep_decision_recorder is None or self.physical_topology is None:
            return

        cloud_idx = self.physical_topology.cloud_idx
        raw_mask = raw_deploy_mask.detach().cpu().bool()
        exec_mask = exec_deploy_mask.detach().cpu().bool()
        actor_debug = self._deployment_decision_debug_fallback(
            logic_edge_index,
            logic_feats,
            phys_feats,
            exec_mask,
            actor_debug,
        )
        collect_debug = collect_debug if isinstance(collect_debug, dict) else {}
        selected_service_idx = int(collect_debug.get("collect_selected_service_idx", -1) or -1)
        selected_device_idx = int(collect_debug.get("collect_selected_device_idx", -1) or -1)
        removed_service_idx = int(collect_debug.get("collect_removed_service_idx", -1) or -1)
        removed_device_idx = int(collect_debug.get("collect_removed_device_idx", -1) or -1)
        for service_idx in range(raw_mask.size(0)):
            raw_nodes = self._device_names_from_mask(raw_mask[service_idx])
            executed_nodes = self._device_names_from_mask(exec_mask[service_idx])
            removed_indices = torch.nonzero(raw_mask[service_idx] & ~exec_mask[service_idx], as_tuple=False)
            added_indices = torch.nonzero(~raw_mask[service_idx] & exec_mask[service_idx], as_tuple=False)
            raw_edge_count = int(raw_mask[service_idx, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
            exec_edge_count = int(exec_mask[service_idx, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
            decoded_tensor = actor_debug.get("decoded_mask")
            if isinstance(decoded_tensor, torch.Tensor) and decoded_tensor.dim() == 2 \
                    and decoded_tensor.size(0) > service_idx:
                decoded_row = decoded_tensor[service_idx].detach().cpu().bool()
            else:
                decoded_row = exec_mask[service_idx]
            decoded_edge_count = int(decoded_row[:cloud_idx].sum().item()) if cloud_idx > 0 else 0
            decode_added_tensor = actor_debug.get("decode_added_mask")
            if isinstance(decode_added_tensor, torch.Tensor) and decode_added_tensor.dim() == 2 \
                    and decode_added_tensor.size(0) > service_idx:
                decode_added_row = decode_added_tensor[service_idx].detach().cpu().bool()
                decode_added_indices = torch.nonzero(decode_added_row, as_tuple=False).flatten().tolist()
            else:
                decode_added_indices = []
            decode_pruned_tensor = actor_debug.get("decode_pruned_mask")
            if isinstance(decode_pruned_tensor, torch.Tensor) and decode_pruned_tensor.dim() == 2 \
                    and decode_pruned_tensor.size(0) > service_idx:
                decode_pruned_row = decode_pruned_tensor[service_idx].detach().cpu().bool()
                decode_pruned_indices = torch.nonzero(decode_pruned_row, as_tuple=False).flatten().tolist()
            else:
                decode_pruned_indices = []
            matrix_added_tensor = actor_debug.get("matrix_added_mask")
            if isinstance(matrix_added_tensor, torch.Tensor) and matrix_added_tensor.dim() == 2 \
                    and matrix_added_tensor.size(0) > service_idx:
                matrix_added_indices = torch.nonzero(
                    matrix_added_tensor[service_idx].detach().cpu().bool(),
                    as_tuple=False,
                ).flatten().tolist()
            else:
                matrix_added_indices = []
            matrix_kept_tensor = actor_debug.get("matrix_kept_mask")
            if isinstance(matrix_kept_tensor, torch.Tensor) and matrix_kept_tensor.dim() == 2 \
                    and matrix_kept_tensor.size(0) > service_idx:
                matrix_kept_indices = torch.nonzero(
                    matrix_kept_tensor[service_idx].detach().cpu().bool(),
                    as_tuple=False,
                ).flatten().tolist()
            else:
                matrix_kept_indices = []
            matrix_removed_tensor = actor_debug.get("matrix_removed_mask")
            if isinstance(matrix_removed_tensor, torch.Tensor) and matrix_removed_tensor.dim() == 2 \
                    and matrix_removed_tensor.size(0) > service_idx:
                matrix_removed_indices = torch.nonzero(
                    matrix_removed_tensor[service_idx].detach().cpu().bool(),
                    as_tuple=False,
                ).flatten().tolist()
            else:
                matrix_removed_indices = []
            capacity_removed_tensor = actor_debug.get("capacity_removed_mask")
            if isinstance(capacity_removed_tensor, torch.Tensor) and capacity_removed_tensor.dim() == 2 \
                    and capacity_removed_tensor.size(0) > service_idx:
                capacity_removed_row = capacity_removed_tensor[service_idx].detach().cpu().bool()
                capacity_removed_indices = torch.nonzero(capacity_removed_row, as_tuple=False).flatten().tolist()
            else:
                capacity_removed_indices = []
            added_reason_tensor = actor_debug.get("decode_added_reason")
            if isinstance(added_reason_tensor, torch.Tensor) and added_reason_tensor.dim() == 2 \
                    and added_reason_tensor.size(0) > service_idx:
                added_reason_row = added_reason_tensor[service_idx].detach().cpu().long()
                added_reason_codes = [
                    int(added_reason_row[idx].item()) for idx in decode_added_indices
                    if 0 <= idx < added_reason_row.numel()
                ]
            else:
                added_reason_codes = []
            pruned_reason_tensor = actor_debug.get("decode_pruned_reason")
            if isinstance(pruned_reason_tensor, torch.Tensor) and pruned_reason_tensor.dim() == 2 \
                    and pruned_reason_tensor.size(0) > service_idx:
                pruned_reason_row = pruned_reason_tensor[service_idx].detach().cpu().long()
                pruned_reason_codes = [
                    int(pruned_reason_row[idx].item()) for idx in decode_pruned_indices
                    if 0 <= idx < pruned_reason_row.numel()
                ]
            else:
                pruned_reason_codes = []
            reason_names = [
                {
                    1: "bernoulli_sample",
                }.get(code, "")
                for code in added_reason_codes
                if code > 0
            ]
            pruned_reason_names = [
                {1: "matrix_removed"}.get(code, "")
                for code in pruned_reason_codes
                if code > 0
            ]
            edge_feasible_count = self._actor_debug_vector_value(actor_debug, "edge_feasible_count", service_idx)
            row = dict(
                step=step,
                epoch=self._epoch,
                update_steps=self._deployment_update_steps,
                service_idx=service_idx,
                service=self._service_name(service_idx),
                service_model_flops=self._latest_feature_value(logic_feats, "model_flops", service_idx),
                service_model_mem=self._latest_feature_value(logic_feats, "model_mem", service_idx),
                state_complexity=self._latest_feature_value(logic_feats, "task_complexity_seq", service_idx),
                raw_nodes=self._json_for_record(raw_nodes),
                executed_nodes=self._json_for_record(executed_nodes),
                removed_nodes=self._json_for_record(
                    self._device_names_from_indices(removed_indices.flatten().tolist())
                ),
                added_nodes=self._json_for_record(
                    self._device_names_from_indices(added_indices.flatten().tolist())
                ),
                raw_edge_replicas=raw_edge_count,
                executed_edge_replicas=exec_edge_count,
                cloud_replica=bool(exec_mask[service_idx, cloud_idx].item()),
                raw_zero_edge=int(raw_edge_count <= 0 and edge_feasible_count > 0.0),
                decoded_zero_edge=int(decoded_edge_count <= 0 and edge_feasible_count > 0.0),
                matrix_added_nodes=self._json_for_record(self._device_names_from_indices(matrix_added_indices)),
                matrix_kept_nodes=self._json_for_record(self._device_names_from_indices(matrix_kept_indices)),
                matrix_removed_nodes=self._json_for_record(self._device_names_from_indices(matrix_removed_indices)),
                decode_added_nodes=self._json_for_record(self._device_names_from_indices(decode_added_indices)),
                decode_pruned_nodes=self._json_for_record(self._device_names_from_indices(decode_pruned_indices)),
                decode_added_reason=self._json_for_record(reason_names),
                decode_pruned_reason=self._json_for_record(pruned_reason_names),
                capacity_removed_nodes=self._json_for_record(
                    self._device_names_from_indices(capacity_removed_indices)
                ),
                service_pressure=self._actor_debug_vector_value(actor_debug, "service_pressure", service_idx),
                edge_feasible_count=edge_feasible_count,
                edge_replica_count=self._actor_debug_vector_value(actor_debug, "edge_replica_count", service_idx),
                device_replica_count=self._json_for_record(
                    self._actor_debug_device_vector_map(actor_debug, "device_replica_count")
                ),
                active_pair_hotspot=self._json_for_record(
                    self._actor_debug_row_map(actor_debug, "active_pair_hotspot", service_idx)
                ),
                collect_behavior=collect_debug.get("collect_behavior", ""),
                collect_operation=collect_debug.get("collect_operation", ""),
                collect_fallback_selected_best=collect_debug.get("collect_fallback_selected_best", ""),
                collect_selected_service=(
                    self._service_name(selected_service_idx) if selected_service_idx >= 0 else ""
                ),
                collect_selected_device=(
                    self._device_name(selected_device_idx) if selected_device_idx >= 0 else ""
                ),
                collect_removed_service=(
                    self._service_name(removed_service_idx) if removed_service_idx >= 0 else ""
                ),
                collect_removed_device=(
                    self._device_name(removed_device_idx) if removed_device_idx >= 0 else ""
                ),
                collect_base_queue_pressure=collect_debug.get("collect_base_queue_pressure", ""),
                collect_base_hotspot_cost=collect_debug.get("collect_base_hotspot_cost", ""),
                collect_base_runtime_risk=collect_debug.get("collect_base_runtime_risk", ""),
                collect_base_min_pair_quality=collect_debug.get("collect_base_min_pair_quality", ""),
                collect_candidate_queue_pressure=self._actor_debug_vector_value(
                    collect_debug,
                    "collect_candidate_queue_pressure_by_service",
                    service_idx,
                ),
                collect_candidate_hotspot_cost=self._actor_debug_vector_value(
                    collect_debug,
                    "collect_candidate_hotspot_by_service",
                    service_idx,
                ),
                collect_candidate_runtime_risk=collect_debug.get("collect_candidate_runtime_risk", ""),
                collect_candidate_min_pair_quality=collect_debug.get("collect_candidate_min_pair_quality", ""),
                collect_delta_queue_pressure=collect_debug.get("collect_delta_queue_pressure", ""),
                collect_delta_hotspot_cost=collect_debug.get("collect_delta_hotspot_cost", ""),
                collect_delta_runtime_risk=collect_debug.get("collect_delta_runtime_risk", ""),
                collect_delta_min_pair_quality=collect_debug.get("collect_delta_min_pair_quality", ""),
                collect_new_edge_count=collect_debug.get("collect_new_edge_count", ""),
                collect_removed_edge_count=collect_debug.get("collect_removed_edge_count", ""),
                collect_new_edge_runtime_risk=collect_debug.get("collect_new_edge_runtime_risk", ""),
                collect_new_edge_min_pair_quality=collect_debug.get("collect_new_edge_min_pair_quality", ""),
                collect_base_low_quality_count=collect_debug.get("collect_base_low_quality_count", ""),
                collect_candidate_low_quality_count=collect_debug.get("collect_candidate_low_quality_count", ""),
                collect_delta_low_quality_count=collect_debug.get("collect_delta_low_quality_count", ""),
                collect_base_risky_pair_count=collect_debug.get("collect_base_risky_pair_count", ""),
                collect_candidate_risky_pair_count=collect_debug.get("collect_candidate_risky_pair_count", ""),
                capacity_corrected=bool(not torch.equal(raw_mask[service_idx], exec_mask[service_idx])),
                deployment_policy_deterministic=(
                    "" if policy_deterministic is None else int(bool(policy_deterministic))
                ),
            )
            if self.record_cfg.decision_candidate_features_debug:
                row.update({
                    "deployment_runtime_pair_features": self._json_for_record(
                        self._pair_feature_map(
                            logic_feats,
                            "runtime_pair_feat",
                            service_idx,
                            names=RUNTIME_PAIR_FEATURE_NAMES,
                        )
                    ),
                    "deployment_device_capability_features": self._json_for_record(
                        self._device_feature_map(
                            phys_feats,
                            key="device_capability_feat",
                            names=DEVICE_CAPABILITY_FEATURE_NAMES,
                        )
                    ),
                    "deployment_candidate_features": self._json_for_record(
                        self._actor_debug_pair_feature_map(actor_debug, "candidate_feature", service_idx)
                    ),
                })
            if self.record_cfg.decision_actor_debug:
                row.update({
                    "deployment_qk_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_score", service_idx)
                    ),
                    "deployment_qk_features": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_feature", service_idx)
                    ),
                    "deployment_pair_adjustments": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "pair_adjustment", service_idx)
                    ),
                    "deployment_base_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "base_score", service_idx)
                    ),
                    "deployment_centered_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "centered_score", service_idx)
                    ),
                    "deployment_final_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "final_score", service_idx)
                    ),
                    "deployment_select_logits": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "select_logit", service_idx)
                    ),
                    "deployment_select_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "select_prob", service_idx)
                    ),
                    "deployment_decode_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "decode_score", service_idx)
                    ),
                    "deployment_safety_prior": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "safety_prior", service_idx)
                    ),
                    "deployment_static_option_score": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "static_option_score", service_idx)
                    ),
                    "deployment_runtime_risk_score": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_risk_score", service_idx)
                    ),
                    "deployment_evidence_confidence": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "evidence_confidence", service_idx)
                    ),
                    "deployment_evidence_untrusted": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "evidence_untrusted", service_idx)
                    ),
                    "deployment_low_quality_gap": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "low_quality_gap", service_idx)
                    ),
                    "deployment_pair_quality": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "pair_quality_score", service_idx)
                    ),
                    "deployment_queue_pressure": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "queue_pressure", service_idx)
                    ),
                    "deployment_runtime_unknown_risk": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_unknown_risk", service_idx)
                    ),
                    "deployment_runtime_stale_risk": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_stale_risk", service_idx)
                    ),
                    "deployment_runtime_relative_weakness": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_relative_weakness", service_idx)
                    ),
                    "deployment_qk_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_term", service_idx)
                    ),
                    "deployment_pair_adjustment_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "pair_adjustment_term", service_idx)
                    ),
                    "deployment_quality_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "quality_term", service_idx)
                    ),
                    "deployment_confidence_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "confidence_term", service_idx)
                    ),
                    "deployment_service_pressure_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "service_pressure_term", service_idx)
                    ),
                    "deployment_inertia_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "inertia_term", service_idx)
                    ),
                    "deployment_unknown_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "unknown_penalty_term", service_idx)
                    ),
                    "deployment_stale_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "stale_penalty_term", service_idx)
                    ),
                    "deployment_runtime_risk_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_risk_penalty_term", service_idx)
                    ),
                    "deployment_low_quality_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "low_quality_penalty_term", service_idx)
                    ),
                    "deployment_queue_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "queue_penalty_term", service_idx)
                    ),
                    "deployment_memory_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "memory_penalty_term", service_idx)
                    ),
                    "deployment_device_load_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "device_load_penalty_term", service_idx)
                    ),
                    "deployment_hotspot_penalty_term": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "hotspot_penalty_term", service_idx)
                    ),
                    "deployment_policy_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "policy_prob", service_idx)
                    ),
                    "deployment_raw_threshold_nodes": self._json_for_record(
                        self._device_names_from_indices(
                            torch.nonzero(
                                actor_debug.get("raw_threshold_mask", torch.zeros_like(exec_mask))[service_idx]
                                .detach().cpu().bool(),
                                as_tuple=False,
                            ).flatten().tolist()
                        ) if isinstance(actor_debug.get("raw_threshold_mask"), torch.Tensor) else []
                    ),
                    "deployment_decoded_nodes": self._json_for_record(
                        self._device_names_from_indices(
                            torch.nonzero(decoded_row, as_tuple=False).flatten().tolist()
                        )
                    ),
                    "deployment_positive_mask": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "positive_mask", service_idx)
                    ),
                    "deployment_negative_mask": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "negative_mask", service_idx)
                    ),
                    "deployment_service_pressure": self._json_for_record(
                        self._actor_debug_vector_value(actor_debug, "service_pressure", service_idx)
                    ),
                    "deployment_edge_replica_counts": self._json_for_record(
                        self._actor_debug_vector_value(actor_debug, "edge_replica_count", service_idx)
                    ),
                    "deployment_device_replica_counts": self._json_for_record(
                        self._actor_debug_device_vector_map(actor_debug, "device_replica_count")
                    ),
                    "deployment_active_pair_hotspots": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "active_pair_hotspot", service_idx)
                    ),
                    "deployment_static_mask": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "static_mask", service_idx)
                    ),
                })
            self.dep_decision_recorder.log_dict(row)

    def _log_offloading_decisions(
            self,
            *,
            step: int,
            actions: torch.Tensor,
            static_mask: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            proposal_actions: Optional[torch.Tensor] = None,
            projection_reasons: Optional[List[str]] = None,
            actor_debug: Optional[Dict[str, Any]] = None,
            policy_deterministic: Optional[bool] = None,
    ) -> None:
        if self.off_decision_recorder is None or self.physical_topology is None:
            return

        actions = actions.detach().cpu().long()
        if proposal_actions is None:
            proposal_actions = actions
        else:
            proposal_actions = proposal_actions.detach().cpu().long()
        projection_reasons = projection_reasons or ["none" for _ in range(actions.numel())]
        static_mask = static_mask.detach().cpu().bool()
        parents = [[] for _ in range(actions.numel())]
        if self.logical_topology is not None:
            for parent, child in self.logical_topology.links:
                parents[child].append(parent)

        cloud_idx = self.physical_topology.cloud_idx
        for service_idx in range(actions.numel()):
            target_idx = int(actions[service_idx].item())
            proposal_idx = int(proposal_actions[service_idx].item())
            feasible_row = static_mask[service_idx].clone()
            if not feasible_row.any():
                feasible_row[cloud_idx] = True
            feasible_indices = torch.nonzero(feasible_row, as_tuple=False).flatten().tolist()
            parent_target_names = [
                self._device_name(int(actions[parent].item()))
                for parent in parents[service_idx]
            ]

            row = dict(
                step=step,
                epoch=self._epoch,
                update_steps=self._offloading_update_steps,
                service_idx=service_idx,
                service=self._service_name(service_idx),
                service_model_flops=self._latest_feature_value(logic_feats, "model_flops", service_idx),
                service_model_mem=self._latest_feature_value(logic_feats, "model_mem", service_idx),
                state_complexity=self._latest_feature_value(logic_feats, "task_complexity_seq", service_idx),
                proposal_target=self._device_name(proposal_idx) if 0 <= proposal_idx < len(self.physical_topology) else "",
                target=self._device_name(target_idx),
                projected=bool(proposal_idx != target_idx),
                projection_reason=(
                    projection_reasons[service_idx]
                    if service_idx < len(projection_reasons) else "none"
                ),
                is_cloud=bool(target_idx == cloud_idx),
                feasible_targets=self._json_for_record(self._device_names_from_indices(feasible_indices)),
                feasible_target_count=len(feasible_indices),
                parent_targets=self._json_for_record(parent_target_names),
                target_gpu_flops=self._latest_feature_value(phys_feats, "gpu_flops", target_idx),
                target_bandwidth=self._latest_feature_value(phys_feats, "bandwidth_latest", target_idx),
                target_role=self._latest_feature_value(phys_feats, "role_id", target_idx),
                target_qk_feature=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "qk_feature"
                ),
                target_compute_gap=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "compute_gap"
                ),
                target_arrival_rate_short=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "arrival_rate_short"
                ),
                target_runtime_ratio=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "runtime_ratio"
                ),
                target_runtime_confidence=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "runtime_confidence"
                ),
                target_runtime_recency=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "runtime_recency"
                ),
                target_queue_freshness=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "queue_freshness"
                ),
                target_speed_evidence=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "speed_evidence"
                ),
                target_capacity_pressure=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "capacity_pressure"
                ),
                target_pair_load=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "pair_load"
                ),
                target_device_load=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "device_load"
                ),
                target_service_time_factor=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "service_time_factor"
                ),
                target_cross_tier_penalty=self._actor_debug_candidate_value(
                    actor_debug, service_idx, target_idx, "cross_tier_penalty"
                ),
                target_static_prior=self._actor_debug_matrix_value(
                    actor_debug, "static_prior", service_idx, target_idx
                ),
                target_runtime_risk=self._actor_debug_matrix_value(
                    actor_debug, "runtime_risk", service_idx, target_idx
                ),
                target_load_pressure=self._actor_debug_matrix_value(
                    actor_debug, "load_pressure", service_idx, target_idx
                ),
                target_base_queue_risk=self._actor_debug_matrix_value(
                    actor_debug, "base_queue_risk", service_idx, target_idx
                ),
                target_relative_queue_risk=self._actor_debug_matrix_value(
                    actor_debug, "relative_queue_risk", service_idx, target_idx
                ),
                target_overload_risk=self._actor_debug_matrix_value(
                    actor_debug, "overload_risk", service_idx, target_idx
                ),
                target_queue_risk_total=self._actor_debug_matrix_value(
                    actor_debug, "queue_risk_total", service_idx, target_idx
                ),
                target_planned_pressure=self._actor_debug_matrix_value(
                    actor_debug, "planned_pressure", service_idx, target_idx
                ),
                target_planned_load_risk=self._actor_debug_matrix_value(
                    actor_debug, "planned_load_risk", service_idx, target_idx
                ),
                target_relative_planned_load_risk=self._actor_debug_matrix_value(
                    actor_debug, "relative_planned_load_risk", service_idx, target_idx
                ),
                target_offered_load_pressure=self._actor_debug_matrix_value(
                    actor_debug, "offered_load_pressure", service_idx, target_idx
                ),
                target_offered_load_risk=self._actor_debug_matrix_value(
                    actor_debug, "offered_load_risk", service_idx, target_idx
                ),
                target_compute_relative_weakness=self._actor_debug_matrix_value(
                    actor_debug, "compute_relative_weakness", service_idx, target_idx
                ),
                target_runtime_relative_weakness=self._actor_debug_matrix_value(
                    actor_debug, "runtime_relative_weakness", service_idx, target_idx
                ),
                target_relative_weakness=self._actor_debug_matrix_value(
                    actor_debug, "relative_weakness", service_idx, target_idx
                ),
                target_weak_pressure=self._actor_debug_matrix_value(
                    actor_debug, "weak_pressure", service_idx, target_idx
                ),
                target_weak_replica_risk=self._actor_debug_matrix_value(
                    actor_debug, "weak_replica_risk", service_idx, target_idx
                ),
                target_cloud_penalty=self._actor_debug_matrix_value(
                    actor_debug, "cloud_penalty", service_idx, target_idx
                ),
                target_cross_tier_penalty_term=self._actor_debug_matrix_value(
                    actor_debug, "cross_tier_penalty_term", service_idx, target_idx
                ),
                target_dynamic_risk=self._actor_debug_matrix_value(
                    actor_debug, "dynamic_risk", service_idx, target_idx
                ),
                target_planned_device_load=self._actor_debug_matrix_value(
                    actor_debug, "planned_device_load", service_idx, target_idx
                ),
                target_final_score=self._actor_debug_matrix_value(
                    actor_debug, "final_score", service_idx, target_idx
                ),
                offloading_policy_deterministic=(
                    "" if policy_deterministic is None else int(bool(policy_deterministic))
                ),
            )
            if self.record_cfg.decision_candidate_features_debug:
                row.update({
                    "offloading_service_demand_features": self._json_for_record(
                        self._service_feature_vector(
                            logic_feats,
                            "service_demand_feat",
                            service_idx,
                            names=SERVICE_DEMAND_FEATURE_NAMES,
                        )
                    ),
                    "offloading_device_capability_features": self._json_for_record(
                        self._device_feature_map(
                            phys_feats,
                            "device_capability_feat",
                            names=DEVICE_CAPABILITY_FEATURE_NAMES,
                        )
                    ),
                    "offloading_runtime_pair_features": self._json_for_record(
                        self._pair_feature_map(
                            logic_feats,
                            "runtime_pair_feat",
                            service_idx,
                            names=RUNTIME_PAIR_FEATURE_NAMES,
                        )
                    ),
                    "offloading_candidate_features": self._json_for_record(
                        self._actor_debug_pair_feature_map(actor_debug, "candidate_feature", service_idx)
                    ),
                })
            if self.record_cfg.decision_actor_debug:
                row.update({
                    "offloading_qk_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_score", service_idx)
                    ),
                    "offloading_qk_features": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_feature", service_idx)
                    ),
                    "offloading_static_priors": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "static_prior", service_idx)
                    ),
                    "offloading_runtime_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_risk", service_idx)
                    ),
                    "offloading_service_time_factors": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "service_time_factor", service_idx)
                    ),
                    "offloading_runtime_recencies": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_recency", service_idx)
                    ),
                    "offloading_queue_freshnesses": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "queue_freshness", service_idx)
                    ),
                    "offloading_speed_evidences": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "speed_evidence", service_idx)
                    ),
                    "offloading_capacity_pressures": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "capacity_pressure", service_idx)
                    ),
                    "offloading_load_pressures": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "load_pressure", service_idx)
                    ),
                    "offloading_base_queue_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "base_queue_risk", service_idx)
                    ),
                    "offloading_relative_queue_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "relative_queue_risk", service_idx)
                    ),
                    "offloading_overload_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "overload_risk", service_idx)
                    ),
                    "offloading_queue_risk_totals": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "queue_risk_total", service_idx)
                    ),
                    "offloading_planned_pressures": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "planned_pressure", service_idx)
                    ),
                    "offloading_planned_load_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "planned_load_risk", service_idx)
                    ),
                    "offloading_relative_planned_load_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "relative_planned_load_risk", service_idx)
                    ),
                    "offloading_offered_load_pressures": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "offered_load_pressure", service_idx)
                    ),
                    "offloading_offered_load_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "offered_load_risk", service_idx)
                    ),
                    "offloading_compute_relative_weaknesses": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "compute_relative_weakness", service_idx)
                    ),
                    "offloading_runtime_relative_weaknesses": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "runtime_relative_weakness", service_idx)
                    ),
                    "offloading_relative_weaknesses": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "relative_weakness", service_idx)
                    ),
                    "offloading_weak_pressures": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "weak_pressure", service_idx)
                    ),
                    "offloading_weak_replica_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "weak_replica_risk", service_idx)
                    ),
                    "offloading_cloud_penalties": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "cloud_penalty", service_idx)
                    ),
                    "offloading_cross_tier_penalty_terms": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "cross_tier_penalty_term", service_idx)
                    ),
                    "offloading_dynamic_risks": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "dynamic_risk", service_idx)
                    ),
                    "offloading_planned_device_loads": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "planned_device_load", service_idx)
                    ),
                    "offloading_final_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "final_score", service_idx)
                    ),
                    "offloading_base_policy_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "base_policy_prob", service_idx)
                    ),
                    "offloading_unknown_policy_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "unknown_policy_prob", service_idx)
                    ),
                    "offloading_unknown_exploration_weights": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "unknown_exploration_weight", service_idx)
                    ),
                    "offloading_unknown_exploration_eps": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "unknown_exploration_eps", service_idx)
                    ),
                    "offloading_risk_policy_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "risk_policy_prob", service_idx)
                    ),
                    "offloading_risk_exploration_weights": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "risk_exploration_weight", service_idx)
                    ),
                    "offloading_risk_exploration_eps": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "risk_exploration_eps", service_idx)
                    ),
                    "offloading_policy_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "policy_prob", service_idx)
                    ),
                    "offloading_effective_mask": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "effective_mask", service_idx)
                    ),
                })
            self.off_decision_recorder.log_dict(row)

    def register_topology_encoder(self):
        if self.shared_topology_encoder:
            return

        self.shared_topology_encoder = TopologyEncoders(
            d_model=self.encoder_cfg.embedding_dim,
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
            unknown_exploration_prob=self.offloading_agent_params['unknown_exploration_prob'],
            risk_exploration_prob=self.offloading_agent_params["risk_exploration_prob"],
            risk_exploration_temperature=self.offloading_agent_params["risk_exploration_temperature"],
            risk_exploration_min_gap=self.offloading_agent_params["risk_exploration_min_gap"],
            static_prior_scale=self.offloading_agent_params["score_static_prior_scale"],
            runtime_weight=self.offloading_agent_params["score_runtime_weight"],
            runtime_clip=self.offloading_agent_params["score_runtime_clip"],
            absolute_queue_weight=self.offloading_agent_params["score_absolute_queue_weight"],
            relative_queue_weight=self.offloading_agent_params["score_relative_queue_weight"],
            overload_weight=self.offloading_agent_params["score_overload_weight"],
            planned_load_weight=self.offloading_agent_params["score_planned_load_weight"],
            relative_planned_load_weight=self.offloading_agent_params["score_relative_planned_load_weight"],
            offered_load_weight=self.offloading_agent_params["score_offered_load_weight"],
            offered_load_clip=self.offloading_agent_params["score_offered_load_clip"],
            weak_replica_weight=self.offloading_agent_params["score_weak_replica_weight"],
            weak_gap_clip=self.offloading_agent_params["score_weak_gap_clip"],
            runtime_weak_gap_clip=self.offloading_agent_params["score_runtime_weak_gap_clip"],
            runtime_weakness_min_confidence=self.offloading_agent_params["score_runtime_weakness_min_confidence"],
            runtime_recency_floor=self.offloading_agent_params["score_runtime_recency_floor"],
            weak_service_time_weight=self.offloading_agent_params["score_weak_service_time_weight"],
            weak_capacity_weight=self.offloading_agent_params["score_weak_capacity_weight"],
            weak_queue_amplifier=self.offloading_agent_params["score_weak_queue_amplifier"],
            cloud_fallback_penalty=self.offloading_agent_params["score_cloud_fallback_penalty"],
            cross_tier_weight=self.offloading_agent_params["score_cross_tier_weight"],
            planned_load_clip=self.offloading_agent_params["score_planned_load_clip"],
            load_clip=self.offloading_agent_params["score_load_clip"],
            risk_clip=self.offloading_agent_params["score_risk_clip"],
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
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
            self._maybe_autostart_deployment_offline("deployment offline state buffer already registered")
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
        self._maybe_autostart_deployment_offline("deployment offline state buffer registration")

    def register_initial_deployment(self, deployment_plan):
        assert self.logical_topology, "Logical topology must be registered before registering initial deployment."
        assert self.physical_topology, "Physical topology must be registered before registering initial deployment."

        if self.initial_deployment_plan is not None:
            self._maybe_autostart_deployment_offline("deployment offline initial deployment already registered")
            return
        self.initial_deployment_plan = copy.deepcopy(deployment_plan)
        self.deployment_plan = copy.deepcopy(deployment_plan)
        self.pending_deployment_plan = None
        self.pending_deploy_mask = None
        self._pending_deployment_force_serve = False
        self._pending_deployment_reason = None

        # Cache the current deployment mask.
        self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(deployment_plan or {})
        LOGGER.info(
            f"[Hedger][Deployment] Registered initial deployment: "
            f"{self._summarize_deployment_plan(self.initial_deployment_plan)}"
        )
        self._maybe_autostart_deployment_offline("deployment offline initial deployment registration")
        LOGGER.debug(f"[Hedger][Deployment] Initial deployment detail: {self.initial_deployment_plan}")

    def get_offloading_plan(self):
        if self.should_force_default_decisions():
            return None
        with self._data_lock:
            return copy.deepcopy(self.offloading_plan)

    def get_initial_deployment_plan(self):
        return copy.deepcopy(self.initial_deployment_plan)

    def get_active_deployment_version(self) -> int:
        with self._deployment_version_cond:
            return int(self._deployment_served_version)

    def get_redeployment_plan(self):
        if self.should_force_default_decisions():
            if self.latency_guard_cfg.force_default_decisions:
                self._apply_default_decisions_for_latency_guard("redeployment request")
            with self._data_lock:
                plan = copy.deepcopy(self.deployment_plan or self.initial_deployment_plan)
            if plan is not None:
                LOGGER.warning(
                    "[Hedger][Deployment] Latency guard is active; "
                    "serve default deployment plan instead of learned redeployment."
                )
                return plan

        now = time.monotonic()
        served_version = None
        throttled = False
        with self._data_lock:
            has_pending = (
                self._deployment_decision_version > self._deployment_served_version
                and self.pending_deployment_plan is not None
                and self.pending_deploy_mask is not None
            )
            if has_pending:
                elapsed = (
                    float("inf")
                    if self._last_deployment_served_monotonic <= 0.0
                    else now - self._last_deployment_served_monotonic
                )
                force_serve = bool(getattr(self, "_pending_deployment_force_serve", False))
                pending_reason = getattr(self, "_pending_deployment_reason", None)
                if elapsed < self.deployment_interval and not force_serve:
                    throttled = True
                else:
                    self.deployment_plan = copy.deepcopy(self.pending_deployment_plan)
                    self.cur_deploy_mask = self.pending_deploy_mask.detach().clone().cpu()
                    self.pending_deployment_plan = None
                    self.pending_deploy_mask = None
                    self._pending_deployment_force_serve = False
                    self._pending_deployment_reason = None
                    self._last_deployment_served_monotonic = now
                    served_version = self._mark_deployment_decision_served()
                    if force_serve:
                        LOGGER.warning(
                            f"[Hedger][Deployment] Force serving deployment decision: "
                            f"version={served_version}, reason={pending_reason}, "
                            f"elapsed={self._format_log_value(elapsed, 2)}s, "
                            f"configured_interval={self._format_log_value(self.deployment_interval, 2)}s."
                        )

            if self.deployment_plan is None and self.cur_deploy_mask is not None:
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)
            elif self.cur_deploy_mask is None and self.deployment_plan is not None:
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan)

            plan = copy.deepcopy(self.deployment_plan)

        if throttled:
            if now - self._last_redeployment_throttle_log_t >= 10.0:
                self._last_redeployment_throttle_log_t = now
                LOGGER.debug(
                    f"[Hedger][Deployment] Redeployment request throttled: "
                    f"served_version={self._deployment_served_version}, "
                    f"pending_version={self._deployment_decision_version}, "
                    f"min_interval={self._format_log_value(self.deployment_interval, 2)}s"
                )

        if served_version is not None:
            LOGGER.info(
                f"[Hedger][Deployment] Served deployment decision version={served_version}; "
                f"feedback window reset."
            )
        return plan

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

    def _ensure_deployment_collect_anchor(
            self,
            prev_deploy_mask: Optional[torch.Tensor],
            static_allowed: torch.Tensor,
    ) -> torch.Tensor:
        if self._deployment_collect_anchor_mask is None:
            if prev_deploy_mask is None:
                anchor = self._current_deploy_mask()
            else:
                anchor = prev_deploy_mask.detach().clone().cpu().bool()
            self._deployment_collect_anchor_mask = anchor

        anchor = self._deployment_collect_anchor_mask.detach().clone().to(self.device).bool()
        if tuple(anchor.shape) != tuple(static_allowed.shape):
            anchor = self._current_deploy_mask().to(self.device).bool()
            self._deployment_collect_anchor_mask = anchor.detach().cpu()
        anchor = torch.where(static_allowed.bool(), anchor, torch.zeros_like(anchor))
        anchor[:, self.physical_topology.cloud_idx] = True
        return anchor

    def _deployment_collect_base_mask(
            self,
            prev_deploy_mask: Optional[torch.Tensor],
            static_allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool, bool]:
        cfg = self.training_cfg.deployment_collect
        anchor = self._ensure_deployment_collect_anchor(prev_deploy_mask, static_allowed)
        should_reset = (
            cfg.reset_on_bad_streak
            and self._deployment_collect_bad_streak >= cfg.bad_streak_threshold
            and random.random() <= cfg.reset_to_anchor_prob
        )
        if should_reset:
            return anchor, True, True

        if prev_deploy_mask is None:
            base = anchor
            anchor_used = True
        else:
            base = prev_deploy_mask.detach().clone().to(self.device).bool()
            anchor_used = False
        base = torch.where(static_allowed.bool(), base, torch.zeros_like(base))
        base[:, self.physical_topology.cloud_idx] = True
        return base, anchor_used, False

    def _sample_deployment_action_for_training(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_edge_index: torch.Tensor,
            phys_feats: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor],
    ):
        mode = self.stage_cfg.deployment_train_mode if self.stage_cfg is not None else "ppo"
        if mode == "collect":
            return self._sample_deployment_collection_action(
                logic_edge_index,
                logic_feats,
                phys_edge_index,
                phys_feats,
                prev_deploy_mask,
            )
        deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
            logic_edge_index=logic_edge_index,
            logic_feats=logic_feats,
            phys_edge_index=phys_edge_index,
            phys_feats=phys_feats,
            topo_order=None,
            prev_deploy_mask=prev_deploy_mask,
            deterministic=self.training_cfg.deployment_rollout_deterministic,
        )
        aux["behavior_kind"] = "actor"
        return deploy_mask, logp, ent, value, aux

    def _sample_deployment_collection_action(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_edge_index: torch.Tensor,
            phys_feats: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor],
    ):
        cfg = self.training_cfg.deployment_collect
        static_allowed = self.deployment_agent._static_allowed_mask(phys_feats, logic_feats)
        base_mask, anchor_used, reset_triggered = self._deployment_collect_base_mask(
            prev_deploy_mask,
            static_allowed,
        )
        reject_reasons: List[str] = []
        attempts = max(1, int(cfg.max_action_attempts))
        best_soft_candidate = None
        best_soft_score = float("inf")

        for attempt_idx in range(attempts):
            roll = random.random()
            collect_debug = {
                "collect_anchor_used": int(anchor_used),
                "collect_reset_triggered": int(reset_triggered),
                "collect_bad_streak_before": int(self._deployment_collect_bad_streak),
            }
            if roll < cfg.actor_prob:
                deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats,
                    phys_edge_index=phys_edge_index,
                    phys_feats=phys_feats,
                    topo_order=None,
                    prev_deploy_mask=prev_deploy_mask,
                    deterministic=self.training_cfg.deployment_rollout_deterministic,
                    logit_noise_std=cfg.actor_logit_noise_std,
                )
                collect_debug.update({
                    "collect_behavior": "actor",
                    "collect_operation": "actor_noise" if cfg.actor_logit_noise_std > 0.0 else "actor",
                })
            elif roll < cfg.actor_prob + cfg.perturb_prob:
                raw_mask, collect_debug = self._sample_safe_collect_edit(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats,
                    phys_feats=phys_feats,
                    static_allowed=static_allowed,
                    base_mask=base_mask,
                    collect_debug=collect_debug,
                )
                deploy_mask, logp, ent, value, aux = self._finalize_collect_raw_mask(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats,
                    phys_edge_index=phys_edge_index,
                    phys_feats=phys_feats,
                    prev_deploy_mask=prev_deploy_mask,
                    static_allowed=static_allowed,
                    raw_mask=raw_mask,
                    behavior_kind=str(collect_debug.get("collect_operation", "safe_edit")),
                )
            else:
                raw_mask = base_mask.detach().clone()
                collect_debug.update({
                    "collect_behavior": "keep",
                    "collect_operation": "anchor_reset" if reset_triggered else "keep",
                })
                deploy_mask, logp, ent, value, aux = self._finalize_collect_raw_mask(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats,
                    phys_edge_index=phys_edge_index,
                    phys_feats=phys_feats,
                    prev_deploy_mask=prev_deploy_mask,
                    static_allowed=static_allowed,
                    raw_mask=raw_mask,
                    behavior_kind=str(collect_debug["collect_operation"]),
                )

            risk_debug = self._deployment_collect_candidate_risk(
                logic_edge_index=logic_edge_index,
                logic_feats=logic_feats,
                phys_feats=phys_feats,
                static_allowed=static_allowed,
                base_mask=base_mask,
                raw_mask=aux["raw_deploy_mask"],
                deploy_mask=deploy_mask,
                capacity_relax_cost=float(aux.get("capacity_relax_cost", 0.0)),
                edge_cover_unmet=int(aux.get("edge_cover_unmet", 0)),
            )
            collect_debug.update(risk_debug)
            accepted, reason = self._deployment_collect_candidate_accepted(collect_debug)
            if accepted:
                aux.update(collect_debug)
                aux.update({
                    "collect_attempts": attempt_idx + 1,
                    "collect_reject_cnt": len(reject_reasons),
                    "collect_reject_reasons": ";".join(reject_reasons),
                    "collect_fallback_selected_best": 0,
                    "behavior_kind": str(collect_debug.get("collect_operation", aux.get("behavior_kind", "collect"))),
                })
                return deploy_mask, logp, ent, value, aux
            hard_reason = str(collect_debug.get("collect_hard_reject_reason", "") or "")
            if not hard_reason and cfg.fallback_to_best_candidate:
                candidate_score_raw = collect_debug.get("collect_predicted_risk", float("inf"))
                candidate_score = float(candidate_score_raw) if candidate_score_raw is not None else float("inf")
                quality_gain = float(collect_debug.get("collect_delta_min_pair_quality", 0.0) or 0.0)
                risk_drop = max(0.0, -float(collect_debug.get("collect_delta_runtime_risk", 0.0) or 0.0))
                queue_drop = max(0.0, -float(collect_debug.get("collect_delta_queue_pressure", 0.0) or 0.0))
                hotspot_drop = max(0.0, -float(collect_debug.get("collect_delta_hotspot_cost", 0.0) or 0.0))
                adjusted_score = candidate_score - 0.25 * (max(0.0, quality_gain) + risk_drop + queue_drop + hotspot_drop)
                if adjusted_score < best_soft_score:
                    best_soft_score = adjusted_score
                    best_soft_candidate = (
                        deploy_mask,
                        logp,
                        ent,
                        value,
                        dict(aux),
                        dict(collect_debug),
                    )
            reject_reasons.append(reason)

        if best_soft_candidate is not None:
            deploy_mask, logp, ent, value, aux, collect_debug = best_soft_candidate
            collect_debug = dict(collect_debug)
            collect_debug["collect_behavior"] = "best_soft_fallback"
            collect_debug["collect_operation"] = "best_" + str(
                collect_debug.get("collect_operation", "soft_candidate")
            )
            collect_debug["collect_fallback_selected_best"] = 1
            aux.update(collect_debug)
            aux.update({
                "collect_attempts": attempts,
                "collect_reject_cnt": len(reject_reasons),
                "collect_reject_reasons": ";".join(reject_reasons),
                "behavior_kind": str(collect_debug.get("collect_operation", "best_soft_fallback")),
            })
            return deploy_mask, logp, ent, value, aux

        raw_mask = base_mask.detach().clone()
        fallback_operation = "fallback_anchor" if anchor_used or reset_triggered else "fallback_keep"
        deploy_mask, logp, ent, value, aux = self._finalize_collect_raw_mask(
            logic_edge_index=logic_edge_index,
            logic_feats=logic_feats,
            phys_edge_index=phys_edge_index,
            phys_feats=phys_feats,
            prev_deploy_mask=prev_deploy_mask,
            static_allowed=static_allowed,
            raw_mask=raw_mask,
            behavior_kind=fallback_operation,
        )
        collect_debug = {
            "collect_behavior": "fallback",
            "collect_operation": fallback_operation,
            "collect_anchor_used": int(anchor_used or reset_triggered),
            "collect_reset_triggered": int(reset_triggered),
            "collect_bad_streak_before": int(self._deployment_collect_bad_streak),
            "collect_attempts": attempts,
            "collect_reject_cnt": len(reject_reasons),
            "collect_reject_reasons": ";".join(reject_reasons),
            "collect_fallback_selected_best": 0,
        }
        collect_debug.update(self._deployment_collect_candidate_risk(
            logic_edge_index=logic_edge_index,
            logic_feats=logic_feats,
            phys_feats=phys_feats,
            static_allowed=static_allowed,
            base_mask=base_mask,
            raw_mask=aux["raw_deploy_mask"],
            deploy_mask=deploy_mask,
            capacity_relax_cost=float(aux.get("capacity_relax_cost", 0.0)),
            edge_cover_unmet=int(aux.get("edge_cover_unmet", 0)),
        ))
        aux.update(collect_debug)
        aux["behavior_kind"] = fallback_operation
        return deploy_mask, logp, ent, value, aux

    def _finalize_collect_raw_mask(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_edge_index: torch.Tensor,
            phys_feats: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor],
            static_allowed: torch.Tensor,
            raw_mask: torch.Tensor,
            behavior_kind: str,
    ):
        cloud_idx = self.physical_topology.cloud_idx
        raw_mask = torch.where(static_allowed.bool(), raw_mask.bool(), torch.zeros_like(raw_mask.bool()))
        raw_mask[:, cloud_idx] = True
        raw_probs = torch.full(raw_mask.shape, 0.15, dtype=torch.float32, device=self.device)
        raw_probs = torch.where(raw_mask, torch.full_like(raw_probs, 0.85), raw_probs)
        raw_probs = torch.where(static_allowed, raw_probs, torch.zeros_like(raw_probs))
        raw_probs[:, cloud_idx] = 1.0
        (
            deploy_mask,
            capacity_relax_cnt,
            capacity_relax_cost,
            edge_cover_repair_cnt,
            edge_cover_repair_cost,
            edge_cover_unmet,
            hotspot_repair_cnt,
            hotspot_repair_cost,
            hotspot_unmet,
            risk_metrics,
            _executed_pair_ctx,
            capacity_removed_mask,
        ) = self.deployment_agent._project_deployment_mask(
            raw_mask,
            raw_probs=raw_probs,
            logic_feats=logic_feats,
            phys_feats=phys_feats,
            logic_edge_index=logic_edge_index,
            prev_deploy_mask=prev_deploy_mask,
            static_allowed=static_allowed,
        )
        logp, ent, value = self.deployment_agent.evaluate(
            logic_edge_index,
            logic_feats,
            phys_edge_index,
            phys_feats,
            deploy_mask,
            prev_deploy_mask=prev_deploy_mask,
            topo_order=None,
        )
        return deploy_mask, logp, ent, value, {
            "capacity_relax_cnt": capacity_relax_cnt,
            "capacity_relax_cost": capacity_relax_cost,
            "edge_cover_repair_cnt": edge_cover_repair_cnt,
            "edge_cover_repair_cost": edge_cover_repair_cost,
            "edge_cover_unmet": edge_cover_unmet,
            "hotspot_repair_cnt": hotspot_repair_cnt,
            "hotspot_repair_cost": hotspot_repair_cost,
            "hotspot_unmet": hotspot_unmet,
            **risk_metrics,
            "raw_deploy_mask": raw_mask,
            "capacity_removed_mask": capacity_removed_mask,
            "capacity_removed_cnt": int(capacity_removed_mask[:, :cloud_idx].sum().detach().cpu().item())
            if cloud_idx > 0 else 0,
            "behavior_kind": behavior_kind,
        }

    def _sample_safe_collect_edit(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            base_mask: torch.Tensor,
            collect_debug: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw_mask = base_mask.detach().clone().bool()
        collect_debug = dict(collect_debug)
        collect_debug["collect_behavior"] = "safe_edit"

        for operation in self._deployment_collect_operation_order():
            edited_mask, operation_debug = self._try_deployment_collect_operation(
                operation=operation,
                logic_edge_index=logic_edge_index,
                logic_feats=logic_feats,
                phys_feats=phys_feats,
                static_allowed=static_allowed,
                base_mask=raw_mask,
            )
            if edited_mask is None:
                continue
            collect_debug.update(operation_debug)
            return edited_mask, collect_debug

        collect_debug.update({
            "collect_operation": "safe_edit_noop",
            "collect_selected_service_idx": -1,
            "collect_selected_device_idx": -1,
            "collect_removed_service_idx": -1,
            "collect_removed_device_idx": -1,
        })
        return raw_mask, collect_debug

    def _deployment_collect_operation_order(self) -> List[str]:
        cfg = self.training_cfg.deployment_collect
        roll = random.random()
        if roll < cfg.safe_add_prob:
            first = "safe_add"
        elif roll < cfg.safe_add_prob + cfg.safe_swap_prob:
            first = "safe_swap"
        else:
            first = "safe_remove"
        order = [first]
        for operation in ("safe_add", "safe_swap", "safe_remove"):
            if operation not in order:
                order.append(operation)
        return order

    def _try_deployment_collect_operation(
            self,
            operation: str,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            base_mask: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        cloud_idx = self.physical_topology.cloud_idx
        if cloud_idx <= 0:
            return None, {}

        pair_ctx = self.deployment_agent._deployment_pair_context(
            logic_edge_index,
            logic_feats,
            static_allowed,
            deploy_mask=base_mask,
        )
        pressure_matrix = self._deployment_collect_pair_pressure_matrix(
            logic_feats,
            static_allowed,
            base_mask.device,
        )
        quality_matrix = self._deployment_collect_pair_quality_matrix(
            logic_feats,
            phys_feats,
            static_allowed,
            pressure_matrix,
        )
        service_pressure = pair_ctx["service_pressure"].detach().float()
        active_hotspot = pair_ctx["active_pair_hotspot"].detach().float()
        edge_counts = base_mask[:, :cloud_idx].float().sum(dim=1)
        feasible_counts = static_allowed[:, :cloud_idx].float().sum(dim=1)

        candidates: List[Tuple[float, int, int, int]] = []
        if operation == "safe_add":
            for service_idx in range(base_mask.size(0)):
                feasible_target = min(2.0, float(feasible_counts[service_idx].item()))
                current_edges = float(edge_counts[service_idx].item())
                if current_edges >= feasible_target or feasible_target <= 0.0:
                    continue
                if current_edges > 0.0 and float(service_pressure[service_idx].item()) < 0.40:
                    continue
                for device_idx in range(cloud_idx):
                    if bool(base_mask[service_idx, device_idx].item()):
                        continue
                    if not self._deployment_collect_pair_capacity_ok(
                            base_mask,
                            service_idx,
                            device_idx,
                            logic_feats,
                            phys_feats,
                            static_allowed,
                    ):
                        continue
                    score = (
                        float(quality_matrix[service_idx, device_idx].item())
                        + 1.5 * float(service_pressure[service_idx].item())
                        + 0.30 * max(0.0, feasible_target - current_edges)
                    )
                    candidates.append((score, service_idx, device_idx, -1))
            if not candidates:
                return None, {}
            _, service_idx, device_idx, _ = max(candidates)
            raw_mask = base_mask.detach().clone()
            raw_mask[service_idx, device_idx] = True
            return raw_mask, {
                "collect_operation": "safe_add",
                "collect_selected_service_idx": int(service_idx),
                "collect_selected_device_idx": int(device_idx),
                "collect_removed_service_idx": -1,
                "collect_removed_device_idx": -1,
            }

        if operation == "safe_swap":
            for service_idx in range(base_mask.size(0)):
                for source_device in range(cloud_idx):
                    if not bool(base_mask[service_idx, source_device].item()):
                        continue
                    current_quality = float(quality_matrix[service_idx, source_device].item())
                    current_pressure = float(pressure_matrix[service_idx, source_device].item())
                    current_hotspot = float(active_hotspot[service_idx, source_device].item())
                    current_risk = current_pressure + current_hotspot - 0.25 * current_quality
                    for target_device in range(cloud_idx):
                        if target_device == source_device:
                            continue
                        if bool(base_mask[service_idx, target_device].item()):
                            continue
                        trial_mask = base_mask.detach().clone()
                        trial_mask[service_idx, source_device] = False
                        if not self._deployment_collect_pair_capacity_ok(
                                trial_mask,
                                service_idx,
                                target_device,
                                logic_feats,
                                phys_feats,
                                static_allowed,
                        ):
                            continue
                        target_quality = float(quality_matrix[service_idx, target_device].item())
                        target_pressure = float(pressure_matrix[service_idx, target_device].item())
                        improvement = target_quality - current_quality + current_risk - target_pressure
                        if improvement <= 0.0 and current_pressure < 0.10 and current_hotspot < 0.01:
                            continue
                        candidates.append((improvement, service_idx, target_device, source_device))
            if not candidates:
                return None, {}
            _, service_idx, target_device, source_device = max(candidates)
            raw_mask = base_mask.detach().clone()
            raw_mask[service_idx, source_device] = False
            raw_mask[service_idx, target_device] = True
            return raw_mask, {
                "collect_operation": "safe_swap",
                "collect_selected_service_idx": int(service_idx),
                "collect_selected_device_idx": int(target_device),
                "collect_removed_service_idx": int(service_idx),
                "collect_removed_device_idx": int(source_device),
            }

        if operation == "safe_remove":
            for service_idx in range(base_mask.size(0)):
                current_edges = int(edge_counts[service_idx].item())
                if current_edges <= 1 and not self.training_cfg.deployment_collect.allow_remove_last_edge_replica:
                    continue
                if float(service_pressure[service_idx].item()) >= 0.60 and current_edges <= 2:
                    continue
                for device_idx in range(cloud_idx):
                    if not bool(base_mask[service_idx, device_idx].item()):
                        continue
                    score = (
                        1.0 - float(service_pressure[service_idx].item())
                        + float(pressure_matrix[service_idx, device_idx].item())
                        - 0.20 * float(quality_matrix[service_idx, device_idx].item())
                    )
                    candidates.append((score, service_idx, -1, device_idx))
            if not candidates:
                return None, {}
            _, service_idx, _, source_device = max(candidates)
            raw_mask = base_mask.detach().clone()
            raw_mask[service_idx, source_device] = False
            raw_mask[:, cloud_idx] = True
            return raw_mask, {
                "collect_operation": "safe_remove",
                "collect_selected_service_idx": -1,
                "collect_selected_device_idx": -1,
                "collect_removed_service_idx": int(service_idx),
                "collect_removed_device_idx": int(source_device),
            }

        return None, {}

    def _deployment_collect_pair_capacity_ok(
            self,
            base_mask: torch.Tensor,
            service_idx: int,
            device_idx: int,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
    ) -> bool:
        cloud_idx = self.physical_topology.cloud_idx
        if device_idx < 0 or device_idx >= cloud_idx:
            return False
        if not bool(static_allowed[service_idx, device_idx].item()):
            return False

        candidate = base_mask.detach().clone().bool()
        candidate[service_idx, device_idx] = True
        model_mem = logic_feats["model_mem"].float().to(candidate.device)
        residual = self.deployment_agent._initial_residual_mem(
            phys_feats,
            logic_feats,
            None,
        ).to(candidate.device).float()
        max_edge_replicas = self.deployment_agent._max_edge_replicas_per_device()

        selected = candidate[:, device_idx].bool()
        if max_edge_replicas is not None and int(selected.sum().item()) > int(max_edge_replicas):
            return False
        used_mem = float(model_mem[selected].sum().item()) if selected.any() else 0.0
        return used_mem <= float(residual[device_idx].item()) + 1e-6

    def _deployment_collect_pair_pressure_matrix(
            self,
            logic_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            device: torch.device,
    ) -> torch.Tensor:
        num_services, num_devices = static_allowed.shape
        dtype = torch.float32
        queue_normalizer = max(1e-6, float(self.deployment_agent_params.get("queue_normalizer", 8.0)))
        pressure = torch.zeros((num_services, num_devices), dtype=dtype, device=device)

        runtime_feat = logic_feats.get("runtime_pair_feat")
        if isinstance(runtime_feat, torch.Tensor) and runtime_feat.dim() == 3 and runtime_feat.size(-1) >= 6:
            runtime_feat = runtime_feat.to(device=device, dtype=dtype)
            runtime_queue = (
                runtime_feat[..., 0].clamp_min(0.0)
                + 0.5 * runtime_feat[..., 1].clamp_min(0.0)
            ) * torch.maximum(runtime_feat[..., 5].clamp(0.0, 1.0), torch.full_like(runtime_feat[..., 5], 0.25))
            pressure = torch.maximum(pressure, (runtime_queue / queue_normalizer).clamp(0.0, 1.0))

        guard_pressure = torch.zeros_like(pressure)
        now = time.monotonic()
        with self._latency_guard_lock:
            observations = copy.deepcopy(getattr(self, "_latency_guard_queue_observations", {}))
        for device_name, record in observations.items():
            if not isinstance(record, dict):
                continue
            try:
                ts = float(record.get("ts", 0.0))
            except (TypeError, ValueError):
                continue
            if now - ts > self.latency_guard_cfg.queue_recovery_stale_timeout_s:
                continue
            values = record.get("values") or {}
            if not isinstance(values, dict):
                continue
            try:
                device_idx = self.physical_topology.index(str(device_name))
            except ValueError:
                continue
            if device_idx < 0 or device_idx >= num_devices:
                continue
            for service_name, value in values.items():
                try:
                    service_idx = self.logical_topology.index(str(service_name))
                    queue_length = max(0.0, float(value))
                except (TypeError, ValueError):
                    continue
                if service_idx < 0 or service_idx >= num_services:
                    continue
                guard_pressure[service_idx, device_idx] = max(
                    float(guard_pressure[service_idx, device_idx].item()),
                    min(queue_length / queue_normalizer, 1.0),
                )

        pressure = torch.maximum(pressure, guard_pressure)
        pressure = torch.where(static_allowed.to(device=device).bool(), pressure, torch.zeros_like(pressure))
        return pressure

    def _deployment_collect_pair_quality_matrix(
            self,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            pressure_matrix: torch.Tensor,
    ) -> torch.Tensor:
        num_services, num_devices = static_allowed.shape
        device = pressure_matrix.device
        dtype = torch.float32
        cloud_idx = self.physical_topology.cloud_idx
        device_cap = phys_feats.get("device_capability_feat")
        if isinstance(device_cap, torch.Tensor) and device_cap.dim() == 2 and device_cap.size(0) == num_devices \
                and device_cap.size(1) >= 3:
            relative_compute = device_cap[:, 2].to(device=device, dtype=dtype)
        else:
            gpu_flops = phys_feats.get("gpu_flops")
            if isinstance(gpu_flops, torch.Tensor) and gpu_flops.numel() == num_devices:
                gpu_flops = gpu_flops.to(device=device, dtype=dtype)
                relative_compute = gpu_flops / gpu_flops.max().clamp_min(1e-6)
            else:
                relative_compute = torch.zeros((num_devices,), device=device, dtype=dtype)

        runtime_feat = logic_feats.get("runtime_pair_feat")
        runtime_conf = torch.zeros((num_services, num_devices), device=device, dtype=dtype)
        runtime_time_norm = torch.zeros_like(runtime_conf)
        runtime_unknown = torch.zeros_like(runtime_conf)
        runtime_stale = torch.zeros_like(runtime_conf)
        if isinstance(runtime_feat, torch.Tensor) and runtime_feat.dim() == 3 and runtime_feat.size(-1) >= 6:
            runtime_feat = runtime_feat.to(device=device, dtype=dtype)
            observed_conf = runtime_feat[..., 3].clamp(0.0, 1.0)
            recency = runtime_feat[..., 4].clamp(0.0, 1.0)
            runtime_conf = observed_conf * recency
            runtime_unknown = (1.0 - observed_conf).clamp(0.0, 1.0)
            runtime_stale = (1.0 - recency).clamp(0.0, 1.0)
            runtime_time = runtime_feat[..., 2].clamp_min(0.0)
            valid_time = runtime_time.masked_select(static_allowed.to(device=device).bool())
            if valid_time.numel() > 0:
                runtime_time_norm = runtime_time / valid_time.max().clamp_min(1e-6)

        evidence_untrusted = torch.maximum(runtime_unknown, runtime_stale)
        quality = (
            relative_compute.view(1, num_devices).expand(num_services, -1)
            * runtime_conf
            * (1.0 - pressure_matrix.clamp(0.0, 1.0))
            * (1.0 - runtime_time_norm.clamp(0.0, 1.0))
            * (1.0 - evidence_untrusted).clamp(0.0, 1.0)
        )
        quality = torch.where(
            static_allowed.to(device=device).bool(),
            quality,
            torch.full_like(quality, -1e6),
        )
        if 0 <= cloud_idx < num_devices:
            quality[:, cloud_idx] = -1e6
        return quality

    def _deployment_collect_candidate_risk(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            base_mask: torch.Tensor,
            raw_mask: torch.Tensor,
            deploy_mask: torch.Tensor,
            capacity_relax_cost: float,
            edge_cover_unmet: int,
    ) -> Dict[str, Any]:
        cfg = self.training_cfg.deployment_collect
        cloud_idx = self.physical_topology.cloud_idx
        device = deploy_mask.device
        pressure_matrix = self._deployment_collect_pair_pressure_matrix(
            logic_feats,
            static_allowed,
            device,
        )
        num_services = deploy_mask.size(0)

        def _edge_mask(mask: torch.Tensor) -> torch.Tensor:
            if cloud_idx <= 0:
                return torch.zeros((num_services, 0), dtype=torch.bool, device=device)
            return mask[:, :cloud_idx].bool()

        def _plan_metrics(mask: torch.Tensor) -> Dict[str, Any]:
            selected_edge = _edge_mask(mask)
            edge_pressure = pressure_matrix[:, :cloud_idx] if cloud_idx > 0 \
                else torch.zeros_like(selected_edge.float())
            masked_pressure = torch.where(selected_edge, edge_pressure, torch.zeros_like(edge_pressure))
            service_queue_pressure = masked_pressure.max(dim=1).values if masked_pressure.numel() else torch.zeros(
                (num_services,),
                device=device,
            )
            queue_pressure = float(service_queue_pressure.max().detach().cpu().item()) \
                if service_queue_pressure.numel() else 0.0

            pair_ctx = self.deployment_agent._deployment_pair_context(
                logic_edge_index,
                logic_feats,
                static_allowed,
                deploy_mask=mask,
            )
            option_ctx = self.deployment_agent._deployment_runtime_context(
                logic_feats,
                phys_feats,
                torch.zeros_like(static_allowed, dtype=torch.float32, device=device),
                static_allowed,
                pair_ctx,
                prev_deploy_mask=base_mask,
            )
            active_hotspot = pair_ctx["active_pair_hotspot"].to(device=device).float()
            edge_hotspot = active_hotspot[:, :cloud_idx] if cloud_idx > 0 else torch.zeros_like(masked_pressure)
            service_hotspot = edge_hotspot.max(dim=1).values if edge_hotspot.numel() else torch.zeros(
                (num_services,),
                device=device,
            )
            hotspot = float(service_hotspot.max().detach().cpu().item()) if service_hotspot.numel() else 0.0
            runtime_risk = option_ctx["runtime_risk_score"].to(device=device).float()
            pair_quality = option_ctx["pair_quality_score"].to(device=device).float()
            edge_runtime_risk = runtime_risk[:, :cloud_idx] if cloud_idx > 0 else torch.zeros_like(masked_pressure)
            edge_pair_quality = pair_quality[:, :cloud_idx] if cloud_idx > 0 else torch.zeros_like(masked_pressure)
            selected_risk_values = edge_runtime_risk.masked_select(selected_edge) if selected_edge.numel() \
                else torch.zeros((0,), device=device)
            selected_quality_values = edge_pair_quality.masked_select(selected_edge) if selected_edge.numel() \
                else torch.zeros((0,), device=device)
            runtime_risk_max = float(selected_risk_values.max().detach().cpu().item()) \
                if selected_risk_values.numel() else 0.0
            min_pair_quality = float(selected_quality_values.min().detach().cpu().item()) \
                if selected_quality_values.numel() else 0.0
            min_pair_quality_for_delta = min_pair_quality if selected_quality_values.numel() else 1.0
            low_quality_count = int((selected_quality_values < cfg.min_pair_quality).sum().detach().cpu().item()) \
                if selected_quality_values.numel() else 0
            risky_pair_count = int((selected_risk_values > cfg.max_runtime_risk).sum().detach().cpu().item()) \
                if selected_risk_values.numel() else 0
            return {
                "edge_mask": selected_edge,
                "service_queue_pressure": service_queue_pressure,
                "queue_pressure": queue_pressure,
                "service_hotspot": service_hotspot,
                "hotspot": hotspot,
                "edge_runtime_risk": edge_runtime_risk,
                "edge_pair_quality": edge_pair_quality,
                "runtime_risk": runtime_risk_max,
                "min_pair_quality": min_pair_quality,
                "min_pair_quality_for_delta": min_pair_quality_for_delta,
                "low_quality_count": low_quality_count,
                "risky_pair_count": risky_pair_count,
            }

        base_metrics = _plan_metrics(base_mask.bool())
        candidate_metrics = _plan_metrics(deploy_mask.bool())
        service_queue_pressure = candidate_metrics["service_queue_pressure"]
        service_hotspot = candidate_metrics["service_hotspot"]
        candidate_queue_pressure = float(candidate_metrics["queue_pressure"])
        candidate_hotspot = float(candidate_metrics["hotspot"])
        candidate_runtime_risk = float(candidate_metrics["runtime_risk"])
        candidate_min_pair_quality = float(candidate_metrics["min_pair_quality"])
        base_queue_pressure = float(base_metrics["queue_pressure"])
        base_hotspot = float(base_metrics["hotspot"])
        base_runtime_risk = float(base_metrics["runtime_risk"])
        base_min_pair_quality = float(base_metrics["min_pair_quality"])
        delta_queue_pressure = candidate_queue_pressure - base_queue_pressure
        delta_hotspot = candidate_hotspot - base_hotspot
        delta_runtime_risk = candidate_runtime_risk - base_runtime_risk
        delta_min_pair_quality = (
            float(candidate_metrics["min_pair_quality_for_delta"])
            - float(base_metrics["min_pair_quality_for_delta"])
        )
        delta_low_quality_count = int(candidate_metrics["low_quality_count"] - base_metrics["low_quality_count"])

        selected_edge = candidate_metrics["edge_mask"]
        raw_edge = raw_mask[:, :cloud_idx].bool() if cloud_idx > 0 else torch.zeros_like(selected_edge)
        base_edge = base_mask[:, :cloud_idx].bool() if cloud_idx > 0 else torch.zeros_like(selected_edge)
        exec_edge = deploy_mask[:, :cloud_idx].bool() if cloud_idx > 0 else torch.zeros_like(selected_edge)
        raw_change_count = int(torch.logical_xor(raw_edge, base_edge).sum().item()) if raw_edge.numel() else 0
        exec_change_count = int(torch.logical_xor(exec_edge, base_edge).sum().item()) if exec_edge.numel() else 0
        new_edge = exec_edge & ~base_edge
        removed_edge = base_edge & ~exec_edge
        new_edge_count = int(new_edge.sum().detach().cpu().item()) if new_edge.numel() else 0
        removed_edge_count = int(removed_edge.sum().detach().cpu().item()) if removed_edge.numel() else 0
        if new_edge_count > 0:
            new_edge_runtime = candidate_metrics["edge_runtime_risk"].masked_select(new_edge)
            new_edge_quality = candidate_metrics["edge_pair_quality"].masked_select(new_edge)
            new_edge_runtime_risk = float(new_edge_runtime.max().detach().cpu().item()) \
                if new_edge_runtime.numel() else 0.0
            new_edge_min_pair_quality = float(new_edge_quality.min().detach().cpu().item()) \
                if new_edge_quality.numel() else 0.0
        else:
            new_edge_runtime_risk = 0.0
            new_edge_min_pair_quality = 1.0

        feasible_edge = static_allowed[:, :cloud_idx].bool() if cloud_idx > 0 else torch.zeros_like(selected_edge)
        raw_removed_last = False
        if feasible_edge.numel():
            base_counts = base_edge.float().sum(dim=1)
            raw_counts = raw_edge.float().sum(dim=1)
            feasible_counts = feasible_edge.float().sum(dim=1)
            raw_removed_last = bool(((base_counts > 0.0) & (raw_counts <= 0.0) & (feasible_counts > 0.0)).any().item())

        hard_reasons = []
        soft_reasons = []
        if edge_cover_unmet > 0:
            hard_reasons.append("edge_cover_unmet")
        if capacity_relax_cost > cfg.max_capacity_relax_cost:
            hard_reasons.append("capacity_relax")
        if delta_queue_pressure > cfg.max_queue_pressure_increase and candidate_queue_pressure > cfg.max_queue_pressure:
            soft_reasons.append("queue_pressure")
        if delta_hotspot > cfg.max_hotspot_cost_increase and candidate_hotspot > cfg.max_hotspot_cost:
            soft_reasons.append("hotspot")
        if delta_runtime_risk > cfg.max_runtime_risk_increase and candidate_runtime_risk > cfg.max_runtime_risk:
            soft_reasons.append("runtime_risk")
        if delta_min_pair_quality < -cfg.max_pair_quality_drop and candidate_min_pair_quality < cfg.min_pair_quality:
            soft_reasons.append("pair_quality")
        if raw_removed_last and not cfg.allow_remove_last_edge_replica:
            hard_reasons.append("removed_last_edge")
        raw_change_limit = max(1, cfg.max_perturb_flips) * 2
        if raw_change_count > raw_change_limit:
            hard_reasons.append("too_many_raw_changes")
        reasons = [*hard_reasons, *soft_reasons]

        normalized_risks = [
            capacity_relax_cost / max(cfg.max_capacity_relax_cost, 1e-6),
            max(0.0, delta_queue_pressure) / max(cfg.max_queue_pressure_increase, 1e-6),
            max(0.0, delta_hotspot) / max(cfg.max_hotspot_cost_increase, 1e-6),
            max(0.0, delta_runtime_risk) / max(cfg.max_runtime_risk_increase, 1e-6),
            max(0.0, -delta_min_pair_quality) / max(cfg.max_pair_quality_drop, 1e-6),
            1.0 if edge_cover_unmet > 0 else 0.0,
            1.0 if raw_removed_last and not cfg.allow_remove_last_edge_replica else 0.0,
            1.0 if raw_change_count > raw_change_limit else 0.5 * raw_change_count / float(raw_change_limit),
        ]
        return {
            "collect_base_queue_pressure": base_queue_pressure,
            "collect_base_hotspot_cost": base_hotspot,
            "collect_base_runtime_risk": base_runtime_risk,
            "collect_base_min_pair_quality": base_min_pair_quality,
            "collect_candidate_queue_pressure": candidate_queue_pressure,
            "collect_candidate_hotspot_cost": candidate_hotspot,
            "collect_candidate_runtime_risk": candidate_runtime_risk,
            "collect_candidate_min_pair_quality": candidate_min_pair_quality,
            "collect_delta_queue_pressure": delta_queue_pressure,
            "collect_delta_hotspot_cost": delta_hotspot,
            "collect_delta_runtime_risk": delta_runtime_risk,
            "collect_delta_min_pair_quality": delta_min_pair_quality,
            "collect_new_edge_count": new_edge_count,
            "collect_removed_edge_count": removed_edge_count,
            "collect_new_edge_runtime_risk": new_edge_runtime_risk,
            "collect_new_edge_min_pair_quality": new_edge_min_pair_quality,
            "collect_base_low_quality_count": int(base_metrics["low_quality_count"]),
            "collect_candidate_low_quality_count": int(candidate_metrics["low_quality_count"]),
            "collect_delta_low_quality_count": delta_low_quality_count,
            "collect_base_risky_pair_count": int(base_metrics["risky_pair_count"]),
            "collect_candidate_risky_pair_count": int(candidate_metrics["risky_pair_count"]),
            "collect_predicted_risk": float(max(normalized_risks)),
            "collect_raw_change_count": raw_change_count,
            "collect_exec_change_count": exec_change_count,
            "collect_removed_last_edge": int(raw_removed_last),
            "collect_reject_reason": ",".join(reasons),
            "collect_hard_reject_reason": ",".join(hard_reasons),
            "collect_soft_reject_reason": ",".join(soft_reasons),
            "collect_candidate_queue_pressure_by_service": service_queue_pressure.detach().cpu(),
            "collect_candidate_hotspot_by_service": service_hotspot.detach().cpu(),
        }

    @staticmethod
    def _deployment_collect_candidate_accepted(collect_debug: Dict[str, Any]) -> Tuple[bool, str]:
        reason = str(collect_debug.get("collect_reject_reason", "") or "")
        if reason:
            return False, reason
        return True, ""

    def _deployment_collect_quality_bucket(self, metrics: Dict[str, Any], feedback_result) -> str:
        guard_interrupted = bool(getattr(feedback_result, "guard_interrupted", False)) \
            or bool(metrics.get("feedback_guard_interrupted", False))
        slo_violation = float(metrics.get("e2e_slo_violation", 0.0) or 0.0)
        p95 = float(metrics.get("e2e_latency_p95", 0.0) or 0.0)
        max_queue = float(metrics.get("latency_guard_max_queue", 0.0) or 0.0)
        if guard_interrupted or slo_violation >= 0.80 or p95 >= 12.0 or max_queue >= 16.0:
            return "bad"
        if slo_violation <= 0.40 and p95 <= 6.0:
            return "good"
        return "mid"

    def _update_deployment_collect_quality(self, quality_bucket: str) -> None:
        if quality_bucket == "bad":
            self._deployment_collect_bad_streak += 1
            self._deployment_collect_bad_count += 1
        else:
            self._deployment_collect_bad_streak = 0
            if quality_bucket == "good":
                self._deployment_collect_good_count += 1
        self._deployment_collect_last_quality = str(quality_bucket)

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

    def _sleep_until_next_inference_deployment_decision(
            self,
            last_tick: float,
            interval_s: float,
            last_guard_trigger_seq: int,
    ):
        """Sleep for the deployment cadence, but wake on a new latency-guard trigger."""
        interval_s = max(0.0, float(interval_s))
        if interval_s <= 0.0:
            return time.time(), "interval", int(last_guard_trigger_seq), self._deployment_event_wait_status("interval")

        now = time.time()
        target_t = now + interval_s if last_tick <= 0 else last_tick + interval_s
        poll_s = max(0.1, min(1.0, float(getattr(self.latency_guard_cfg, "poll_interval_s", 1.0))))
        last_guard_trigger_seq = int(last_guard_trigger_seq)

        while not self.deployment_thread_stop_event.is_set():
            current_seq = self._latency_guard_trigger_seq_value()
            if current_seq > last_guard_trigger_seq:
                self._latency_guard_trigger_event.clear()
                LOGGER.warning(
                    f"[Hedger][Inference][Deployment] Wake deployment decision immediately "
                    f"for latency guard trigger: seq={current_seq}, "
                    f"configured_interval={self._format_log_value(interval_s, 2)}s."
                )
                return time.time(), "latency_guard_trigger", int(current_seq), {
                    "triggered": True,
                    "reason": "latency_guard_trigger",
                    "queue_pressure": 0.0,
                    "hotspot_pressure": 0.0,
                    "max_queue": 0.0,
                    **self._deployment_event_suppression_defaults(),
                }

            event_status = self._deployment_event_trigger_status()
            if bool(event_status.get("triggered", False)):
                self._mark_deployment_event_triggered(event_status)
                LOGGER.warning(
                    f"[Hedger][DeploymentEvent] Wake deployment decision immediately: "
                    f"reason={event_status.get('reason')}, "
                    f"queue_pressure={self._format_log_value(event_status.get('queue_pressure', 0.0))}, "
                    f"hotspot_pressure={self._format_log_value(event_status.get('hotspot_pressure', 0.0))}, "
                    f"configured_interval={self._format_log_value(interval_s, 2)}s."
                )
                return time.time(), str(event_status.get("reason") or "event_trigger"), current_seq, event_status

            remaining_s = target_t - time.time()
            if remaining_s <= 0.0:
                return time.time(), "interval", last_guard_trigger_seq, self._deployment_event_wait_status("interval")

            wait_s = min(remaining_s, poll_s)
            self._latency_guard_trigger_event.wait(timeout=wait_s)
            if self._latency_guard_trigger_event.is_set() and \
                    self._latency_guard_trigger_seq_value() <= last_guard_trigger_seq:
                self._latency_guard_trigger_event.clear()
            if self._deployment_event_trigger_event.is_set():
                self._deployment_event_trigger_event.clear()

        return time.time(), "stop", last_guard_trigger_seq, self._deployment_event_wait_status("stop")

    def _wait_for_initial_inference_task_feedback(self, worker_name: str, stop_event: threading.Event) -> int:
        cfg = self.inference_cfg
        if not cfg.wait_for_initial_task_feedback or self.state_buffer is None:
            return self.state_buffer.get_task_observation_version() if self.state_buffer is not None else 0

        min_samples = max(1, int(cfg.initial_task_feedback_min_samples))
        timeout_s = cfg.initial_task_feedback_timeout_s
        start_t = time.monotonic()
        last_log_t = 0.0
        while not stop_event.is_set():
            current_count = self.state_buffer.get_task_observation_version()
            if current_count >= min_samples:
                LOGGER.info(
                    f"[Hedger][Inference][{worker_name}] Initial task feedback gate passed: "
                    f"samples={current_count}/{min_samples}."
                )
                return current_count

            now = time.monotonic()
            if timeout_s is not None and now - start_t >= float(timeout_s):
                LOGGER.warning(
                    f"[Hedger][Inference][{worker_name}] Initial task feedback gate timed out: "
                    f"samples={current_count}/{min_samples}, "
                    f"timeout={self._format_log_value(timeout_s, 2)}s. "
                    "Proceed with the best-effort state snapshot."
                )
                return current_count

            if now - last_log_t >= 10.0:
                last_log_t = now
                LOGGER.info(
                    f"[Hedger][Inference][{worker_name}] Waiting for initial completed-task feedback: "
                    f"samples={current_count}/{min_samples}."
                )
            time.sleep(0.5)

        return self.state_buffer.get_task_observation_version()

    def _wait_for_inference_deployment_served(self, decision_version: int) -> bool:
        if not self.inference_cfg.deployment_wait_until_served:
            return True

        timeout_s = self.inference_cfg.deployment_feedback_timeout_s
        start_t = time.monotonic()
        last_log_t = 0.0
        while not self.deployment_thread_stop_event.is_set():
            with self._deployment_version_cond:
                served_version = int(self._deployment_served_version)
                if served_version >= int(decision_version):
                    return True
                self._deployment_version_cond.wait(timeout=0.5)

            now = time.monotonic()
            if now - last_log_t >= 10.0:
                last_log_t = now
                elapsed = now - start_t
                LOGGER.info(
                    f"[Hedger][Inference][Deployment] Waiting for scheduler to serve deployment decision: "
                    f"served={served_version}, target={decision_version}, "
                    f"elapsed={self._format_log_value(elapsed, 2)}s."
                )
            if timeout_s is not None and now - start_t >= float(timeout_s):
                LOGGER.warning(
                    f"[Hedger][Inference][Deployment] Deployment serve wait exceeded "
                    f"{self._format_log_value(timeout_s, 2)}s for version={decision_version}; "
                    "continue waiting to avoid overwriting an unserved inference decision."
                )
                start_t = now

        return False

    def _wait_for_inference_deployment_feedback_samples(self, deployment_version: Optional[int]) -> Dict[str, Any]:
        cfg = self.inference_cfg
        required = max(1, int(cfg.deployment_min_version_matched_samples))
        if (
                not cfg.deployment_require_version_matched_feedback
                or self.state_buffer is None
                or deployment_version is None
        ):
            return {
                "enabled": bool(cfg.deployment_require_version_matched_feedback),
                "required": required,
                "count": self._deployment_feedback_count(deployment_version),
                "timed_out": False,
                "skipped": True,
            }

        timeout_s = cfg.deployment_feedback_timeout_s
        start_t = time.monotonic()
        last_log_t = 0.0
        while not self.deployment_thread_stop_event.is_set():
            count = self._deployment_feedback_count(deployment_version)
            if count >= required:
                return {
                    "enabled": True,
                    "required": required,
                    "count": count,
                    "timed_out": False,
                    "skipped": False,
                }

            if self._latency_guard_trigger_event_for_deployment(deployment_version) is not None:
                LOGGER.warning(
                    f"[Hedger][Inference][Deployment] Use guard-truncated deployment feedback: "
                    f"version={deployment_version}, samples={count}/{required}."
                )
                return {
                    "enabled": True,
                    "required": required,
                    "count": count,
                    "timed_out": False,
                    "skipped": False,
                    "guard_truncated": True,
                }

            now = time.monotonic()
            if timeout_s is not None and now - start_t >= float(timeout_s):
                LOGGER.warning(
                    f"[Hedger][Inference][Deployment] Version-matched feedback gate timed out: "
                    f"version={deployment_version}, samples={count}/{required}, "
                    f"timeout={self._format_log_value(timeout_s, 2)}s. "
                    "Proceed with available inference feedback."
                )
                return {
                    "enabled": True,
                    "required": required,
                    "count": count,
                    "timed_out": True,
                    "skipped": False,
                }

            if now - last_log_t >= 10.0:
                last_log_t = now
                LOGGER.info(
                    f"[Hedger][Inference][Deployment] Waiting for version-matched feedback: "
                    f"version={deployment_version}, samples={count}/{required}."
                )
            self.state_buffer.wait_for_offloading_rewards(
                required,
                timeout_s=0.5,
                deployment_version=deployment_version,
            )

        return {
            "enabled": True,
            "required": required,
            "count": self._deployment_feedback_count(deployment_version),
            "timed_out": False,
            "skipped": False,
        }

    @staticmethod
    def _build_service_demand_features(
            task_complexity_seq: torch.Tensor,
            model_flops: torch.Tensor,
            model_mem: torch.Tensor,
            task_arrival_rate_seq: torch.Tensor,
    ) -> torch.Tensor:
        task_complexity_seq = task_complexity_seq.detach().float()
        model_flops = model_flops.detach().float()
        model_mem = model_mem.detach().float()
        task_arrival_rate_seq = task_arrival_rate_seq.detach().float()
        current = task_complexity_seq[:, -1]
        mean = task_complexity_seq.mean(dim=1)
        std = task_complexity_seq.std(dim=1, unbiased=False)
        log_compute_demand = torch.log1p(current.clamp_min(0.0) * model_flops.clamp_min(0.0))
        complexity_zscore = (current - mean) / (std + 1e-6)
        arrival_rate = task_arrival_rate_seq[:, -1].clamp_min(0.0)
        log_model_mem = torch.log1p(model_mem.clamp_min(0.0))
        return torch.stack([log_compute_demand, complexity_zscore, arrival_rate, log_model_mem], dim=-1)

    @staticmethod
    def _build_device_capability_features(
            gpu_flops: torch.Tensor,
            mem_capacity: torch.Tensor,
            role_id: torch.Tensor,
            bandwidth_latest: torch.Tensor,
            cloud_idx: int,
    ) -> torch.Tensor:
        gpu_flops = gpu_flops.detach().float()
        mem_capacity = mem_capacity.detach().float()
        role_id = role_id.detach().long()
        bandwidth_latest = bandwidth_latest.detach().float().clamp_min(1e-6)
        is_cloud = (role_id == 1).float()
        if 0 <= cloud_idx < is_cloud.numel():
            is_cloud[cloud_idx] = 1.0
        edge_mask = is_cloud < 0.5
        log_gpu = torch.log1p(gpu_flops.clamp_min(0.0))
        log_mem = torch.log1p(mem_capacity.clamp_min(0.0))
        if edge_mask.any():
            edge_compute_mean = log_gpu[edge_mask].mean()
            edge_bw_ref = bandwidth_latest[edge_mask].mean().clamp_min(1e-6)
        else:
            edge_compute_mean = log_gpu.mean()
            edge_bw_ref = bandwidth_latest.mean().clamp_min(1e-6)
        relative_edge_compute = torch.where(edge_mask, log_gpu - edge_compute_mean, torch.zeros_like(log_gpu))
        cloud_bandwidth_penalty = is_cloud * torch.log1p(edge_bw_ref / bandwidth_latest)
        return torch.stack(
            [log_gpu, log_mem, relative_edge_compute, is_cloud, cloud_bandwidth_penalty],
            dim=-1,
        )

    @staticmethod
    def _snapshot_pair_tensor(
            snapshot: Dict[str, Any],
            key: str,
            shape: Tuple[int, int],
    ) -> torch.Tensor:
        value = snapshot.get(key) if isinstance(snapshot, dict) else None
        try:
            tensor = torch.tensor(value, dtype=torch.float32)
        except (TypeError, ValueError):
            return torch.zeros(shape, dtype=torch.float32)
        if tensor.dim() != 2 or tuple(tensor.shape) != tuple(shape):
            return torch.zeros(shape, dtype=torch.float32)
        return tensor

    @staticmethod
    def _build_runtime_pair_features(
            runtime_pair_snapshot: Dict[str, Any],
            queue_pair_snapshot: Dict[str, Any],
    ) -> torch.Tensor:
        runtime_pair = torch.tensor(
            runtime_pair_snapshot.get("pair_time_per_complexity_short", []),
            dtype=torch.float32,
        )
        queue_pair = torch.tensor(queue_pair_snapshot.get("pair_short", []), dtype=torch.float32)
        if runtime_pair.dim() != 2 or queue_pair.dim() != 2 or tuple(runtime_pair.shape) != tuple(queue_pair.shape):
            return torch.zeros((0, 0, len(RUNTIME_PAIR_FEATURE_NAMES)), dtype=torch.float32)
        shape = (runtime_pair.size(0), runtime_pair.size(1))
        real_time_per_complexity = Hedger._snapshot_pair_tensor(
            runtime_pair_snapshot,
            "pair_time_per_complexity_short",
            shape,
        )
        queue_short = Hedger._snapshot_pair_tensor(queue_pair_snapshot, "pair_short", shape)
        queue_busy = Hedger._snapshot_pair_tensor(queue_pair_snapshot, "pair_busy", shape)
        queue_count = Hedger._snapshot_pair_tensor(queue_pair_snapshot, "pair_count", shape)
        runtime_count = Hedger._snapshot_pair_tensor(runtime_pair_snapshot, "pair_count", shape)
        runtime_last_task_v = Hedger._snapshot_pair_tensor(runtime_pair_snapshot, "pair_last_task_v", shape)
        queue_last_t = Hedger._snapshot_pair_tensor(queue_pair_snapshot, "pair_last_t", shape)
        current_task_version = float(runtime_pair_snapshot.get("current_task_version", 0.0) or 0.0)
        monotonic_time = float(runtime_pair_snapshot.get("monotonic_time", 0.0) or 0.0)
        runtime_confidence = torch.clamp(
            torch.log1p(runtime_count.clamp_min(0.0)) / math.log1p(20.0),
            min=0.0,
            max=1.0,
        )
        runtime_age = torch.clamp(current_task_version - runtime_last_task_v, min=0.0)
        runtime_recency = torch.where(runtime_count > 0.0, 1.0 / (1.0 + runtime_age), torch.zeros_like(runtime_age))
        queue_age = torch.clamp(monotonic_time - queue_last_t, min=0.0)
        queue_freshness = torch.where(queue_count > 0.0, torch.exp(-queue_age / 30.0), torch.zeros_like(queue_age))
        return torch.stack(
            [
                queue_short,
                queue_busy,
                real_time_per_complexity,
                runtime_confidence,
                runtime_recency,
                queue_freshness,
            ],
            dim=-1,
        )

    def _collect_graph_state(
            self,
            seq_len: int,
            current_deployment_version: Optional[int] = None,
    ):
        assert self.state_buffer is not None, "State buffer must be registered before collecting state."
        logic_feats_raw, phys_feats_raw, state_debug = self.state_buffer.get_state_bundle(
            seq_len=seq_len,
            wait_cfg=self._build_state_wait_cfg(),
            pad_mode="edge",
            current_deployment_version=current_deployment_version,
        )

        logic_feats = {
            "model_flops": torch.tensor(logic_feats_raw["model_flops"], dtype=torch.float32),
            "model_mem": torch.tensor(logic_feats_raw["model_mem"], dtype=torch.float32),
            "task_complexity_seq": torch.tensor(logic_feats_raw["task_complexity_seq"], dtype=torch.float32),
            "task_arrival_rate_seq": torch.tensor(logic_feats_raw["task_arrival_rate_seq"], dtype=torch.float32),
        }
        logic_feats["service_demand_feat"] = self._build_service_demand_features(
            logic_feats["task_complexity_seq"],
            logic_feats["model_flops"],
            logic_feats["model_mem"],
            logic_feats["task_arrival_rate_seq"],
        )
        logic_feats["runtime_pair_feat"] = self._build_runtime_pair_features(
            state_debug.get("runtime_pair_snapshot", {}),
            state_debug.get("queue_pair_snapshot", {}),
        )
        phys_feats = {
            "gpu_flops": torch.tensor(phys_feats_raw["gpu_flops"], dtype=torch.float32),
            "role_id": torch.tensor(phys_feats_raw["role_id"], dtype=torch.long),
            "mem_capacity": torch.tensor(phys_feats_raw["mem_capacity"], dtype=torch.float32),
            "bandwidth_latest": torch.tensor(phys_feats_raw["bandwidth_latest"], dtype=torch.float32),
        }
        cloud_idx = self.physical_topology.cloud_idx if self.physical_topology is not None else len(
            phys_feats_raw["gpu_flops"]
        ) - 1
        phys_feats["device_capability_feat"] = self._build_device_capability_features(
            phys_feats["gpu_flops"],
            phys_feats["mem_capacity"],
            phys_feats["role_id"],
            phys_feats["bandwidth_latest"],
            cloud_idx,
        )
        return logic_feats, phys_feats, state_debug

    @staticmethod
    def _percentile(values, percentile: float) -> float:
        if values is None:
            iterable = []
        elif isinstance(values, torch.Tensor):
            iterable = values.detach().float().flatten().cpu().tolist()
        else:
            iterable = values

        clean_values = []
        for value in iterable:
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(float_value):
                clean_values.append(float_value)

        if not clean_values:
            return 0.0
        if len(clean_values) == 1:
            return float(clean_values[0])

        clean_values.sort()
        rank = (len(clean_values) - 1) * (float(percentile) / 100.0)
        lower = int(math.floor(rank))
        upper = min(lower + 1, len(clean_values) - 1)
        weight = rank - lower
        return float(clean_values[lower] + (clean_values[upper] - clean_values[lower]) * weight)

    def _compute_slo_violation(self, latency_values) -> float:
        if self.state_cfg.latency_slo is None:
            return 0.0
        if isinstance(latency_values, torch.Tensor):
            values = latency_values.detach().float().flatten()
        else:
            values = torch.tensor(list(latency_values or []), dtype=torch.float32)
        if values.numel() == 0:
            return 0.0
        threshold = float(self.state_cfg.latency_slo)
        return float((values > threshold).float().mean().item())

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
        """Normalized deployment change ratio over service-edge replica slots."""
        if prev_deploy_mask is None:
            return 0.0

        current_mask = self._current_deploy_mask().bool()
        prev_mask = prev_deploy_mask.detach().clone().cpu().bool()
        if current_mask.shape != prev_mask.shape:
            return 0.0

        cloud_idx = self.physical_topology.cloud_idx
        current_edge = current_mask[:, :cloud_idx]
        prev_edge = prev_mask[:, :cloud_idx]
        edge_slots = int(current_edge.numel())
        if edge_slots <= 0:
            return 0.0
        changed = float(torch.logical_xor(current_edge, prev_edge).sum().item())
        return float(changed / float(edge_slots))

    def _deployment_layout_metrics(self, exec_deploy_mask: torch.Tensor) -> Dict[str, float]:
        exec_deploy_mask = exec_deploy_mask.detach().cpu().bool()
        cloud_idx = self.physical_topology.cloud_idx
        num_services = max(1, int(exec_deploy_mask.size(0)))
        edge_device_count = max(0, int(cloud_idx))
        if edge_device_count <= 0:
            return {
                "cloud_only": 0,
                "cloud_only_ratio": 0.0,
                "empty_edge_devices": 0,
                "empty_edge_device_ratio": 0.0,
            }

        edge_mask = exec_deploy_mask[:, :cloud_idx]
        cloud_only = int((~edge_mask.any(dim=1)).sum().item())
        used_edge_devices = edge_mask.any(dim=0)
        empty_edge_devices = int((~used_edge_devices).sum().item())
        return {
            "cloud_only": cloud_only,
            "cloud_only_ratio": float(cloud_only) / float(num_services),
            "empty_edge_devices": empty_edge_devices,
            "empty_edge_device_ratio": float(empty_edge_devices) / float(edge_device_count),
        }

    def _collect_deployment_state(self, prev_deploy_mask: Optional[torch.Tensor] = None,
                                  deployment_version: Optional[int] = None):
        state_deployment_version = self.get_active_deployment_version()
        logic_feats, phys_feats, state_debug = self._collect_graph_state(
            self.state_cfg.deployment_seq_len,
            current_deployment_version=state_deployment_version,
        )
        reward_stats = self.state_buffer.get_offloading_reward_stats(
            last_k=self.state_cfg.deployment_reward_window,
            deployment_version=deployment_version,
        )
        latency_stats = self.state_buffer.get_task_end_to_end_latency_stats(
            deployment_version=deployment_version,
            last_k=self.state_cfg.deployment_reward_window,
        )
        latency_values = latency_stats.get("latencies", [])
        metrics = {
            "avg_offloading_reward": float(reward_stats["mean"]),
            "offloading_reward_std": float(reward_stats["std"]),
            "offloading_reward_count": int(reward_stats.get("count", 0)),
            "e2e_latency_count": int(latency_stats.get("count", 0)),
            "e2e_latency_mean": float(latency_stats.get("mean", 0.0)),
            "e2e_latency_latest": float(latency_stats.get("latest", 0.0)),
            "e2e_latency_p50": self._percentile(latency_values, 50),
            "e2e_latency_p90": self._percentile(latency_values, 90),
            "e2e_latency_p95": self._percentile(latency_values, 95),
            "e2e_latency_p99": self._percentile(latency_values, 99),
            "e2e_slo_violation": self._compute_slo_violation(latency_values),
            "deploy_change_cost": self._compute_deploy_change_cost(prev_deploy_mask),
        }
        done = False
        return logic_feats, phys_feats, metrics, done, state_debug

    def _attach_deployment_feedback_status(
            self,
            metrics: Dict[str, Any],
            feedback_result: HedgerDeploymentFeedbackWaitResult,
            required_samples: int,
    ) -> Dict[str, Any]:
        required_samples = max(1, int(required_samples))
        actual_count = int(metrics.get("offloading_reward_count", feedback_result.count))
        sample_shortfall = max(0, required_samples - actual_count)
        shortfall_ratio = float(sample_shortfall) / float(required_samples)
        guard_stats = feedback_result.guard_stats or {}
        guard_sample_count = int(guard_stats.get("count", 0) or 0)
        guard_bad_count = int(guard_stats.get("bad_count", 0) or 0)
        guard_bad_ratio = float(guard_stats.get("bad_ratio", 0.0) or 0.0)
        guard_max_queue = float(guard_stats.get("max_queue_length", 0.0) or 0.0)

        guard_penalty_cost = 0.0
        if feedback_result.guard_interrupted:
            guard_penalty_cost = max(
                guard_bad_ratio,
                float(getattr(self.latency_guard_cfg, "trigger_violation_ratio", 0.0)),
            )

        metrics.update({
            "feedback_required_samples": required_samples,
            "feedback_sample_shortfall": sample_shortfall,
            "feedback_shortfall_ratio": shortfall_ratio,
            "feedback_timed_out": int(bool(feedback_result.timed_out)),
            "feedback_timeout_s": (
                0.0 if feedback_result.timeout_s is None else float(feedback_result.timeout_s)
            ),
            "feedback_timeout_penalty_cost": shortfall_ratio if feedback_result.timed_out else 0.0,
            "feedback_guard_interrupted": int(bool(feedback_result.guard_interrupted)),
            "latency_guard_trigger_seq": int(feedback_result.guard_event_seq),
            "latency_guard_bad_ratio": guard_bad_ratio,
            "latency_guard_bad_count": guard_bad_count,
            "latency_guard_sample_count": guard_sample_count,
            "latency_guard_max_queue": guard_max_queue,
            "latency_guard_penalty_cost": float(guard_penalty_cost),
        })
        return metrics

    def _reset_deployment_feedback_window(self, deployment_version: Optional[int] = None):
        if self.state_buffer is not None:
            self.state_buffer.clear_offloading_rewards(deployment_version=deployment_version)

    def _mark_deployment_decision_pending(self) -> int:
        with self._deployment_version_cond:
            self._deployment_decision_version += 1
            version = self._deployment_decision_version
            self._deployment_version_cond.notify_all()
            return version

    def _mark_deployment_decision_served(self) -> Optional[int]:
        served_version = None
        with self._deployment_version_cond:
            if self._deployment_decision_version > self._deployment_served_version:
                self._deployment_served_version = self._deployment_decision_version
                served_version = self._deployment_served_version
                self._reset_deployment_feedback_window(deployment_version=served_version)
                self._deployment_version_cond.notify_all()
        return served_version

    def _wait_for_deployment_decision_served(self, decision_version: int,
                                             abort_on_guard: bool = True) -> bool:
        while not self.deployment_thread_stop_event.is_set():
            if self.is_latency_guard_active():
                if abort_on_guard or self.latency_guard_cfg.force_default_decisions:
                    LOGGER.warning(
                        f"[Hedger][Train][Deployment] Abort waiting for deployment decision "
                        f"version={decision_version} because latency guard is active."
                    )
                    return False
                self._sleep_while_latency_guard_active("deployment decision wait")
                continue
            with self._deployment_version_cond:
                if self._deployment_served_version >= decision_version:
                    return True

                wait_timeout = self.state_cfg.deployment_feedback_timeout_s
                wait_s = 5.0 if wait_timeout is None else min(5.0, max(0.1, float(wait_timeout)))
                self._deployment_version_cond.wait(timeout=wait_s)
                served_version = self._deployment_served_version

            if served_version < decision_version:
                LOGGER.warning(
                    f"[Hedger][Train][Deployment] Waiting for scheduler to serve deployment decision: "
                    f"served={served_version}, target={decision_version}"
                )

        return False

    def _delete_error_processor_pods_if_needed(self, reason: str) -> int:
        now = time.monotonic()
        if now - self._last_processor_pod_cleanup_t < self._processor_pod_cleanup_cooldown_s:
            return 0

        self._last_processor_pod_cleanup_t = now
        try:
            deleted = KubeConfig.delete_error_processor_pods(max_deletions=10)
        except Exception as exc:
            LOGGER.warning(
                f"[Hedger][Recovery] Failed to delete error processor pods while {reason}: {exc}"
            )
            return 0

        deleted_count = sum(1 for item in deleted if item.get("deleted"))
        if deleted_count > 0:
            LOGGER.warning(
                f"[Hedger][Recovery] Deleted error processor pods while {reason}: "
                f"count={deleted_count}"
            )
        return deleted_count

    def _deployment_feedback_min_samples(self) -> int:
        return max(
            1,
            min(self.state_cfg.deployment_reward_min_samples, self.state_cfg.deployment_reward_window),
        )

    def _deployment_feedback_count(self, deployment_version: Optional[int] = None) -> int:
        if self.state_buffer is None:
            return 0
        return int(self.state_buffer.wait_for_offloading_rewards(
            0,
            timeout_s=0.0,
            deployment_version=deployment_version,
        ))

    def _wait_for_deployment_feedback_samples_result(
            self,
            min_samples: int,
            deployment_version: Optional[int] = None,
            allow_guard_truncated: bool = False,
    ) -> HedgerDeploymentFeedbackWaitResult:
        if self.state_buffer is None:
            return HedgerDeploymentFeedbackWaitResult(ok=False)

        min_samples = max(1, min(int(min_samples), self.state_cfg.deployment_reward_window))
        start_t = time.monotonic()
        timeout_s = self.state_cfg.deployment_feedback_timeout_s
        while not self.deployment_thread_stop_event.is_set():
            if allow_guard_truncated:
                guard_event = self._latency_guard_trigger_event_for_deployment(deployment_version)
                if guard_event is not None:
                    count = self._deployment_feedback_count(deployment_version)
                    LOGGER.warning(
                        f"[Hedger][Train][Deployment] Use guard-truncated deployment feedback: "
                        f"version={deployment_version}, samples={count}/{min_samples}, "
                        f"guard_event_seq={guard_event['seq']}."
                    )
                    return HedgerDeploymentFeedbackWaitResult(
                        ok=True,
                        count=count,
                        guard_interrupted=True,
                        guard_event_seq=int(guard_event["seq"]),
                        guard_stats=copy.deepcopy(guard_event.get("stats", {}) or {}),
                    )
            if self.is_latency_guard_active():
                if allow_guard_truncated:
                    if self.latency_guard_cfg.force_default_decisions:
                        LOGGER.warning(
                            f"[Hedger][Train][Deployment] Abort waiting for deployment feedback "
                            f"version={deployment_version} because latency guard is forcing default decisions."
                        )
                        return HedgerDeploymentFeedbackWaitResult(
                            ok=False,
                            count=self._deployment_feedback_count(deployment_version),
                        )
                    self._sleep_while_latency_guard_active("deployment feedback wait")
                    continue
                LOGGER.warning(
                    f"[Hedger][Train][Deployment] Abort waiting for deployment feedback "
                    f"version={deployment_version} because latency guard is active."
                )
                return HedgerDeploymentFeedbackWaitResult(
                    ok=False,
                    count=self._deployment_feedback_count(deployment_version),
                )
            self._delete_error_processor_pods_if_needed(
                reason=f"waiting for deployment feedback samples={min_samples}"
            )
            wait_timeout_s = 5.0
            if timeout_s is not None:
                wait_timeout_s = min(5.0, max(0.5, float(timeout_s)))
            count = self.state_buffer.wait_for_offloading_rewards(
                min_samples,
                timeout_s=wait_timeout_s,
                deployment_version=deployment_version,
            )
            if count >= min_samples:
                return HedgerDeploymentFeedbackWaitResult(ok=True, count=count)

            if timeout_s is not None and time.monotonic() - start_t >= float(timeout_s):
                LOGGER.warning(
                    f"[Hedger][Train][Deployment] Feedback wait timed out: "
                    f"version={deployment_version}, samples={count}/{min_samples}, "
                    f"timeout={self._format_log_value(timeout_s, 2)}s. "
                    "Proceed with available training feedback and apply shortfall penalty."
                )
                return HedgerDeploymentFeedbackWaitResult(
                    ok=True,
                    count=count,
                    timed_out=True,
                    timeout_s=float(timeout_s),
                )

            LOGGER.warning(
                f"[Hedger][Train][Deployment] Waiting for fresh offloading feedback: "
                f"version={deployment_version}, samples={count}/{min_samples}, "
                f"timeout={self._format_log_value(timeout_s, 2)}s"
            )

        return HedgerDeploymentFeedbackWaitResult(
            ok=False,
            count=self._deployment_feedback_count(deployment_version),
        )

    def _wait_for_deployment_feedback_samples(self, min_samples: int,
                                              deployment_version: Optional[int] = None) -> bool:
        return self._wait_for_deployment_feedback_samples_result(
            min_samples,
            deployment_version=deployment_version,
            allow_guard_truncated=False,
        ).ok

    def _run_deployment_default_warmup(self) -> None:
        """
        Keep serving the template deployment for a short burn-in window.

        This phase fills the state buffer and establishes baseline offloading
        feedback before the deployment actor is allowed to publish learned
        redeployment actions. No deployment PPO transition is recorded here,
        because the environment is executing the default template rather than
        an action sampled from the deployment policy.
        """
        if self.training_cfg is None:
            return
        cfg = self.training_cfg.deployment_default_warmup
        if not cfg.enabled:
            return
        if cfg.min_intervals <= 0 and cfg.min_feedback_samples <= 0:
            LOGGER.info(
                "[Hedger][Train][DeploymentWarmup] Default-deployment warmup is enabled "
                "but both min_intervals and min_feedback_samples are zero; skip."
            )
            return
        if self.state_buffer is None:
            LOGGER.warning(
                "[Hedger][Train][DeploymentWarmup] State buffer is not registered; "
                "skip default-deployment warmup."
            )
            return

        with self._data_lock:
            self.pending_deployment_plan = None
            self.pending_deploy_mask = None
            if self.initial_deployment_plan is not None:
                self.deployment_plan = copy.deepcopy(self.initial_deployment_plan)
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.initial_deployment_plan
                ).detach().cpu()
            elif self.deployment_plan is not None:
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.deployment_plan
                ).detach().cpu()

        active_version = self.get_active_deployment_version()
        if cfg.clear_feedback_window:
            self.state_buffer.clear_offloading_rewards(deployment_version=active_version)

        LOGGER.info(
            f"[Hedger][Train][DeploymentWarmup] Start default-deployment warmup: "
            f"deployment_version={active_version}, "
            f"min_intervals={cfg.min_intervals}, "
            f"min_feedback_samples={cfg.min_feedback_samples}, "
            f"timeout_s={self._format_log_value(cfg.timeout_s, 2)}, "
            f"{self._summarize_deployment_plan(self.deployment_plan)}"
        )

        start_t = time.monotonic()
        last_log_t = 0.0
        reward_count = 0
        completed_intervals = 0

        while not self.deployment_thread_stop_event.is_set():
            if self._sleep_while_latency_guard_active("deployment default warmup"):
                active_version = self.get_active_deployment_version()
                if cfg.clear_feedback_window:
                    self.state_buffer.clear_offloading_rewards(deployment_version=active_version)
                start_t = time.monotonic()
                last_log_t = 0.0
                reward_count = 0
                completed_intervals = 0
                LOGGER.info(
                    "[Hedger][Train][DeploymentWarmup] Restart default-deployment warmup "
                    "after latency guard recovery."
                )
                continue

            now = time.monotonic()
            elapsed = max(0.0, now - start_t)
            completed_intervals = int(elapsed // max(self.deployment_interval, 1e-6))

            interval_ready = completed_intervals >= cfg.min_intervals
            feedback_ready = reward_count >= cfg.min_feedback_samples
            if interval_ready and feedback_ready:
                break

            if cfg.timeout_s is not None and elapsed >= cfg.timeout_s:
                LOGGER.warning(
                    f"[Hedger][Train][DeploymentWarmup] Timeout before all warmup targets were met: "
                    f"elapsed={self._format_log_value(elapsed, 2)}s, "
                    f"intervals={completed_intervals}/{cfg.min_intervals}, "
                    f"feedback={reward_count}/{cfg.min_feedback_samples}. "
                    f"Proceed with learned deployment decisions."
                )
                break

            wait_timeout = 1.0
            if cfg.timeout_s is not None:
                wait_timeout = min(wait_timeout, max(0.0, cfg.timeout_s - elapsed))

            if not feedback_ready and cfg.min_feedback_samples > 0:
                reward_count = self.state_buffer.wait_for_offloading_rewards(
                    cfg.min_feedback_samples,
                    timeout_s=wait_timeout,
                    deployment_version=active_version,
                )
            else:
                reward_count = self.state_buffer.wait_for_offloading_rewards(
                    0,
                    timeout_s=0.0,
                    deployment_version=active_version,
                )
                sleep_s = wait_timeout
                if cfg.min_intervals > 0:
                    target_elapsed = float(cfg.min_intervals) * float(self.deployment_interval)
                    sleep_s = min(sleep_s, max(0.0, target_elapsed - elapsed))
                if sleep_s > 0.0:
                    time.sleep(min(1.0, sleep_s))

            now = time.monotonic()
            if now - last_log_t >= 10.0:
                last_log_t = now
                elapsed = max(0.0, now - start_t)
                completed_intervals = int(elapsed // max(self.deployment_interval, 1e-6))
                LOGGER.info(
                    f"[Hedger][Train][DeploymentWarmup] Waiting on default deployment: "
                    f"elapsed={self._format_log_value(elapsed, 2)}s, "
                    f"intervals={completed_intervals}/{cfg.min_intervals}, "
                    f"feedback={reward_count}/{cfg.min_feedback_samples}, "
                    f"deployment_version={active_version}"
                )

        if self.deployment_thread_stop_event.is_set():
            return

        elapsed = max(0.0, time.monotonic() - start_t)
        completed_intervals = int(elapsed // max(self.deployment_interval, 1e-6))
        reward_count = self.state_buffer.wait_for_offloading_rewards(
            0,
            timeout_s=0.0,
            deployment_version=active_version,
        )
        LOGGER.info(
            f"[Hedger][Train][DeploymentWarmup] Complete default-deployment warmup: "
            f"elapsed={self._format_log_value(elapsed, 2)}s, "
            f"intervals={completed_intervals}/{cfg.min_intervals}, "
            f"feedback={reward_count}/{cfg.min_feedback_samples}. "
            f"Learned deployment decisions are now enabled."
        )

    def _collect_offloading_state(
            self,
            since_task_version: Optional[int] = None,
            deployment_version: Optional[int] = None,
    ):
        state_deployment_version = self.get_active_deployment_version()
        logic_feats, phys_feats, state_debug = self._collect_graph_state(
            self.state_cfg.offloading_seq_len,
            current_deployment_version=state_deployment_version,
        )
        latency_stats = self.state_buffer.get_task_end_to_end_latency_stats(
            since_version=since_task_version,
            deployment_version=deployment_version,
            last_k=None if since_task_version is not None else 1,
        )
        latency_values = latency_stats.get("latencies", [])
        metrics = {
            "latency": float(latency_stats["mean"]),
            "slo_violation": self._compute_slo_violation(latency_values),
            "cloud_fraction": self._compute_cloud_fraction(),
            "task_latency_count": int(latency_stats.get("count", 0)),
            "latest_task_latency": float(latency_stats.get("latest", 0.0)),
        }
        done = False
        return logic_feats, phys_feats, metrics, done, state_debug

    def inference_hedger(self):
        assert self.logical_topology is not None, "Logical topology must be registered before inference."
        assert self.physical_topology is not None, "Physical topology must be registered before inference."
        assert self.state_buffer is not None, "State buffer must be registered before inference."
        if self.checkpoint_cfg.load.enabled and self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "[Hedger][Inference] Checkpoint loading was enabled but no checkpoint was loaded. "
                "Please check checkpoint.load.from_stage/which/path before running inference."
            )
        if not self.checkpoint_cfg.load.enabled:
            raise RuntimeError(
                "[Hedger][Inference] checkpoint.load.enabled must be true for inference mode, "
                "otherwise Hedger would serve randomly initialized policies."
            )

        LOGGER.info(f"[Hedger][Inference] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        LOGGER.info(
            f"[Hedger][Inference] Worker config: "
            f"deployment={self.inference_cfg.run_deployment_worker}, "
            f"offloading={self.inference_cfg.run_offloading_worker}, "
            f"deployment_deterministic={self.inference_cfg.deployment_deterministic}, "
            f"offloading_deterministic={self.inference_cfg.offloading_deterministic}"
        )
        self.set_seed()

        self.shared_topology_encoder.eval()
        if self.inference_cfg.run_deployment_worker:
            self.deployment_agent.eval()
        if self.inference_cfg.run_offloading_worker:
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

        if self.inference_cfg.run_deployment_worker:
            self.dep_recorder = Recorder(
                self._inference_log_path("deployment_inference.csv"),
                fmt="csv",
                fieldnames=self._inference_deployment_fieldnames(),
                overwrite=True,
                flush_every=1,
            )
            self.dep_decision_recorder = Recorder(
                self._inference_log_path("deployment_decisions.csv"),
                fmt="csv",
                fieldnames=self._deployment_decision_fieldnames(),
                overwrite=True,
                flush_every=1,
            )
        else:
            self.dep_recorder = None
            self.dep_decision_recorder = None

        if self.inference_cfg.run_offloading_worker:
            self.off_recorder = Recorder(
                self._inference_log_path("offloading_inference.csv"),
                fmt="csv",
                fieldnames=self._inference_offloading_fieldnames(),
                overwrite=True,
                flush_every=1,
            )
            self.off_decision_recorder = Recorder(
                self._inference_log_path("offloading_decisions.csv"),
                fmt="csv",
                fieldnames=self._offloading_decision_fieldnames(),
                overwrite=True,
                flush_every=1,
            )
        else:
            self.off_recorder = None
            self.off_decision_recorder = None

        deployment_thread = None
        offloading_thread = None
        if self.inference_cfg.run_deployment_worker:
            deployment_thread = threading.Thread(target=self.inference_deployment_agent, daemon=True)
            deployment_thread.start()
        if self.inference_cfg.run_offloading_worker:
            offloading_thread = threading.Thread(target=self.inference_offloading_agent, daemon=True)
            offloading_thread.start()

        while True:
            dep_alive = deployment_thread.is_alive() if deployment_thread is not None else False
            off_alive = offloading_thread.is_alive() if offloading_thread is not None else False

            if deployment_thread is None and offloading_thread is None:
                break
            if self.inference_cfg.run_deployment_worker and self.deployment_thread_stop_event.is_set():
                self.offloading_thread_stop_event.set()
                break
            if self.inference_cfg.run_offloading_worker and self.offloading_thread_stop_event.is_set():
                self.deployment_thread_stop_event.set()
                break
            if deployment_thread is not None and not dep_alive:
                LOGGER.warning('[Hedger][Inference] Deployment worker stopped unexpectedly.')
                self.offloading_thread_stop_event.set()
                break
            if offloading_thread is not None and not off_alive:
                LOGGER.warning('[Hedger][Inference] Offloading worker stopped unexpectedly.')
                self.deployment_thread_stop_event.set()
                break
            time.sleep(0.5)

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        if deployment_thread is not None:
            deployment_thread.join(timeout=5.0)
        if offloading_thread is not None:
            offloading_thread.join(timeout=5.0)
        LOGGER.info(
            f"[Hedger][Inference] Finished: "
            f"{self._summarize_deployment_plan(self.deployment_plan)}, "
            f"{self._summarize_offloading_plan(self.offloading_plan)}"
        )

    def inference_deployment_agent(self):
        LOGGER.info(
            f"[Hedger][Inference][Deployment] Worker started: "
            f"interval={self._format_log_value(self.deployment_interval, 2)}s, "
            f"deterministic={self.inference_cfg.deployment_deterministic}"
        )

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting deployment inference."

        self._wait_for_initial_inference_task_feedback(
            "Deployment",
            self.deployment_thread_stop_event,
        )
        if self.deployment_thread_stop_event.is_set():
            LOGGER.info("[Hedger][Inference][Deployment] Worker stopped during initial feedback wait.")
            return

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        deployment_time_ticket = 0
        prev_deploy_mask = self._current_deploy_mask()
        logic_feats, phys_feats, _, _, state_debug = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
        last_guard_trigger_seq = self._latency_guard_trigger_seq_value()
        next_decision_reason = "startup"
        next_event_status: Dict[str, Any] = {"triggered": False, "reason": "startup"}

        step = 0
        while not self.deployment_thread_stop_event.is_set():
            try:
                current_decision_reason = next_decision_reason
                current_event_status = next_event_status
                state_logic_feats = logic_feats
                state_phys_feats = phys_feats
                state_debug_record = state_debug
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

                with self._model_lock:
                    self._sync_decision_device()
                    with self.deployment_overhead_estimator, torch.inference_mode():
                        deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
                            logic_edge_index=logic_edge_index,
                            logic_feats=logic_feats_dev,
                            phys_edge_index=phys_edge_index,
                            phys_feats=phys_feats_dev,
                            topo_order=None,
                            prev_deploy_mask=prev_deploy_mask_dev,
                            deterministic=self.inference_cfg.deployment_deterministic,
                        )
                        deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                        raw_deploy_mask = aux["raw_deploy_mask"].detach().cpu().bool()
                        exec_deploy_mask = deploy_mask.detach().cpu().bool()
                        raw_deploy_plan = self._map_deployment_mask_to_deployment_plan(raw_deploy_mask)
                        self._sync_decision_device()
                deployment_decision_overhead_s = self.get_deployment_decision_overhead()
                with self._data_lock:
                    self.pending_deployment_plan = deploy_plan
                    self.pending_deploy_mask = deploy_mask.detach().cpu()
                    self._pending_deployment_force_serve = current_decision_reason in {
                        "latency_guard_trigger",
                        "event_queue_pressure",
                        "event_pair_hotspot",
                        "event_e2e_slo",
                        "event_e2e_p95",
                    }
                    self._pending_deployment_reason = current_decision_reason
                    decision_version = self._mark_deployment_decision_pending()
                if not self._wait_for_inference_deployment_served(decision_version):
                    break
                (
                    deployment_time_ticket,
                    next_decision_reason,
                    last_guard_trigger_seq,
                    next_event_status,
                ) = self._sleep_until_next_inference_deployment_decision(
                    deployment_time_ticket,
                    self.deployment_interval,
                    last_guard_trigger_seq,
                )

                served_version = self.get_active_deployment_version()
                if next_decision_reason not in {
                        "latency_guard_trigger",
                        "event_queue_pressure",
                        "event_pair_hotspot",
                        "event_e2e_slo",
                        "event_e2e_p95",
                }:
                    feedback_gate = self._wait_for_inference_deployment_feedback_samples(served_version)
                else:
                    feedback_gate = {
                        "enabled": bool(self.inference_cfg.deployment_require_version_matched_feedback),
                        "required": max(1, int(self.inference_cfg.deployment_min_version_matched_samples)),
                        "count": self._deployment_feedback_count(served_version),
                        "timed_out": False,
                        "skipped": False,
                        "guard_truncated": True,
                    }
                new_logic_feats, new_phys_feats, metrics, _, new_state_debug = self._collect_deployment_state(
                    prev_deploy_mask=prev_deploy_mask,
                    deployment_version=served_version,
                )
                cloud_idx = self.physical_topology.cloud_idx
                raw_edge_replicas = int(raw_deploy_mask[:, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
                decoded_mask_aux = aux.get("decoded_deploy_mask")
                decoded_edge_replicas = (
                    int(decoded_mask_aux.detach().cpu().bool()[:, :cloud_idx].sum().item())
                    if isinstance(decoded_mask_aux, torch.Tensor) and cloud_idx > 0 else raw_edge_replicas
                )
                edge_replicas = int(exec_deploy_mask[:, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
                cloud_replicas = int(exec_deploy_mask[:, cloud_idx].sum().item())
                layout_metrics = self._deployment_layout_metrics(exec_deploy_mask)
                cloud_only = int(layout_metrics["cloud_only"])
                cloud_only_ratio = float(layout_metrics["cloud_only_ratio"])
                empty_edge_devices = int(layout_metrics["empty_edge_devices"])
                empty_edge_device_ratio = float(layout_metrics["empty_edge_device_ratio"])
                aux["cloud_only_count"] = cloud_only
                aux["cloud_only_ratio"] = cloud_only_ratio
                aux["empty_edge_device_count"] = empty_edge_devices
                aux["empty_edge_device_ratio"] = empty_edge_device_ratio
                metrics["latency_guard_penalty_cost"] = 0.0
                feedback_required = max(1, int(feedback_gate.get("required", 1) or 1))
                feedback_count = int(feedback_gate.get("count", 0) or 0)
                feedback_shortfall = max(0, feedback_required - feedback_count)
                feedback_shortfall_ratio = float(feedback_shortfall) / float(feedback_required)
                metrics["feedback_timeout_penalty_cost"] = (
                    feedback_shortfall_ratio if bool(feedback_gate.get("timed_out", False)) else 0.0
                )
                dep_reward_breakdown = self._compute_deployment_reward_breakdown(metrics, aux)
                dep_reward_estimate = dep_reward_breakdown["reward"]
                state_record = self._state_record_metrics(state_logic_feats, state_phys_feats, state_debug_record)
                guard_record = self._latency_guard_record_metrics()
                with self._data_lock:
                    active_deployment_plan = copy.deepcopy(self.deployment_plan)
                if self.dep_recorder is not None:
                    row = dict(
                        step=step,
                        epoch=self._epoch,
                        decision_version=decision_version,
                        served_deployment_version=served_version,
                        decision_reason=current_decision_reason,
                        interval_s=self.deployment_interval,
                        deployment_decision_overhead_s=deployment_decision_overhead_s,
                        dep_reward_estimate=dep_reward_estimate,
                        avg_off_reward=metrics["avg_offloading_reward"],
                        off_reward_std=metrics["offloading_reward_std"],
                        off_reward_count=metrics["offloading_reward_count"],
                        dep_change_cost=metrics["deploy_change_cost"],
                        dep_latency_cost=dep_reward_breakdown["dep_latency_cost"],
                        dep_offload_term=dep_reward_breakdown["dep_offload_term"],
                        dep_latency_term=dep_reward_breakdown["dep_latency_term"],
                        dep_slo_term=dep_reward_breakdown["dep_slo_term"],
                        dep_change_term=dep_reward_breakdown["dep_change_term"],
                        dep_cloud_only_term=dep_reward_breakdown["dep_cloud_only_term"],
                        dep_capacity_relax_term=dep_reward_breakdown["dep_capacity_relax_term"],
                        dep_edge_cover_repair_term=dep_reward_breakdown["dep_edge_cover_repair_term"],
                        dep_hotspot_term=dep_reward_breakdown["dep_hotspot_term"],
                        dep_runtime_risk_term=dep_reward_breakdown["dep_runtime_risk_term"],
                        dep_unknown_option_term=dep_reward_breakdown["dep_unknown_option_term"],
                        dep_stale_option_term=dep_reward_breakdown["dep_stale_option_term"],
                        dep_low_quality_term=dep_reward_breakdown["dep_low_quality_term"],
                        dep_latency_guard_penalty_term=dep_reward_breakdown["dep_latency_guard_penalty_term"],
                        dep_feedback_timeout_term=dep_reward_breakdown["dep_feedback_timeout_term"],
                        active_pair_hotspot_cost=dep_reward_breakdown["active_pair_hotspot_cost"],
                        executed_active_pair_hotspot_cost=(
                            dep_reward_breakdown["executed_active_pair_hotspot_cost"]
                        ),
                        e2e_latency_count=metrics["e2e_latency_count"],
                        e2e_latency_mean=metrics["e2e_latency_mean"],
                        e2e_latency_latest=metrics["e2e_latency_latest"],
                        e2e_latency_p50=metrics["e2e_latency_p50"],
                        e2e_latency_p90=metrics["e2e_latency_p90"],
                        e2e_latency_p95=metrics["e2e_latency_p95"],
                        e2e_latency_p99=metrics["e2e_latency_p99"],
                        e2e_slo_violation=metrics["e2e_slo_violation"],
                        feedback_gate_enabled=int(bool(feedback_gate.get("enabled", False))),
                        feedback_gate_required_samples=feedback_required,
                        feedback_gate_collected_samples=feedback_count,
                        feedback_gate_sample_shortfall=feedback_shortfall,
                        feedback_gate_shortfall_ratio=feedback_shortfall_ratio,
                        feedback_gate_timed_out=int(bool(feedback_gate.get("timed_out", False))),
                        feedback_gate_guard_truncated=int(bool(feedback_gate.get("guard_truncated", False))),
                        feedback_timeout_penalty_cost=dep_reward_breakdown["feedback_timeout_penalty_cost"],
                        deployment_event_triggered=int(bool(current_event_status.get("triggered", False))),
                        deployment_event_reason=current_event_status.get("reason", ""),
                        deployment_event_queue_pressure=float(current_event_status.get("queue_pressure", 0.0) or 0.0),
                        deployment_event_hotspot_pressure=float(
                            current_event_status.get("hotspot_pressure", 0.0) or 0.0
                        ),
                        deployment_event_max_queue=float(current_event_status.get("max_queue", 0.0) or 0.0),
                        deployment_event_e2e_slo_violation=float(
                            current_event_status.get("e2e_slo_violation", 0.0) or 0.0
                        ),
                        deployment_event_e2e_p95=float(current_event_status.get("e2e_p95", 0.0) or 0.0),
                        deployment_event_e2e_feedback_count=int(
                            current_event_status.get("e2e_feedback_count", 0) or 0
                        ),
                        deployment_event_suppressed=int(bool(
                            current_event_status.get("deployment_event_suppressed", False)
                        )),
                        deployment_event_suppress_reason=str(
                            current_event_status.get("deployment_event_suppress_reason", "") or ""
                        ),
                        deployment_event_suppressed_event_reason=str(
                            current_event_status.get("deployment_event_suppressed_event_reason", "") or ""
                        ),
                        deployment_event_warmup_remaining_s=float(
                            current_event_status.get("deployment_event_warmup_remaining_s", 0.0) or 0.0
                        ),
                        deployment_event_feedback_count=int(
                            current_event_status.get("deployment_event_feedback_count", 0) or 0
                        ),
                        deployment_event_feedback_required=int(
                            current_event_status.get("deployment_event_feedback_required", 0) or 0
                        ),
                        deployment_deterministic=int(bool(self.inference_cfg.deployment_deterministic)),
                        cap_relax_cnt=aux["capacity_relax_cnt"],
                        cap_relax_cost=aux["capacity_relax_cost"],
                        edge_cover_repair_cnt=aux.get("edge_cover_repair_cnt", 0),
                        edge_cover_repair_cost=aux.get("edge_cover_repair_cost", 0.0),
                        edge_cover_unmet=aux.get("edge_cover_unmet", 0),
                        hotspot_repair_cnt=aux.get("hotspot_repair_cnt", 0),
                        hotspot_repair_cost=aux.get("hotspot_repair_cost", 0.0),
                        hotspot_unmet=aux.get("hotspot_unmet", 0),
                        policy_logp=self._scalar_value(logp),
                        policy_entropy=self._scalar_value(ent),
                        value_estimate=self._scalar_value(value),
                        raw_edge_replicas=raw_edge_replicas,
                        decoded_edge_replicas=decoded_edge_replicas,
                        edge_replicas=edge_replicas,
                        cloud_replicas=cloud_replicas,
                        raw_zero_edge_services=aux.get("raw_zero_edge_services", 0),
                        decoded_zero_edge_services=aux.get("decoded_zero_edge_services", 0),
                        matrix_added_cnt=aux.get("matrix_added_cnt", 0),
                        matrix_kept_cnt=aux.get("matrix_kept_cnt", 0),
                        matrix_removed_cnt=aux.get("matrix_removed_cnt", 0),
                        decode_added_cnt=aux.get("decode_added_cnt", 0),
                        decode_pruned_cnt=aux.get("decode_pruned_cnt", 0),
                        capacity_removed_cnt=aux.get("capacity_removed_cnt", 0),
                        selected_queue_pressure_cost=aux.get("selected_queue_pressure_cost", 0.0),
                        selected_runtime_risk_cost=aux.get("selected_runtime_risk_cost", 0.0),
                        selected_unknown_option_cost=aux.get("selected_unknown_option_cost", 0.0),
                        selected_stale_option_cost=aux.get("selected_stale_option_cost", 0.0),
                        selected_runtime_weakness_cost=aux.get("selected_runtime_weakness_cost", 0.0),
                        selected_low_quality_option_cost=aux.get("selected_low_quality_option_cost", 0.0),
                        selected_evidence_untrusted_cost=aux.get("selected_evidence_untrusted_cost", 0.0),
                        selected_risky_pair_count=aux.get("selected_risky_pair_count", 0.0),
                        selected_low_quality_pair_count=aux.get("selected_low_quality_pair_count", 0.0),
                        cloud_only=cloud_only,
                        cloud_only_ratio=cloud_only_ratio,
                        empty_edge_devices=empty_edge_devices,
                        empty_edge_device_ratio=empty_edge_device_ratio,
                        raw_deployment_plan=self._json_for_record(raw_deploy_plan),
                        deployment_plan=self._json_for_record(deploy_plan),
                        active_deployment_plan=self._json_for_record(active_deployment_plan),
                        **state_record,
                        **guard_record,
                        dep_offload_weight=self.deployment_agent_params["reward_dep_offload_weight"],
                        dep_latency_weight=self.deployment_agent_params["reward_dep_latency_weight"],
                        dep_latency_transform=self.deployment_agent_params["reward_dep_latency_transform"],
                        dep_latency_normalizer=self.deployment_agent_params["reward_dep_latency_normalizer"],
                        dep_latency_clip=self.deployment_agent_params["reward_dep_latency_clip"],
                        dep_slo_weight=self.deployment_agent_params["reward_dep_slo_weight"],
                        dep_change_weight=self.deployment_agent_params["reward_dep_change_weight"],
                        dep_cloud_only_weight=self.deployment_agent_params["reward_dep_cloud_only_weight"],
                        cap_relax_weight=self.deployment_agent_params["penalty_capacity_relax"],
                        edge_cover_repair_weight=self.deployment_agent_params["penalty_edge_cover_repair"],
                        hotspot_weight=self.deployment_agent_params["reward_dep_hotspot_weight"],
                        runtime_risk_weight=self.deployment_agent_params["reward_dep_runtime_risk_weight"],
                        unknown_option_weight=self.deployment_agent_params["reward_dep_unknown_option_weight"],
                        stale_option_weight=self.deployment_agent_params["reward_dep_stale_option_weight"],
                        low_quality_weight=self.deployment_agent_params["reward_dep_low_quality_weight"],
                        latency_guard_penalty_weight=self.deployment_agent_params["penalty_latency_guard_trigger"],
                        feedback_timeout_penalty_weight=self.deployment_agent_params["penalty_feedback_timeout"],
                        max_edge_replicas_per_device=self.deployment_agent_params["max_edge_replicas_per_device"],
                        edge_memory_budget_ratio=self.deployment_agent_params["edge_memory_budget_ratio"],
                        select_threshold=self.deployment_agent_params["select_threshold"],
                        negative_queue_threshold=self.deployment_agent_params["negative_queue_threshold"],
                        negative_hotspot_threshold=self.deployment_agent_params["negative_hotspot_threshold"],
                        negative_runtime_risk_threshold=(
                            self.deployment_agent_params["negative_runtime_risk_threshold"]
                        ),
                        negative_unknown_threshold=self.deployment_agent_params["negative_unknown_threshold"],
                        negative_stale_threshold=self.deployment_agent_params["negative_stale_threshold"],
                        positive_quality_threshold=self.deployment_agent_params["positive_quality_threshold"],
                        queue_normalizer=self.deployment_agent_params["queue_normalizer"],
                        loaded_checkpoint=self._loaded_checkpoint_path,
                    )
                    if self.record_cfg.actor_snapshot_debug:
                        row["state_deployment_actor_snapshot"] = self._json_for_record(aux.get("actor_debug"))
                    self.dep_recorder.log_dict(row)
                self._log_deployment_decisions(
                    step=step,
                    raw_deploy_mask=raw_deploy_mask,
                    exec_deploy_mask=exec_deploy_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                    logic_edge_index=logic_edge_index,
                    actor_debug=aux.get("actor_debug"),
                    policy_deterministic=self.inference_cfg.deployment_deterministic,
                )
                LOGGER.debug(
                    f"[Hedger][Inference][Deployment] step={step}, "
                    f"pending_version={decision_version}, "
                    f"served_version={served_version}, "
                    f"decision_reason={current_decision_reason}, "
                    f"decision_overhead={self._format_log_value(deployment_decision_overhead_s, 6)}s, "
                    f"{self._summarize_deploy_mask(deploy_mask.detach().cpu())}, "
                    f"{self._summarize_deployment_plan(deploy_plan)}, "
                    f"{self._summarize_state_snapshot(new_logic_feats, new_phys_feats, metrics)}, "
                    f"capacity_relax_cnt={aux['capacity_relax_cnt']}, "
                    f"capacity_relax_cost={self._format_log_value(aux['capacity_relax_cost'])}, "
                    f"edge_cover_repair_cnt={aux.get('edge_cover_repair_cnt', 0)}, "
                    f"edge_cover_unmet={aux.get('edge_cover_unmet', 0)}"
                )
                prev_deploy_mask = self._current_deploy_mask()
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                state_debug = new_state_debug
                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Inference][Deployment] Worker loop error: {e}")
                time.sleep(0.5)

        if self.dep_recorder is not None:
            self.dep_recorder.close()
            self.dep_recorder = None
        if self.dep_decision_recorder is not None:
            self.dep_decision_recorder.close()
            self.dep_decision_recorder = None
        LOGGER.info("[Hedger][Inference][Deployment] Worker stopped.")

    def inference_offloading_agent(self):
        LOGGER.info(
            f"[Hedger][Inference][Offloading] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s, "
            f"deterministic={self.inference_cfg.offloading_deterministic}"
        )

        assert self.logical_topology is not None and self.physical_topology is not None, \
            "Topologies must be registered before starting offloading inference."

        self._wait_for_initial_inference_task_feedback(
            "Offloading",
            self.offloading_thread_stop_event,
        )
        if self.offloading_thread_stop_event.is_set():
            LOGGER.info("[Hedger][Inference][Offloading] Worker stopped during initial feedback wait.")
            return

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _, state_debug = self._collect_offloading_state()
        last_task_version = (
            self.state_buffer.get_task_observation_version()
            if self.state_buffer is not None else 0
        )

        step = 0
        while not self.offloading_thread_stop_event.is_set():
            try:
                state_logic_feats = logic_feats
                state_phys_feats = phys_feats
                state_debug_record = state_debug
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                static_mask = self._current_deploy_mask()
                static_mask_dev = static_mask.to(self.device)

                with self._model_lock:
                    self._sync_decision_device()
                    with self.offloading_overhead_estimator, torch.inference_mode():
                        actions, logp, ent, value, aux = self.offloading_agent.policy(
                            logic_edge_index=logic_edge_index,
                            logic_feats=logic_feats_dev,
                            phys_edge_index=phys_edge_index,
                            phys_feats=phys_feats_dev,
                            static_mask=static_mask_dev,
                            topo_order=None,
                            deterministic=self.inference_cfg.offloading_deterministic,
                        )
                        offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                        self._sync_decision_device()
                offloading_decision_overhead_s = self.get_offloading_decision_overhead()
                with self._data_lock:
                    self.offloading_plan = offloading_plan
                offloading_time_ticket = self._sleep_until_next_tick(
                    offloading_time_ticket,
                    self.offloading_interval,
                )

                task_feedback_summary = (
                    self.state_buffer.get_task_observation_deployment_summary(last_task_version)
                    if self.state_buffer is not None
                    else {
                        "current_version": last_task_version,
                        "count": 0,
                        "deployment_version_counts": {},
                        "dominant_deployment_version": None,
                        "unique_deployment_versions": 0,
                        "all_same_deployment_version": False,
                    }
                )
                current_task_version = int(task_feedback_summary["current_version"])
                has_new_feedback = current_task_version > last_task_version
                feedback_deployment_version = None
                if (
                        task_feedback_summary.get("all_same_deployment_version")
                        and task_feedback_summary.get("count", 0) > 0
                ):
                    feedback_deployment_version = task_feedback_summary.get("dominant_deployment_version")

                new_logic_feats, new_phys_feats, metrics, _, new_state_debug = self._collect_offloading_state(
                    since_task_version=last_task_version,
                    deployment_version=feedback_deployment_version,
                )
                if current_task_version > last_task_version:
                    last_task_version = current_task_version

                off_reward_breakdown = self._compute_offloading_reward_breakdown(metrics, aux=aux)
                latency_cost = off_reward_breakdown["latency_cost"]
                off_reward_estimate = off_reward_breakdown["reward"]
                feedback_recorded = False
                if (
                        self.inference_cfg.record_offloading_feedback
                        and self.state_buffer is not None
                        and feedback_deployment_version is not None
                        and has_new_feedback
                ):
                    feedback_recorded = self.state_buffer.add_offloading_reward(
                        off_reward_estimate,
                        task_version=current_task_version,
                        deployment_version=feedback_deployment_version,
                    )
                cloud_idx = self.physical_topology.cloud_idx
                actions_cpu = actions.detach().cpu()
                proposal_actions_cpu = aux["proposal_actions"].detach().cpu()
                proposal_cloud_fraction = (
                    float((proposal_actions_cpu == cloud_idx).float().mean().item())
                    if proposal_actions_cpu.numel() else 0.0
                )
                projected_cloud_fraction = (
                    float((actions_cpu == cloud_idx).float().mean().item())
                    if actions_cpu.numel() else 0.0
                )
                unique_targets = int(actions_cpu.unique().numel())
                feasible_counts = static_mask.detach().cpu().float().sum(dim=1)
                feasible_targets_mean = float(feasible_counts.mean().item()) if feasible_counts.numel() else 0.0
                feasible_targets_min = float(feasible_counts.min().item()) if feasible_counts.numel() else 0.0
                feasible_targets_max = float(feasible_counts.max().item()) if feasible_counts.numel() else 0.0
                state_record = self._state_record_metrics(state_logic_feats, state_phys_feats, state_debug_record)
                guard_record = self._latency_guard_record_metrics()
                served_deployment_version = self.get_active_deployment_version()
                with self._data_lock:
                    active_deployment_plan = copy.deepcopy(self.deployment_plan)
                if self.off_recorder is not None:
                    row = dict(
                        step=step,
                        epoch=self._epoch,
                        served_deployment_version=served_deployment_version,
                        interval_s=self.offloading_interval,
                        offloading_decision_overhead_s=offloading_decision_overhead_s,
                        off_reward_estimate=off_reward_estimate,
                        latency=metrics["latency"],
                        latency_cost=latency_cost,
                        off_latency_term=off_reward_breakdown["off_latency_term"],
                        slo_violation=metrics["slo_violation"],
                        off_slo_term=off_reward_breakdown["off_slo_term"],
                        cloud_fraction=metrics["cloud_fraction"],
                        task_latency_count=metrics["task_latency_count"],
                        latest_task_latency=metrics["latest_task_latency"],
                        off_cloud_term=off_reward_breakdown["off_cloud_term"],
                        off_projection_term=off_reward_breakdown["off_projection_term"],
                        off_queue_term=off_reward_breakdown["off_queue_term"],
                        off_queue_cost=off_reward_breakdown["off_queue_cost"],
                        off_queue_risk_cost=off_reward_breakdown["off_queue_risk_cost"],
                        policy_logp=self._scalar_value(logp),
                        policy_entropy=self._scalar_value(ent),
                        value_estimate=self._scalar_value(value),
                        proposal_cloud_fraction=proposal_cloud_fraction,
                        projected_cloud_fraction=projected_cloud_fraction,
                        offloading_projection_cnt=aux.get("projection_cnt", 0),
                        offloading_dependency_projection_cnt=aux.get("dependency_projection_cnt", 0),
                        offloading_infeasible_projection_cnt=aux.get("infeasible_projection_cnt", 0),
                        offloading_projection_cost=aux.get("projection_cost", 0.0),
                        off_selected_runtime_ratio=aux.get("selected_runtime_ratio", 0.0),
                        off_selected_runtime_recency=aux.get("selected_runtime_recency", 0.0),
                        off_selected_queue_freshness=aux.get("selected_queue_freshness", 0.0),
                        off_selected_speed_evidence=aux.get("selected_speed_evidence", 0.0),
                        off_selected_capacity_pressure=aux.get("selected_capacity_pressure", 0.0),
                        off_selected_pair_load=aux.get("selected_pair_load", 0.0),
                        off_selected_device_load=aux.get("selected_device_load", 0.0),
                        off_selected_load_pressure=aux.get("selected_load_pressure", 0.0),
                        off_selected_service_time_factor=aux.get("selected_service_time_factor", 0.0),
                        off_selected_base_queue_risk=aux.get("selected_base_queue_risk", 0.0),
                        off_selected_relative_queue_risk=aux.get("selected_relative_queue_risk", 0.0),
                        off_selected_overload_risk=aux.get("selected_overload_risk", 0.0),
                        off_selected_queue_risk_total=aux.get("selected_queue_risk_total", 0.0),
                        off_selected_planned_load_risk=aux.get("selected_planned_load_risk", 0.0),
                        off_selected_relative_planned_load_risk=aux.get("selected_relative_planned_load_risk", 0.0),
                        off_selected_dynamic_risk=aux.get("selected_dynamic_risk", 0.0),
                        off_selected_offered_load_pressure=aux.get("selected_offered_load_pressure", 0.0),
                        off_selected_offered_load_risk=aux.get("selected_offered_load_risk", 0.0),
                        off_selected_compute_relative_weakness=aux.get("selected_compute_relative_weakness", 0.0),
                        off_selected_runtime_relative_weakness=aux.get("selected_runtime_relative_weakness", 0.0),
                        off_selected_relative_weakness=aux.get("selected_relative_weakness", 0.0),
                        off_selected_weak_pressure=aux.get("selected_weak_pressure", 0.0),
                        off_selected_weak_replica_risk=aux.get("selected_weak_replica_risk", 0.0),
                        off_selected_runtime_confidence=aux.get("selected_runtime_confidence", 0.0),
                        unique_targets=unique_targets,
                        feasible_targets_mean=feasible_targets_mean,
                        offloading_deterministic=int(bool(self.inference_cfg.offloading_deterministic)),
                        feasible_targets_min=feasible_targets_min,
                        feasible_targets_max=feasible_targets_max,
                        offloading_plan=self._json_for_record(self.offloading_plan),
                        active_deployment_plan=self._json_for_record(active_deployment_plan),
                        **state_record,
                        **guard_record,
                        feedback_task_observations=task_feedback_summary.get("count", 0),
                        feedback_deployment_version=feedback_deployment_version,
                        feedback_deployment_versions=self._json_for_record(
                            {
                                str(key): value
                                for key, value in task_feedback_summary.get("deployment_version_counts", {}).items()
                            }
                        ),
                        feedback_recorded=feedback_recorded,
                        off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                        off_latency_transform=self.offloading_agent_params["reward_off_latency_transform"],
                        off_latency_normalizer=self.offloading_agent_params["reward_off_latency_normalizer"],
                        off_latency_clip=self.offloading_agent_params["reward_off_latency_clip"],
                        off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                        off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                        off_projection_weight=self.offloading_agent_params["reward_off_projection_weight"],
                        off_queue_weight=self.offloading_agent_params["reward_off_queue_weight"],
                        off_queue_clip=self.offloading_agent_params["reward_off_queue_clip"],
                        loaded_checkpoint=self._loaded_checkpoint_path,
                    )
                    if self.record_cfg.actor_snapshot_debug:
                        row["state_offloading_actor_snapshot"] = self._json_for_record(aux.get("actor_debug"))
                    self.off_recorder.log_dict(row)
                self._log_offloading_decisions(
                    step=step,
                    actions=actions_cpu,
                    static_mask=static_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                    proposal_actions=proposal_actions_cpu,
                    projection_reasons=aux.get("projection_reasons"),
                    actor_debug=aux.get("actor_debug"),
                    policy_deterministic=self.inference_cfg.offloading_deterministic,
                )
                LOGGER.debug(
                    f"[Hedger][Inference][Offloading] step={step}, "
                    f"decision_overhead={self._format_log_value(offloading_decision_overhead_s, 6)}s, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(new_logic_feats, new_phys_feats, metrics)}, "
                    f"projection_cnt={aux.get('projection_cnt', 0)}, "
                    f"dependency_projection_cnt={aux.get('dependency_projection_cnt', 0)}, "
                    f"infeasible_projection_cnt={aux.get('infeasible_projection_cnt', 0)}"
                )
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                state_debug = new_state_debug
                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Inference][Offloading] Worker loop error: {e}")
                time.sleep(0.5)

        if self.off_recorder is not None:
            self.off_recorder.close()
            self.off_recorder = None
        if self.off_decision_recorder is not None:
            self.off_decision_recorder.close()
            self.off_decision_recorder = None
        LOGGER.info("[Hedger][Inference][Offloading] Worker stopped.")

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
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)
        elif self.deployment_plan is None:
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self._current_deploy_mask())

        LOGGER.info(
            f"[Hedger][Train] Initial deployment state: {self._summarize_deploy_mask(self.cur_deploy_mask)}"
        )

        if self.stage_cfg.deployment_train_mode == "offline":
            self.train_deployment_offline()
            return

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

        if self.stage_cfg.update_deployment_policy:
            self.dep_update_recorder = Recorder(
                self._stage_log_path("deployment_ppo_updates.csv"),
                fmt="csv",
                fieldnames=self._ppo_update_fieldnames(
                    include_offline_batch=self.stage_cfg.deployment_train_mode == "online"
                ),
                overwrite=True,
                flush_every=1,
            )
        if self.stage_cfg.update_offloading_policy:
            self.off_update_recorder = Recorder(
                self._stage_log_path("offloading_ppo_updates.csv"),
                fmt="csv",
                fieldnames=self._ppo_update_fieldnames(),
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

                if self.is_latency_guard_active():
                    if self.latency_guard_cfg.clear_transition_buffers:
                        self._clear_training_transition_buffers("latency guard active")
                    self._sleep_while_latency_guard_active("coordinator")
                    continue

                updates_in_tick = 0

                # Run a PPO update for the offloading agent.
                if self.stage_cfg.update_offloading_policy and \
                        len(self.offloading_transitions) >= self.training_cfg.offloading_rollout_len:
                    with self._data_lock:
                        off_transitions = self.offloading_transitions[:self.training_cfg.offloading_rollout_len]

                    with self._model_lock:
                        off_ppo_cfg = self.offloading_agent_params["ppo"]
                        off_entropy_coef = self._scheduled_entropy_coef(
                            off_ppo_cfg,
                            self._offloading_update_steps + 1,
                        )
                        off_ppo_stats = self.offloading_agent.ppo_update(
                            off_transitions,
                            epochs=self.training_cfg.ppo_epochs,
                            batch_size=self.training_cfg.offloading_batch_size,
                            entropy_coef=off_entropy_coef,
                            value_coef=off_ppo_cfg.value_coef,
                        )
                    with self._data_lock:
                        del self.offloading_transitions[:len(off_transitions)]
                        off_remaining = len(self.offloading_transitions)
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
                        f"entropy_coef={self._format_log_value(off_ppo_stats.get('entropy_coef', 0.0))}, "
                        f"approx_kl={self._format_log_value(off_ppo_stats.get('approx_kl', 0.0))}, "
                        f"clip_fraction={self._format_log_value(off_ppo_stats.get('clip_fraction', 0.0))}"
                    )

                # Run a deployment update. The online deployment stage uses
                # replay-style AWAC updates over offline+fresh macro transitions;
                # joint fine-tuning keeps the original online PPO update.
                if self.stage_cfg.update_deployment_policy and \
                        len(self.deployment_transitions) >= self.training_cfg.deployment_rollout_len:
                    if self.stage_cfg.deployment_train_mode == "online":
                        new_online = (
                            self._deployment_collected_transition_count
                            - self._deployment_last_online_update_transition_count
                        )
                        if new_online >= self.training_cfg.deployment_offline_rl.online_min_new_transitions:
                            dep_transitions = self._sample_deployment_online_replay_batch()
                            if dep_transitions:
                                batch_quality = summarize_transition_quality(dep_transitions)
                                with self._model_lock:
                                    dep_ppo_stats = self.deployment_agent.offline_update(
                                        dep_transitions,
                                        batch_size=len(dep_transitions),
                                        **self._deployment_offline_update_kwargs(),
                                    )
                                if dep_ppo_stats is not None:
                                    dep_ppo_stats.update(batch_quality)
                                with self._data_lock:
                                    dep_remaining = len(self.deployment_transitions)
                                self._deployment_last_online_update_transition_count = (
                                    self._deployment_collected_transition_count
                                )
                                self._deployment_update_steps += 1
                                updates_in_tick += 1
                                self._record_ppo_update(
                                    self.dep_update_recorder,
                                    "deployment_online",
                                    self._deployment_update_steps,
                                    used=len(dep_transitions),
                                    remaining=dep_remaining,
                                    stats=dep_ppo_stats,
                                )
                                LOGGER.info(
                                    f"[Hedger][Train][DeploymentOnline] replay_update="
                                    f"{self._deployment_update_steps}, used={len(dep_transitions)}, "
                                    f"online_buffer={dep_remaining}, offline_samples="
                                    f"{len(self.deployment_offline_dataset) if self.deployment_offline_dataset else 0}, "
                                    f"reward_mean={self._format_log_value(dep_ppo_stats.get('reward_mean', 0.0))}, "
                                    f"batch_bad_ratio="
                                    f"{self._format_log_value(dep_ppo_stats.get('offline_batch_bad_ratio', 0.0))}, "
                                    f"policy_loss={self._format_log_value(dep_ppo_stats.get('policy_loss', 0.0))}, "
                                    f"value_loss={self._format_log_value(dep_ppo_stats.get('value_loss', 0.0))}, "
                                    f"adv_mean={self._format_log_value(dep_ppo_stats.get('adv_mean', 0.0))}"
                                )
                    else:
                        with self._data_lock:
                            dep_transitions = self.deployment_transitions[:self.training_cfg.deployment_rollout_len]

                        with self._model_lock:
                            dep_ppo_cfg = self.deployment_agent_params["ppo"]
                            dep_entropy_coef = self._scheduled_entropy_coef(
                                dep_ppo_cfg,
                                self._deployment_update_steps + 1,
                            )
                            dep_ppo_stats = self.deployment_agent.ppo_update(
                                dep_transitions,
                                epochs=self.training_cfg.ppo_epochs,
                                batch_size=self.training_cfg.deployment_batch_size,
                                entropy_coef=dep_entropy_coef,
                                value_coef=dep_ppo_cfg.value_coef,
                            )
                        with self._data_lock:
                            del self.deployment_transitions[:len(dep_transitions)]
                            dep_remaining = len(self.deployment_transitions)
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
                            f"entropy_coef={self._format_log_value(dep_ppo_stats.get('entropy_coef', 0.0))}, "
                            f"approx_kl={self._format_log_value(dep_ppo_stats.get('approx_kl', 0.0))}, "
                            f"clip_fraction={self._format_log_value(dep_ppo_stats.get('clip_fraction', 0.0))}"
                        )

                if self.stage_cfg.deployment_train_mode == "collect" and self.deployment_dataset_writer is not None:
                    collect_step = self.deployment_dataset_writer.count
                    if collect_step > self._epoch:
                        prev_epoch = self._epoch
                        self._epoch = collect_step
                        save_interval = self.checkpoint_cfg.save.interval_updates
                        if (prev_epoch // save_interval) != (self._epoch // save_interval):
                            try:
                                self.save_checkpoint(stage_step=self._epoch, is_final=False)
                                self._deployment_last_collect_checkpoint_step = self._epoch
                            except Exception as e:
                                LOGGER.exception(
                                    f"[Hedger][Train] Failed to save collect checkpoint "
                                    f"at stage_step={self._epoch}: {e}"
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
        if self.deployment_dataset_writer is not None:
            self.deployment_dataset_writer.close()
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
            f"update_policy={self.stage_cfg.update_deployment_policy}, "
            f"rollout_deterministic={self.training_cfg.deployment_rollout_deterministic}"
        )

        dep_train_fieldnames = ["step", "epoch", "decision_version", "dep_updates", "dep_reward",
                                "decision_reason",
                                "avg_off_reward", "off_reward_std", "off_reward_count", "dep_change_cost",
                                "dep_latency_cost", "dep_offload_term", "dep_latency_term",
                                "dep_slo_term", "dep_change_term", "dep_cloud_only_term",
                                "dep_capacity_relax_term", "dep_edge_cover_repair_term",
                                "dep_hotspot_term", "dep_runtime_risk_term",
                                "dep_unknown_option_term", "dep_stale_option_term",
                                "dep_low_quality_term",
                                "dep_latency_guard_penalty_term", "dep_feedback_timeout_term",
                                "active_pair_hotspot_cost", "executed_active_pair_hotspot_cost",
                                "e2e_latency_count", "e2e_latency_mean", "e2e_latency_latest",
                                "e2e_latency_p50", "e2e_latency_p90", "e2e_latency_p95",
                                "e2e_latency_p99", "e2e_slo_violation",
                                "feedback_required_samples", "feedback_sample_shortfall",
                                "feedback_shortfall_ratio", "feedback_timed_out",
                                "feedback_timeout_s", "feedback_timeout_penalty_cost",
                                "feedback_guard_interrupted",
                                "latency_guard_trigger_seq", "latency_guard_bad_ratio",
                                "latency_guard_bad_count", "latency_guard_sample_count",
                                "latency_guard_max_queue", "latency_guard_penalty_cost",
                                "deployment_event_triggered", "deployment_event_reason",
                                "deployment_event_queue_pressure", "deployment_event_hotspot_pressure",
                                "deployment_event_max_queue",
                                "deployment_event_e2e_slo_violation", "deployment_event_e2e_p95",
                                "deployment_event_e2e_feedback_count",
                                "cap_relax_cnt", "cap_relax_cost",
                                "edge_cover_repair_cnt", "edge_cover_repair_cost", "edge_cover_unmet",
                                "hotspot_repair_cnt", "hotspot_repair_cost", "hotspot_unmet",
                                "policy_logp", "policy_entropy", "value_estimate", "next_value",
                                "behavior_kind",
                                "collect_behavior", "collect_operation", "collect_attempts",
                                "collect_reject_cnt", "collect_reject_reasons",
                                "collect_reset_triggered", "collect_bad_streak",
                                "collect_quality_bucket", "collect_anchor_used",
                                "collect_fallback_selected_best",
                                "collect_predicted_risk",
                                "collect_base_queue_pressure", "collect_base_hotspot_cost",
                                "collect_base_runtime_risk", "collect_base_min_pair_quality",
                                "collect_candidate_queue_pressure",
                                "collect_candidate_hotspot_cost", "collect_candidate_runtime_risk",
                                "collect_candidate_min_pair_quality",
                                "collect_delta_queue_pressure", "collect_delta_hotspot_cost",
                                "collect_delta_runtime_risk", "collect_delta_min_pair_quality",
                                "collect_new_edge_count", "collect_removed_edge_count",
                                "collect_new_edge_runtime_risk", "collect_new_edge_min_pair_quality",
                                "collect_base_low_quality_count", "collect_candidate_low_quality_count",
                                "collect_delta_low_quality_count", "collect_base_risky_pair_count",
                                "collect_candidate_risky_pair_count",
                                "collect_raw_change_count", "collect_exec_change_count",
                                "deployment_rollout_deterministic",
                                "raw_edge_replicas", "decoded_edge_replicas", "edge_replicas", "cloud_replicas",
                                "raw_zero_edge_services", "decoded_zero_edge_services",
                                "matrix_added_cnt", "matrix_kept_cnt", "matrix_removed_cnt",
                                "decode_added_cnt", "decode_pruned_cnt",
                                "capacity_removed_cnt", "selected_queue_pressure_cost",
                                "selected_runtime_risk_cost", "selected_unknown_option_cost",
                                "selected_stale_option_cost", "selected_runtime_weakness_cost",
                                "selected_low_quality_option_cost", "selected_evidence_untrusted_cost",
                                "selected_risky_pair_count", "selected_low_quality_pair_count",
                                "cloud_only", "cloud_only_ratio",
                                "empty_edge_devices", "empty_edge_device_ratio",
                                "transition_buffer", "dataset_transitions",
                                "raw_deployment_plan", "deployment_plan",
                                *self._state_record_fieldnames()]
        if self.record_cfg.actor_snapshot_debug:
            dep_train_fieldnames.append("state_deployment_actor_snapshot")
        dep_train_fieldnames.extend([
            "dep_offload_weight", "dep_latency_weight", "dep_latency_transform",
            "dep_latency_normalizer", "dep_latency_clip", "dep_slo_weight",
            "dep_change_weight", "dep_cloud_only_weight", "cap_relax_weight", "edge_cover_repair_weight",
            "hotspot_weight", "runtime_risk_weight", "unknown_option_weight",
            "stale_option_weight", "low_quality_weight",
            "latency_guard_penalty_weight", "feedback_timeout_penalty_weight",
            "max_edge_replicas_per_device", "edge_memory_budget_ratio",
            "select_threshold", "negative_queue_threshold", "negative_hotspot_threshold",
            "negative_runtime_risk_threshold", "negative_unknown_threshold",
            "negative_stale_threshold", "positive_quality_threshold",
            "queue_normalizer",
            "deployment_default_warmup_enabled",
            "deployment_default_warmup_min_intervals",
            "deployment_default_warmup_min_feedback_samples",
        ])
        self.dep_recorder = Recorder(
            self._stage_log_path("deployment_train.csv"),
            fmt="csv",
            fieldnames=dep_train_fieldnames,
            overwrite=True,
            flush_every=1,
        )
        self.dep_decision_recorder = Recorder(
            self._stage_log_path("deployment_decisions.csv"),
            fmt="csv",
            fieldnames=self._deployment_decision_fieldnames(),
            overwrite=True,
            flush_every=10,
        )

        # Static graph edge indices can be reused across iterations.
        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)

        self._run_deployment_default_warmup()
        if self.deployment_thread_stop_event.is_set():
            self.dep_recorder.close()
            self.dep_decision_recorder.close()
            LOGGER.info("[Hedger][Train][Deployment] Worker stopped during default-deployment warmup.")
            return

        step = 0

        prev_deploy_mask = copy.deepcopy(self.cur_deploy_mask)
        logic_feats, phys_feats, _, _, state_debug = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
        last_guard_trigger_seq = self._latency_guard_trigger_seq_value()
        next_decision_reason = "startup"
        next_event_status: Dict[str, Any] = {"triggered": False, "reason": "startup"}

        while not self.deployment_thread_stop_event.is_set():
            try:
                if self._sleep_while_latency_guard_active("deployment worker"):
                    prev_deploy_mask = self._current_deploy_mask()
                    logic_feats, phys_feats, _, _, state_debug = self._collect_deployment_state(
                        prev_deploy_mask=prev_deploy_mask
                    )
                    continue

                current_decision_reason = next_decision_reason
                current_event_status = next_event_status
                # Move features onto the active device.
                state_logic_feats = logic_feats
                state_phys_feats = phys_feats
                state_debug_record = state_debug
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_deploy_mask_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None

                with self._model_lock, torch.no_grad():
                    # Sample a raw deployment action, then execute the corrected
                    # deployment mask returned by the policy.
                    deploy_mask, logp, ent, value, aux = self._sample_deployment_action_for_training(
                        logic_edge_index,
                        logic_feats_dev,
                        phys_edge_index,
                        phys_feats_dev,
                        prev_deploy_mask_dev,
                    )
                deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                with self._data_lock:
                    self.pending_deployment_plan = deploy_plan
                    self.pending_deploy_mask = deploy_mask.detach().cpu()
                    self._pending_deployment_force_serve = current_decision_reason in {
                        "latency_guard_trigger",
                        "event_queue_pressure",
                        "event_pair_hotspot",
                        "event_e2e_slo",
                        "event_e2e_p95",
                    }
                    self._pending_deployment_reason = current_decision_reason
                    decision_version = self._mark_deployment_decision_pending()

                if not self._wait_for_deployment_decision_served(decision_version, abort_on_guard=False):
                    break

                min_feedback_required = self._deployment_feedback_min_samples()
                feedback_result = self._wait_for_deployment_feedback_samples_result(
                    1,
                    deployment_version=decision_version,
                    allow_guard_truncated=True,
                )
                if not feedback_result.ok:
                    break

                if feedback_result.guard_interrupted:
                    next_decision_reason = "latency_guard_trigger"
                    next_event_status = {
                        "triggered": True,
                        "reason": "latency_guard_trigger",
                        "queue_pressure": 0.0,
                        "hotspot_pressure": 0.0,
                        "max_queue": 0.0,
                    }
                    LOGGER.warning(
                        f"[Hedger][Train][Deployment] decision_version={decision_version} "
                        f"triggered latency guard before a full active deployment interval; "
                        "record it as a guard-truncated negative sample."
                    )
                else:
                    LOGGER.debug(
                        f"[Hedger][Train][Deployment] decision_version={decision_version} "
                        f"has version-matched fresh feedback; start active deployment interval."
                    )

                    (
                        _deployment_time_ticket,
                        interval_end_reason,
                        last_guard_trigger_seq,
                        interval_event_status,
                    ) = self._sleep_until_next_inference_deployment_decision(
                        0,
                        self.deployment_interval,
                        last_guard_trigger_seq,
                    )
                    next_decision_reason = interval_end_reason
                    next_event_status = interval_event_status
                    feedback_result = self._wait_for_deployment_feedback_samples_result(
                        min_feedback_required,
                        deployment_version=decision_version,
                        allow_guard_truncated=True,
                    )
                    if not feedback_result.ok:
                        break

                new_logic_feats, new_phys_feats, metrics, done, new_state_debug = self._collect_deployment_state(
                    prev_deploy_mask=prev_deploy_mask,
                    deployment_version=decision_version,
                )
                metrics = self._attach_deployment_feedback_status(
                    metrics,
                    feedback_result,
                    min_feedback_required,
                )
                event_status_record = next_event_status if next_event_status.get("triggered") else current_event_status
                metrics.update({
                    "deployment_event_triggered": int(bool(event_status_record.get("triggered", False))),
                    "deployment_event_reason": str(event_status_record.get("reason", "") or ""),
                    "deployment_event_queue_pressure": float(event_status_record.get("queue_pressure", 0.0) or 0.0),
                    "deployment_event_hotspot_pressure": float(
                        event_status_record.get("hotspot_pressure", 0.0) or 0.0
                    ),
                    "deployment_event_max_queue": float(event_status_record.get("max_queue", 0.0) or 0.0),
                    "deployment_event_e2e_slo_violation": float(
                        event_status_record.get("e2e_slo_violation", 0.0) or 0.0
                    ),
                    "deployment_event_e2e_p95": float(event_status_record.get("e2e_p95", 0.0) or 0.0),
                    "deployment_event_e2e_feedback_count": int(
                        event_status_record.get("e2e_feedback_count", 0) or 0
                    ),
                })
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

                cloud_idx = self.physical_topology.cloud_idx
                raw_deploy_mask = aux["raw_deploy_mask"].detach().cpu().bool()
                exec_deploy_mask = deploy_mask.detach().cpu().bool()
                raw_deploy_plan = self._map_deployment_mask_to_deployment_plan(raw_deploy_mask)
                raw_edge_replicas = int(raw_deploy_mask[:, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
                decoded_mask_aux = aux.get("decoded_deploy_mask")
                decoded_edge_replicas = (
                    int(decoded_mask_aux.detach().cpu().bool()[:, :cloud_idx].sum().item())
                    if isinstance(decoded_mask_aux, torch.Tensor) and cloud_idx > 0 else raw_edge_replicas
                )
                edge_replicas = int(exec_deploy_mask[:, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
                cloud_replicas = int(exec_deploy_mask[:, cloud_idx].sum().item())
                layout_metrics = self._deployment_layout_metrics(exec_deploy_mask)
                cloud_only = int(layout_metrics["cloud_only"])
                cloud_only_ratio = float(layout_metrics["cloud_only_ratio"])
                empty_edge_devices = int(layout_metrics["empty_edge_devices"])
                empty_edge_device_ratio = float(layout_metrics["empty_edge_device_ratio"])
                aux["cloud_only_count"] = cloud_only
                aux["cloud_only_ratio"] = cloud_only_ratio
                aux["empty_edge_device_count"] = empty_edge_devices
                aux["empty_edge_device_ratio"] = empty_edge_device_ratio

                # Compute the reward from environment metrics and policy-side auxiliaries.
                dep_reward_breakdown = self._compute_deployment_reward_breakdown(metrics, aux)
                reward = dep_reward_breakdown["reward"]
                if self.is_latency_guard_active() and not feedback_result.guard_interrupted:
                    logic_feats = new_logic_feats
                    phys_feats = new_phys_feats
                    prev_deploy_mask = self._current_deploy_mask()
                    continue
                collect_quality_bucket = self._deployment_collect_quality_bucket(metrics, feedback_result)
                if self.stage_cfg.deployment_train_mode == "collect":
                    self._update_deployment_collect_quality(collect_quality_bucket)
                aux["collect_quality_bucket"] = collect_quality_bucket
                aux["collect_bad_streak"] = int(self._deployment_collect_bad_streak)
                aux["collect_good_count"] = int(self._deployment_collect_good_count)
                aux["collect_bad_count"] = int(self._deployment_collect_bad_count)

                # Keep transition payloads on CPU to avoid cross-thread device issues.
                # Offline/online deployment learning is trained on the executed
                # post-projection action, while `raw_deploy_mask` is still saved
                # for correction analysis and optional ablations.
                tr = {
                    "logic_edge_index": logic_edge_index.cpu(),
                    "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                    "phys_edge_index": phys_edge_index.cpu(),
                    "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                    "next_logic_feats": {k: v.cpu() for k, v in new_logic_feats_dev.items()},
                    "next_phys_feats": {k: v.cpu() for k, v in new_phys_feats_dev.items()},
                    "deploy_mask": deploy_mask.detach().cpu(),
                    "raw_deploy_mask": aux["raw_deploy_mask"].detach().cpu(),
                    "positive_mask": aux.get("positive_mask", deploy_mask).detach().cpu()
                    if isinstance(aux.get("positive_mask"), torch.Tensor) else deploy_mask.detach().cpu(),
                    "negative_mask": aux.get("negative_mask", torch.zeros_like(deploy_mask)).detach().cpu()
                    if isinstance(aux.get("negative_mask"), torch.Tensor) else torch.zeros_like(deploy_mask).detach().cpu(),
                    "topo_order": None,  # Recomputed during evaluation if needed.
                    "prev_deploy_mask": prev_deploy_mask.cpu() if prev_deploy_mask is not None else None,
                    "logp": logp.detach().cpu(),
                    "value": value.detach().cpu(),
                    "next_value": float(next_value),
                    "reward": float(reward),
                    "done": bool(done),
                    "feedback_timed_out": bool(feedback_result.timed_out),
                    "feedback_timeout_penalty_cost": float(metrics["feedback_timeout_penalty_cost"]),
                    "feedback_guard_interrupted": bool(feedback_result.guard_interrupted),
                    "latency_guard_penalty_cost": float(metrics["latency_guard_penalty_cost"]),
                    "deployment_event_triggered": bool(metrics.get("deployment_event_triggered", 0)),
                    "deployment_event_reason": str(metrics.get("deployment_event_reason", "")),
                    "deployment_event_queue_pressure": float(metrics.get("deployment_event_queue_pressure", 0.0)),
                    "deployment_event_hotspot_pressure": float(
                        metrics.get("deployment_event_hotspot_pressure", 0.0)
                    ),
                    "behavior_kind": str(aux.get("behavior_kind", "actor")),
                    "decision_version": int(decision_version),
                    "feedback_count": int(feedback_result.count),
                    "reward_breakdown": dict(dep_reward_breakdown),
                    "metrics": dict(metrics),
                    "capacity_relax_cnt": int(aux["capacity_relax_cnt"]),
                    "edge_cover_repair_cnt": int(aux.get("edge_cover_repair_cnt", 0)),
                    "collect_behavior": str(aux.get("collect_behavior", aux.get("behavior_kind", ""))),
                    "collect_operation": str(aux.get("collect_operation", aux.get("behavior_kind", ""))),
                    "collect_quality_bucket": str(aux.get("collect_quality_bucket", "")),
                    "collect_predicted_risk": float(aux.get("collect_predicted_risk", 0.0)),
                    "collect_reject_cnt": int(aux.get("collect_reject_cnt", 0)),
                    "collect_reset_triggered": int(aux.get("collect_reset_triggered", 0)),
                    "collect_anchor_used": int(aux.get("collect_anchor_used", 0)),
                    "collect_bad_streak": int(aux.get("collect_bad_streak", 0)),
                    "collect_fallback_selected_best": int(aux.get("collect_fallback_selected_best", 0)),
                    "collect_candidate_queue_pressure": float(
                        aux.get("collect_candidate_queue_pressure", 0.0)
                    ),
                    "collect_candidate_hotspot_cost": float(
                        aux.get("collect_candidate_hotspot_cost", 0.0)
                    ),
                    "collect_candidate_runtime_risk": float(
                        aux.get("collect_candidate_runtime_risk", 0.0)
                    ),
                    "collect_candidate_min_pair_quality": float(
                        aux.get("collect_candidate_min_pair_quality", 0.0)
                    ),
                    "collect_base_queue_pressure": float(aux.get("collect_base_queue_pressure", 0.0)),
                    "collect_base_hotspot_cost": float(aux.get("collect_base_hotspot_cost", 0.0)),
                    "collect_base_runtime_risk": float(aux.get("collect_base_runtime_risk", 0.0)),
                    "collect_base_min_pair_quality": float(aux.get("collect_base_min_pair_quality", 0.0)),
                    "collect_delta_queue_pressure": float(aux.get("collect_delta_queue_pressure", 0.0)),
                    "collect_delta_hotspot_cost": float(aux.get("collect_delta_hotspot_cost", 0.0)),
                    "collect_delta_runtime_risk": float(aux.get("collect_delta_runtime_risk", 0.0)),
                    "collect_delta_min_pair_quality": float(aux.get("collect_delta_min_pair_quality", 0.0)),
                    "collect_new_edge_count": int(aux.get("collect_new_edge_count", 0)),
                    "collect_removed_edge_count": int(aux.get("collect_removed_edge_count", 0)),
                    "collect_new_edge_runtime_risk": float(aux.get("collect_new_edge_runtime_risk", 0.0)),
                    "collect_new_edge_min_pair_quality": float(aux.get("collect_new_edge_min_pair_quality", 0.0)),
                    "collect_base_low_quality_count": int(aux.get("collect_base_low_quality_count", 0)),
                    "collect_candidate_low_quality_count": int(aux.get("collect_candidate_low_quality_count", 0)),
                    "collect_delta_low_quality_count": int(aux.get("collect_delta_low_quality_count", 0)),
                    "collect_base_risky_pair_count": int(aux.get("collect_base_risky_pair_count", 0)),
                    "collect_candidate_risky_pair_count": int(aux.get("collect_candidate_risky_pair_count", 0)),
                    "collect_raw_change_count": int(aux.get("collect_raw_change_count", 0)),
                    "collect_exec_change_count": int(aux.get("collect_exec_change_count", 0)),
                    "deployment_rollout_deterministic": bool(
                        self.training_cfg.deployment_rollout_deterministic
                    ),
                    "offloading_rollout_deterministic": bool(
                        self.training_cfg.offloading_rollout_deterministic
                    ),
                }

                if self.deployment_dataset_writer is not None \
                        and self.stage_cfg.deployment_train_mode in {"collect", "online"}:
                    self._deployment_collected_transition_count = self.deployment_dataset_writer.append(tr)

                if self.stage_cfg.update_deployment_policy:
                    with self._data_lock:
                        self.deployment_transitions.append(tr)
                        capacity = self.training_cfg.deployment_offline_rl.online_replay_capacity
                        if len(self.deployment_transitions) > capacity:
                            del self.deployment_transitions[:len(self.deployment_transitions) - capacity]
                        transition_count = len(self.deployment_transitions)
                else:
                    with self._data_lock:
                        transition_count = len(self.deployment_transitions)

                state_record = self._state_record_metrics(state_logic_feats, state_phys_feats, state_debug_record)
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                state_debug = new_state_debug
                prev_deploy_mask = deploy_mask.detach().cpu()

                row = dict(
                    step=step,
                    epoch=self._epoch,
                    decision_version=decision_version,
                    dep_updates=self._deployment_update_steps,
                    dep_reward=reward,
                    decision_reason=current_decision_reason,
                    avg_off_reward=metrics["avg_offloading_reward"],
                    off_reward_std=metrics["offloading_reward_std"],
                    off_reward_count=metrics["offloading_reward_count"],
                    dep_change_cost=metrics["deploy_change_cost"],
                    dep_latency_cost=dep_reward_breakdown["dep_latency_cost"],
                    dep_offload_term=dep_reward_breakdown["dep_offload_term"],
                    dep_latency_term=dep_reward_breakdown["dep_latency_term"],
                    dep_slo_term=dep_reward_breakdown["dep_slo_term"],
                    dep_change_term=dep_reward_breakdown["dep_change_term"],
                    dep_cloud_only_term=dep_reward_breakdown["dep_cloud_only_term"],
                    dep_capacity_relax_term=dep_reward_breakdown["dep_capacity_relax_term"],
                    dep_edge_cover_repair_term=dep_reward_breakdown["dep_edge_cover_repair_term"],
                    dep_hotspot_term=dep_reward_breakdown["dep_hotspot_term"],
                    dep_runtime_risk_term=dep_reward_breakdown["dep_runtime_risk_term"],
                    dep_unknown_option_term=dep_reward_breakdown["dep_unknown_option_term"],
                    dep_stale_option_term=dep_reward_breakdown["dep_stale_option_term"],
                    dep_low_quality_term=dep_reward_breakdown["dep_low_quality_term"],
                    dep_latency_guard_penalty_term=dep_reward_breakdown["dep_latency_guard_penalty_term"],
                    dep_feedback_timeout_term=dep_reward_breakdown["dep_feedback_timeout_term"],
                    active_pair_hotspot_cost=dep_reward_breakdown["active_pair_hotspot_cost"],
                    executed_active_pair_hotspot_cost=(
                        dep_reward_breakdown["executed_active_pair_hotspot_cost"]
                    ),
                    e2e_latency_count=metrics["e2e_latency_count"],
                    e2e_latency_mean=metrics["e2e_latency_mean"],
                    e2e_latency_latest=metrics["e2e_latency_latest"],
                    e2e_latency_p50=metrics["e2e_latency_p50"],
                    e2e_latency_p90=metrics["e2e_latency_p90"],
                    e2e_latency_p95=metrics["e2e_latency_p95"],
                    e2e_latency_p99=metrics["e2e_latency_p99"],
                    e2e_slo_violation=metrics["e2e_slo_violation"],
                    feedback_required_samples=metrics["feedback_required_samples"],
                    feedback_sample_shortfall=metrics["feedback_sample_shortfall"],
                    feedback_shortfall_ratio=metrics["feedback_shortfall_ratio"],
                    feedback_timed_out=metrics["feedback_timed_out"],
                    feedback_timeout_s=metrics["feedback_timeout_s"],
                    feedback_timeout_penalty_cost=metrics["feedback_timeout_penalty_cost"],
                    feedback_guard_interrupted=metrics["feedback_guard_interrupted"],
                    latency_guard_trigger_seq=metrics["latency_guard_trigger_seq"],
                    latency_guard_bad_ratio=metrics["latency_guard_bad_ratio"],
                    latency_guard_bad_count=metrics["latency_guard_bad_count"],
                    latency_guard_sample_count=metrics["latency_guard_sample_count"],
                    latency_guard_max_queue=metrics["latency_guard_max_queue"],
                    latency_guard_penalty_cost=metrics["latency_guard_penalty_cost"],
                    deployment_event_triggered=metrics.get("deployment_event_triggered", 0),
                    deployment_event_reason=metrics.get("deployment_event_reason", ""),
                    deployment_event_queue_pressure=metrics.get("deployment_event_queue_pressure", 0.0),
                    deployment_event_hotspot_pressure=metrics.get("deployment_event_hotspot_pressure", 0.0),
                    deployment_event_max_queue=metrics.get("deployment_event_max_queue", 0.0),
                    deployment_event_e2e_slo_violation=metrics.get("deployment_event_e2e_slo_violation", 0.0),
                    deployment_event_e2e_p95=metrics.get("deployment_event_e2e_p95", 0.0),
                    deployment_event_e2e_feedback_count=metrics.get("deployment_event_e2e_feedback_count", 0),
                    cap_relax_cnt=aux["capacity_relax_cnt"],
                    cap_relax_cost=aux["capacity_relax_cost"],
                    edge_cover_repair_cnt=aux.get("edge_cover_repair_cnt", 0),
                    edge_cover_repair_cost=aux.get("edge_cover_repair_cost", 0.0),
                    edge_cover_unmet=aux.get("edge_cover_unmet", 0),
                    hotspot_repair_cnt=aux.get("hotspot_repair_cnt", 0),
                    hotspot_repair_cost=aux.get("hotspot_repair_cost", 0.0),
                    hotspot_unmet=aux.get("hotspot_unmet", 0),
                    policy_logp=float(logp.detach().cpu().item()),
                    policy_entropy=float(ent.detach().cpu().item()),
                    value_estimate=float(value.detach().cpu().item()),
                    next_value=float(next_value),
                    behavior_kind=aux.get("behavior_kind", "actor"),
                    collect_behavior=aux.get("collect_behavior", aux.get("behavior_kind", "")),
                    collect_operation=aux.get("collect_operation", aux.get("behavior_kind", "")),
                    collect_attempts=aux.get("collect_attempts", 1),
                    collect_reject_cnt=aux.get("collect_reject_cnt", 0),
                    collect_reject_reasons=aux.get("collect_reject_reasons", ""),
                    collect_reset_triggered=aux.get("collect_reset_triggered", 0),
                    collect_bad_streak=aux.get("collect_bad_streak", 0),
                    collect_quality_bucket=aux.get("collect_quality_bucket", ""),
                    collect_anchor_used=aux.get("collect_anchor_used", 0),
                    collect_fallback_selected_best=aux.get("collect_fallback_selected_best", 0),
                    collect_predicted_risk=aux.get("collect_predicted_risk", 0.0),
                    collect_base_queue_pressure=aux.get("collect_base_queue_pressure", 0.0),
                    collect_base_hotspot_cost=aux.get("collect_base_hotspot_cost", 0.0),
                    collect_base_runtime_risk=aux.get("collect_base_runtime_risk", 0.0),
                    collect_base_min_pair_quality=aux.get("collect_base_min_pair_quality", 0.0),
                    collect_candidate_queue_pressure=aux.get("collect_candidate_queue_pressure", 0.0),
                    collect_candidate_hotspot_cost=aux.get("collect_candidate_hotspot_cost", 0.0),
                    collect_candidate_runtime_risk=aux.get("collect_candidate_runtime_risk", 0.0),
                    collect_candidate_min_pair_quality=aux.get("collect_candidate_min_pair_quality", 0.0),
                    collect_delta_queue_pressure=aux.get("collect_delta_queue_pressure", 0.0),
                    collect_delta_hotspot_cost=aux.get("collect_delta_hotspot_cost", 0.0),
                    collect_delta_runtime_risk=aux.get("collect_delta_runtime_risk", 0.0),
                    collect_delta_min_pair_quality=aux.get("collect_delta_min_pair_quality", 0.0),
                    collect_new_edge_count=aux.get("collect_new_edge_count", 0),
                    collect_removed_edge_count=aux.get("collect_removed_edge_count", 0),
                    collect_new_edge_runtime_risk=aux.get("collect_new_edge_runtime_risk", 0.0),
                    collect_new_edge_min_pair_quality=aux.get("collect_new_edge_min_pair_quality", 0.0),
                    collect_base_low_quality_count=aux.get("collect_base_low_quality_count", 0),
                    collect_candidate_low_quality_count=aux.get("collect_candidate_low_quality_count", 0),
                    collect_delta_low_quality_count=aux.get("collect_delta_low_quality_count", 0),
                    collect_base_risky_pair_count=aux.get("collect_base_risky_pair_count", 0),
                    collect_candidate_risky_pair_count=aux.get("collect_candidate_risky_pair_count", 0),
                    collect_raw_change_count=aux.get("collect_raw_change_count", 0),
                    collect_exec_change_count=aux.get("collect_exec_change_count", 0),
                    deployment_rollout_deterministic=int(
                        bool(self.training_cfg.deployment_rollout_deterministic)
                    ),
                    raw_edge_replicas=raw_edge_replicas,
                    decoded_edge_replicas=decoded_edge_replicas,
                    edge_replicas=edge_replicas,
                    cloud_replicas=cloud_replicas,
                    raw_zero_edge_services=aux.get("raw_zero_edge_services", 0),
                    decoded_zero_edge_services=aux.get("decoded_zero_edge_services", 0),
                    matrix_added_cnt=aux.get("matrix_added_cnt", 0),
                    matrix_kept_cnt=aux.get("matrix_kept_cnt", 0),
                    matrix_removed_cnt=aux.get("matrix_removed_cnt", 0),
                    decode_added_cnt=aux.get("decode_added_cnt", 0),
                    decode_pruned_cnt=aux.get("decode_pruned_cnt", 0),
                    capacity_removed_cnt=aux.get("capacity_removed_cnt", 0),
                    selected_queue_pressure_cost=aux.get("selected_queue_pressure_cost", 0.0),
                    selected_runtime_risk_cost=aux.get("selected_runtime_risk_cost", 0.0),
                    selected_unknown_option_cost=aux.get("selected_unknown_option_cost", 0.0),
                    selected_stale_option_cost=aux.get("selected_stale_option_cost", 0.0),
                    selected_runtime_weakness_cost=aux.get("selected_runtime_weakness_cost", 0.0),
                    selected_low_quality_option_cost=aux.get("selected_low_quality_option_cost", 0.0),
                    selected_evidence_untrusted_cost=aux.get("selected_evidence_untrusted_cost", 0.0),
                    selected_risky_pair_count=aux.get("selected_risky_pair_count", 0.0),
                    selected_low_quality_pair_count=aux.get("selected_low_quality_pair_count", 0.0),
                    cloud_only=cloud_only,
                    cloud_only_ratio=cloud_only_ratio,
                    empty_edge_devices=empty_edge_devices,
                    empty_edge_device_ratio=empty_edge_device_ratio,
                    transition_buffer=transition_count,
                    dataset_transitions=self._deployment_collected_transition_count,
                    raw_deployment_plan=self._json_for_record(raw_deploy_plan),
                    deployment_plan=self._json_for_record(self.deployment_plan),
                    **state_record,
                    dep_offload_weight=self.deployment_agent_params["reward_dep_offload_weight"],
                    dep_latency_weight=self.deployment_agent_params["reward_dep_latency_weight"],
                    dep_latency_transform=self.deployment_agent_params["reward_dep_latency_transform"],
                    dep_latency_normalizer=self.deployment_agent_params["reward_dep_latency_normalizer"],
                    dep_latency_clip=self.deployment_agent_params["reward_dep_latency_clip"],
                    dep_slo_weight=self.deployment_agent_params["reward_dep_slo_weight"],
                    dep_change_weight=self.deployment_agent_params["reward_dep_change_weight"],
                    dep_cloud_only_weight=self.deployment_agent_params["reward_dep_cloud_only_weight"],
                    cap_relax_weight=self.deployment_agent_params["penalty_capacity_relax"],
                    edge_cover_repair_weight=self.deployment_agent_params["penalty_edge_cover_repair"],
                    hotspot_weight=self.deployment_agent_params["reward_dep_hotspot_weight"],
                    runtime_risk_weight=self.deployment_agent_params["reward_dep_runtime_risk_weight"],
                    unknown_option_weight=self.deployment_agent_params["reward_dep_unknown_option_weight"],
                    stale_option_weight=self.deployment_agent_params["reward_dep_stale_option_weight"],
                    low_quality_weight=self.deployment_agent_params["reward_dep_low_quality_weight"],
                    latency_guard_penalty_weight=self.deployment_agent_params["penalty_latency_guard_trigger"],
                    feedback_timeout_penalty_weight=self.deployment_agent_params["penalty_feedback_timeout"],
                    max_edge_replicas_per_device=self.deployment_agent_params["max_edge_replicas_per_device"],
                    edge_memory_budget_ratio=self.deployment_agent_params["edge_memory_budget_ratio"],
                    select_threshold=self.deployment_agent_params["select_threshold"],
                    negative_queue_threshold=self.deployment_agent_params["negative_queue_threshold"],
                    negative_hotspot_threshold=self.deployment_agent_params["negative_hotspot_threshold"],
                    negative_runtime_risk_threshold=(
                        self.deployment_agent_params["negative_runtime_risk_threshold"]
                    ),
                    negative_unknown_threshold=self.deployment_agent_params["negative_unknown_threshold"],
                    negative_stale_threshold=self.deployment_agent_params["negative_stale_threshold"],
                    positive_quality_threshold=self.deployment_agent_params["positive_quality_threshold"],
                    queue_normalizer=self.deployment_agent_params["queue_normalizer"],
                    deployment_default_warmup_enabled=self.training_cfg.deployment_default_warmup.enabled,
                    deployment_default_warmup_min_intervals=self.training_cfg.deployment_default_warmup.min_intervals,
                    deployment_default_warmup_min_feedback_samples=(
                        self.training_cfg.deployment_default_warmup.min_feedback_samples
                    ),
                )
                if self.record_cfg.actor_snapshot_debug:
                    row["state_deployment_actor_snapshot"] = self._json_for_record(aux.get("actor_debug"))
                self.dep_recorder.log_dict(row)
                self._log_deployment_decisions(
                    step=step,
                    raw_deploy_mask=raw_deploy_mask,
                    exec_deploy_mask=exec_deploy_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                    logic_edge_index=logic_edge_index,
                    actor_debug=aux.get("actor_debug"),
                    collect_debug=aux,
                    policy_deterministic=self.training_cfg.deployment_rollout_deterministic,
                )
                LOGGER.debug(
                    f"[Hedger][Train][Deployment] step={step}, reward={self._format_log_value(reward)}, "
                    f"decision_version={decision_version}, "
                    f"{self._summarize_deploy_mask(self.cur_deploy_mask)}, "
                    f"{self._summarize_deployment_plan(self.deployment_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"off_reward_count={metrics['offloading_reward_count']}, "
                    f"off_reward_std={self._format_log_value(metrics['offloading_reward_std'])}, "
                    f"feedback_guard_interrupted={metrics['feedback_guard_interrupted']}, "
                    f"latency_guard_penalty_cost="
                    f"{self._format_log_value(metrics['latency_guard_penalty_cost'])}, "
                    f"capacity_relax_cnt={aux['capacity_relax_cnt']}, "
                    f"capacity_relax_cost={self._format_log_value(aux['capacity_relax_cost'])}, "
                    f"edge_cover_repair_cnt={aux.get('edge_cover_repair_cnt', 0)}, "
                    f"edge_cover_unmet={aux.get('edge_cover_unmet', 0)}, "
                    f"update_policy={self.stage_cfg.update_deployment_policy}, transitions={transition_count}"
                )

                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train][Deployment] Worker loop error: {e}")
                time.sleep(0.5)

        self.dep_recorder.close()
        self.dep_decision_recorder.close()
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

        log_scope = "OffloadingPPO" if self.stage_cfg.update_offloading_policy else "OffloadingRollout"
        LOGGER.info(
            f"[Hedger][Train][{log_scope}] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s, "
            f"rollout={self.training_cfg.offloading_rollout_len}, "
            f"batch={self.training_cfg.offloading_batch_size}, "
            f"update_policy={self.stage_cfg.update_offloading_policy}, "
            f"rollout_deterministic={self.training_cfg.offloading_rollout_deterministic}, "
            f"rollout_agent={'frozen' if self._frozen_offloading_agent is not None else 'live'}"
        )

        off_train_fieldnames = ["step", "epoch", "off_updates", "off_reward", "latency", "latency_cost",
                                "off_latency_term", "off_latency_normalizer", "off_latency_clip", "off_latency_transform",
                                "slo_violation", "off_slo_term", "cloud_fraction", "off_cloud_term",
                                "off_projection_term",
                                "off_queue_term", "off_queue_cost", "off_queue_risk_cost",
                                "off_latency_weight", "off_slo_weight", "off_cloud_weight",
                                "off_projection_weight", "off_queue_weight", "off_queue_clip",
                                "policy_logp", "policy_entropy", "value_estimate",
                                "next_value", "offloading_rollout_deterministic",
                                "proposal_cloud_fraction", "projected_cloud_fraction",
                                "offloading_projection_cnt", "offloading_dependency_projection_cnt",
                                "offloading_infeasible_projection_cnt", "offloading_projection_cost",
                                "off_selected_runtime_ratio", "off_selected_runtime_recency",
                                "off_selected_queue_freshness", "off_selected_speed_evidence",
                                "off_selected_capacity_pressure", "off_selected_pair_load",
                                "off_selected_device_load", "off_selected_load_pressure",
                                "off_selected_service_time_factor", "off_selected_base_queue_risk",
                                "off_selected_relative_queue_risk", "off_selected_overload_risk",
                                "off_selected_queue_risk_total", "off_selected_planned_load_risk",
                                "off_selected_relative_planned_load_risk", "off_selected_dynamic_risk",
                                "off_selected_offered_load_pressure", "off_selected_offered_load_risk",
                                "off_selected_compute_relative_weakness",
                                "off_selected_runtime_relative_weakness",
                                "off_selected_relative_weakness", "off_selected_weak_replica_risk",
                                "off_selected_weak_pressure",
                                "off_selected_runtime_confidence",
                                "unique_targets", "feasible_targets_mean", "feasible_targets_min",
                                "feasible_targets_max", "transition_buffer",
                                "offloading_plan", *self._state_record_fieldnames()]
        if self.record_cfg.actor_snapshot_debug:
            off_train_fieldnames.append("state_offloading_actor_snapshot")
        off_train_fieldnames.extend([
            "feedback_recorded", "feedback_deployment_version",
            "feedback_task_observations", "feedback_deployment_versions",
        ])
        self.off_recorder = Recorder(
            self._stage_log_path("offloading_train.csv"),
            fmt="csv",
            fieldnames=off_train_fieldnames,
            overwrite=True,
            flush_every=10,
        )
        self.off_decision_recorder = Recorder(
            self._stage_log_path("offloading_decisions.csv"),
            fmt="csv",
            fieldnames=self._offloading_decision_fieldnames(),
            overwrite=True,
            flush_every=50,
        )

        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)
        rollout_agent = self._current_offloading_rollout_agent()

        step = 0
        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _, state_debug = self._collect_offloading_state()
        last_reward_task_version = (
            self.state_buffer.get_task_observation_version()
            if self.state_buffer is not None else 0
        )
        while not self.offloading_thread_stop_event.is_set():
            try:
                if self._sleep_while_latency_guard_active(f"{log_scope} worker"):
                    logic_feats, phys_feats, _, _, state_debug = self._collect_offloading_state()
                    last_reward_task_version = (
                        self.state_buffer.get_task_observation_version()
                        if self.state_buffer is not None else last_reward_task_version
                    )
                    continue

                state_logic_feats = logic_feats
                state_phys_feats = phys_feats
                state_debug_record = state_debug
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
                        deterministic=self.training_cfg.offloading_rollout_deterministic,
                        enable_exploration=self.stage_cfg.update_offloading_policy,
                    )
                    offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                    self.offloading_plan = offloading_plan

                offloading_time_ticket = self._sleep_until_next_tick(
                    offloading_time_ticket,
                    self.offloading_interval,
                )
                if self.is_latency_guard_active():
                    logic_feats, phys_feats, _, _, state_debug = self._collect_offloading_state()
                    last_reward_task_version = (
                        self.state_buffer.get_task_observation_version()
                        if self.state_buffer is not None else last_reward_task_version
                    )
                    continue

                task_feedback_summary = (
                    self.state_buffer.get_task_observation_deployment_summary(last_reward_task_version)
                    if self.state_buffer is not None
                    else {
                        "current_version": last_reward_task_version,
                        "count": 0,
                        "deployment_version_counts": {},
                        "dominant_deployment_version": None,
                        "unique_deployment_versions": 0,
                        "all_same_deployment_version": False,
                    }
                )
                current_task_version = int(task_feedback_summary["current_version"])
                feedback_deployment_version = None
                if (
                        task_feedback_summary.get("all_same_deployment_version")
                        and task_feedback_summary.get("count", 0) > 0
                ):
                    feedback_deployment_version = task_feedback_summary.get("dominant_deployment_version")
                elif task_feedback_summary.get("count", 0) > 0:
                    LOGGER.debug(
                        f"[Hedger][Train][{log_scope}] Fresh task observations span multiple deployment versions; "
                        f"skip deployment feedback recording: versions="
                        f"{task_feedback_summary.get('deployment_version_counts', {})}"
                    )
                new_logic_feats, new_phys_feats, metrics, done, new_state_debug = self._collect_offloading_state(
                    since_task_version=last_reward_task_version,
                    deployment_version=feedback_deployment_version,
                )
                if current_task_version <= last_reward_task_version:
                    logic_feats = new_logic_feats
                    phys_feats = new_phys_feats
                    state_debug = new_state_debug
                    LOGGER.debug(
                        f"[Hedger][Train][{log_scope}] No fresh task observation; "
                        f"skip reward and transition."
                    )
                    continue
                last_reward_task_version = current_task_version

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

                off_reward_breakdown = self._compute_offloading_reward_breakdown(metrics, aux=aux)
                latency_cost = off_reward_breakdown["latency_cost"]
                reward = off_reward_breakdown["reward"]
                if self.is_latency_guard_active():
                    logic_feats = new_logic_feats
                    phys_feats = new_phys_feats
                    state_debug = new_state_debug
                    continue
                feedback_recorded = False
                if self.state_buffer is not None and feedback_deployment_version is not None:
                    feedback_recorded = self.state_buffer.add_offloading_reward(
                        reward,
                        task_version=current_task_version,
                        deployment_version=feedback_deployment_version,
                    )

                if self.stage_cfg.update_offloading_policy:
                    tr = {
                        "logic_edge_index": logic_edge_index.cpu(),
                        "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                        "phys_edge_index": phys_edge_index.cpu(),
                        "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                        "proposal_actions": aux["proposal_actions"].detach().cpu(),
                        "static_mask": static_mask_dev.cpu(),
                        "topo_order": None,
                        "exploration_enabled": bool(self.stage_cfg.update_offloading_policy),
                        "rollout_deterministic": bool(self.training_cfg.offloading_rollout_deterministic),
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

                state_record = self._state_record_metrics(state_logic_feats, state_phys_feats, state_debug_record)
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                state_debug = new_state_debug

                cloud_idx = self.physical_topology.cloud_idx
                actions_cpu = actions.detach().cpu()
                proposal_actions_cpu = aux["proposal_actions"].detach().cpu()
                proposal_cloud_fraction = (
                    float((proposal_actions_cpu == cloud_idx).float().mean().item())
                    if proposal_actions_cpu.numel() else 0.0
                )
                projected_cloud_fraction = (
                    float((actions_cpu == cloud_idx).float().mean().item())
                    if actions_cpu.numel() else 0.0
                )
                unique_targets = int(actions_cpu.unique().numel())
                feasible_counts = static_mask.detach().cpu().float().sum(dim=1)
                feasible_targets_mean = float(feasible_counts.mean().item()) if feasible_counts.numel() else 0.0
                feasible_targets_min = float(feasible_counts.min().item()) if feasible_counts.numel() else 0.0
                feasible_targets_max = float(feasible_counts.max().item()) if feasible_counts.numel() else 0.0

                row = dict(
                    step=step,
                    epoch=self._epoch,
                    off_updates=self._offloading_update_steps,
                    off_reward=reward,
                    latency=metrics["latency"],
                    latency_cost=latency_cost,
                    off_latency_term=off_reward_breakdown["off_latency_term"],
                    off_latency_normalizer=self.offloading_agent_params["reward_off_latency_normalizer"],
                    off_latency_clip=self.offloading_agent_params["reward_off_latency_clip"],
                    off_latency_transform=self.offloading_agent_params["reward_off_latency_transform"],
                    slo_violation=metrics["slo_violation"],
                    off_slo_term=off_reward_breakdown["off_slo_term"],
                    cloud_fraction=metrics["cloud_fraction"],
                    off_cloud_term=off_reward_breakdown["off_cloud_term"],
                    off_projection_term=off_reward_breakdown["off_projection_term"],
                    off_queue_term=off_reward_breakdown["off_queue_term"],
                    off_queue_cost=off_reward_breakdown["off_queue_cost"],
                    off_queue_risk_cost=off_reward_breakdown["off_queue_risk_cost"],
                    off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                    off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                    off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                    off_projection_weight=self.offloading_agent_params["reward_off_projection_weight"],
                    off_queue_weight=self.offloading_agent_params["reward_off_queue_weight"],
                    off_queue_clip=self.offloading_agent_params["reward_off_queue_clip"],
                    policy_logp=float(logp.detach().cpu().item()),
                    policy_entropy=float(ent.detach().cpu().item()),
                    value_estimate=float(value.detach().cpu().item()),
                    next_value=float(next_value),
                    offloading_rollout_deterministic=int(
                        bool(self.training_cfg.offloading_rollout_deterministic)
                    ),
                    proposal_cloud_fraction=proposal_cloud_fraction,
                    projected_cloud_fraction=projected_cloud_fraction,
                    offloading_projection_cnt=aux.get("projection_cnt", 0),
                    offloading_dependency_projection_cnt=aux.get("dependency_projection_cnt", 0),
                    offloading_infeasible_projection_cnt=aux.get("infeasible_projection_cnt", 0),
                    offloading_projection_cost=aux.get("projection_cost", 0.0),
                    off_selected_runtime_ratio=aux.get("selected_runtime_ratio", 0.0),
                    off_selected_runtime_recency=aux.get("selected_runtime_recency", 0.0),
                    off_selected_queue_freshness=aux.get("selected_queue_freshness", 0.0),
                    off_selected_speed_evidence=aux.get("selected_speed_evidence", 0.0),
                    off_selected_capacity_pressure=aux.get("selected_capacity_pressure", 0.0),
                    off_selected_pair_load=aux.get("selected_pair_load", 0.0),
                    off_selected_device_load=aux.get("selected_device_load", 0.0),
                    off_selected_load_pressure=aux.get("selected_load_pressure", 0.0),
                    off_selected_service_time_factor=aux.get("selected_service_time_factor", 0.0),
                    off_selected_base_queue_risk=aux.get("selected_base_queue_risk", 0.0),
                    off_selected_relative_queue_risk=aux.get("selected_relative_queue_risk", 0.0),
                    off_selected_overload_risk=aux.get("selected_overload_risk", 0.0),
                    off_selected_queue_risk_total=aux.get("selected_queue_risk_total", 0.0),
                    off_selected_planned_load_risk=aux.get("selected_planned_load_risk", 0.0),
                    off_selected_relative_planned_load_risk=aux.get("selected_relative_planned_load_risk", 0.0),
                    off_selected_dynamic_risk=aux.get("selected_dynamic_risk", 0.0),
                    off_selected_offered_load_pressure=aux.get("selected_offered_load_pressure", 0.0),
                    off_selected_offered_load_risk=aux.get("selected_offered_load_risk", 0.0),
                    off_selected_compute_relative_weakness=aux.get("selected_compute_relative_weakness", 0.0),
                    off_selected_runtime_relative_weakness=aux.get("selected_runtime_relative_weakness", 0.0),
                    off_selected_relative_weakness=aux.get("selected_relative_weakness", 0.0),
                    off_selected_weak_pressure=aux.get("selected_weak_pressure", 0.0),
                    off_selected_weak_replica_risk=aux.get("selected_weak_replica_risk", 0.0),
                    off_selected_runtime_confidence=aux.get("selected_runtime_confidence", 0.0),
                    unique_targets=unique_targets,
                    feasible_targets_mean=feasible_targets_mean,
                    feasible_targets_min=feasible_targets_min,
                    feasible_targets_max=feasible_targets_max,
                    transition_buffer=transition_count,
                    offloading_plan=self._json_for_record(self.offloading_plan),
                    **state_record,
                    feedback_recorded=feedback_recorded,
                    feedback_deployment_version=feedback_deployment_version,
                    feedback_task_observations=task_feedback_summary.get("count", 0),
                    feedback_deployment_versions=self._json_for_record(
                        {
                            str(key): value
                            for key, value in task_feedback_summary.get("deployment_version_counts", {}).items()
                        }
                    ),
                )
                if self.record_cfg.actor_snapshot_debug:
                    row["state_offloading_actor_snapshot"] = self._json_for_record(aux.get("actor_debug"))
                self.off_recorder.log_dict(row)
                self._log_offloading_decisions(
                    step=step,
                    actions=actions_cpu,
                    static_mask=static_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                    proposal_actions=proposal_actions_cpu,
                    projection_reasons=aux.get("projection_reasons"),
                    actor_debug=aux.get("actor_debug"),
                    policy_deterministic=self.training_cfg.offloading_rollout_deterministic,
                )
                LOGGER.debug(
                    f"[Hedger][Train][{log_scope}] step={step}, reward={self._format_log_value(reward)}, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"projection_cnt={aux.get('projection_cnt', 0)}, "
                    f"dependency_projection_cnt={aux.get('dependency_projection_cnt', 0)}, "
                    f"infeasible_projection_cnt={aux.get('infeasible_projection_cnt', 0)}, "
                    f"feedback_recorded={feedback_recorded}, "
                    f"update_policy={self.stage_cfg.update_offloading_policy}, transitions={transition_count}"
                )

                step += 1
            except Exception as e:
                LOGGER.exception(f"[Hedger][Train][{log_scope}] Worker loop error: {e}")
                time.sleep(0.5)

        self.off_recorder.close()
        self.off_decision_recorder.close()
        LOGGER.info(f"[Hedger][Train][{log_scope}] Worker stopped.")

    @staticmethod
    def _compute_latency_cost(
            latency: float,
            *,
            transform: str,
            normalizer: float,
            clip_value: Optional[float],
    ) -> float:
        """Convert measured latency into a bounded cost shared by both agents."""
        latency = max(0.0, float(latency))
        ratio = latency / max(normalizer, 1e-6)

        if clip_value is not None:
            ratio = min(ratio, float(clip_value))

        if transform == "raw":
            cost = latency
            if clip_value is not None:
                cost = min(cost, float(clip_value))
            return float(cost)
        if transform == "log_ratio":
            return float(math.log1p(ratio))
        return float(ratio)

    def _compute_offloading_latency_cost(self, latency: float) -> float:
        return self._compute_latency_cost(
            latency,
            transform=self.offloading_agent_params["reward_off_latency_transform"],
            normalizer=float(self.offloading_agent_params["reward_off_latency_normalizer"]),
            clip_value=self.offloading_agent_params["reward_off_latency_clip"],
        )

    def _compute_deployment_latency_cost(self, latency: float) -> float:
        return self._compute_latency_cost(
            latency,
            transform=self.deployment_agent_params["reward_dep_latency_transform"],
            normalizer=float(self.deployment_agent_params["reward_dep_latency_normalizer"]),
            clip_value=self.deployment_agent_params["reward_dep_latency_clip"],
        )

    @staticmethod
    def _sum_reward_terms(terms: Dict[str, float]) -> float:
        return float(sum(float(value) for value in terms.values()))

    def _compute_offloading_reward_breakdown(self, metrics, aux=None) -> Dict[str, float]:
        """Named reward terms for offloading, reused by logging and PPO."""
        metrics = metrics or {}
        aux = aux or {}
        latency = float(metrics["latency"])
        latency_cost = self._compute_offloading_latency_cost(latency)
        slo_v = float(metrics["slo_violation"])
        cloud_frac = float(metrics["cloud_fraction"])
        projection_cost = float(aux.get("projection_cost", 0.0) or 0.0)
        queue_risk_cost = max(0.0, float(aux.get("selected_queue_risk_cost", 0.0) or 0.0))
        queue_clip = self.offloading_agent_params.get("reward_off_queue_clip")
        queue_cost = queue_risk_cost
        if queue_clip is not None:
            queue_cost = min(queue_cost, max(0.0, float(queue_clip)))

        w_lat = float(self.offloading_agent_params["reward_off_latency_weight"])
        w_slo = float(self.offloading_agent_params["reward_off_slo_weight"])
        w_cloud = float(self.offloading_agent_params["reward_off_cloud_weight"])
        w_projection = float(self.offloading_agent_params.get("reward_off_projection_weight", 0.0))
        w_queue = float(self.offloading_agent_params.get("reward_off_queue_weight", 0.0))

        terms = {
            "off_latency_term": -w_lat * latency_cost,
            "off_slo_term": -w_slo * slo_v,
            "off_cloud_term": -w_cloud * cloud_frac,
            "off_projection_term": -w_projection * projection_cost,
            "off_queue_term": -w_queue * queue_cost,
        }
        return {
            "latency_cost": float(latency_cost),
            "off_projection_cost": float(projection_cost),
            "off_queue_cost": float(queue_cost),
            "off_queue_risk_cost": float(queue_risk_cost),
            **terms,
            "reward": self._sum_reward_terms(terms),
        }

    def _compute_offloading_reward(self, metrics, aux=None) -> float:
        return self._compute_offloading_reward_breakdown(metrics, aux=aux)["reward"]

    def _compute_deployment_reward_breakdown(self, metrics, aux) -> Dict[str, float]:
        """Named deployment reward terms mixing direct e2e and bottom-up feedback."""
        metrics = metrics or {}

        avg_off_r = float(metrics["avg_offloading_reward"])
        latency_count = int(metrics.get("e2e_latency_count", 0) or 0)
        e2e_latency_mean = float(metrics.get("e2e_latency_mean", 0.0) or 0.0)
        e2e_slo_violation = float(metrics.get("e2e_slo_violation", 0.0) or 0.0)
        deploy_change_cost = float(metrics["deploy_change_cost"])
        cloud_only_ratio = float(aux.get("cloud_only_ratio", 0.0))
        cap_relax_cost = float(aux.get("capacity_relax_cost", 0.0))
        edge_cover_repair_cost = float(aux.get("edge_cover_repair_cost", 0.0))
        active_pair_hotspot_cost = float(aux.get("executed_active_pair_hotspot_cost", aux.get("active_pair_hotspot_cost", 0.0)))
        selected_runtime_risk_cost = float(aux.get("selected_runtime_risk_cost", 0.0))
        selected_unknown_option_cost = float(aux.get("selected_unknown_option_cost", 0.0))
        selected_stale_option_cost = float(aux.get("selected_stale_option_cost", 0.0))
        selected_low_quality_option_cost = float(aux.get("selected_low_quality_option_cost", 0.0))
        latency_guard_penalty_cost = float(metrics.get("latency_guard_penalty_cost", 0.0))
        feedback_timeout_penalty_cost = float(metrics.get("feedback_timeout_penalty_cost", 0.0))

        w_off = float(self.deployment_agent_params["reward_dep_offload_weight"])
        w_lat = float(self.deployment_agent_params.get("reward_dep_latency_weight", 0.0))
        w_slo = float(self.deployment_agent_params.get("reward_dep_slo_weight", 0.0))
        w_change = float(self.deployment_agent_params["reward_dep_change_weight"])
        w_cloud_only = float(self.deployment_agent_params.get("reward_dep_cloud_only_weight", 0.0))
        w_runtime_risk = float(self.deployment_agent_params.get("reward_dep_runtime_risk_weight", 0.0))
        w_unknown_option = float(self.deployment_agent_params.get("reward_dep_unknown_option_weight", 0.0))
        w_stale_option = float(self.deployment_agent_params.get("reward_dep_stale_option_weight", 0.0))
        w_low_quality = float(self.deployment_agent_params.get("reward_dep_low_quality_weight", 0.0))
        penalty_capacity_relax = float(self.deployment_agent_params["penalty_capacity_relax"])
        penalty_edge_cover_repair = float(self.deployment_agent_params.get("penalty_edge_cover_repair", 0.0))
        w_hotspot = float(self.deployment_agent_params.get("reward_dep_hotspot_weight", 0.0))
        penalty_latency_guard = float(self.deployment_agent_params.get("penalty_latency_guard_trigger", 0.0))
        penalty_feedback_timeout = float(self.deployment_agent_params.get("penalty_feedback_timeout", 0.0))

        latency_cost = 0.0
        if latency_count > 0:
            latency_cost = self._compute_deployment_latency_cost(e2e_latency_mean)

        terms = {
            "dep_offload_term": w_off * avg_off_r,
            "dep_latency_term": -w_lat * latency_cost,
            "dep_slo_term": -w_slo * e2e_slo_violation,
            "dep_change_term": -w_change * deploy_change_cost,
            "dep_cloud_only_term": -w_cloud_only * cloud_only_ratio,
            "dep_capacity_relax_term": -penalty_capacity_relax * cap_relax_cost,
            "dep_edge_cover_repair_term": -penalty_edge_cover_repair * edge_cover_repair_cost,
            "dep_hotspot_term": -w_hotspot * active_pair_hotspot_cost,
            "dep_runtime_risk_term": -w_runtime_risk * selected_runtime_risk_cost,
            "dep_unknown_option_term": -w_unknown_option * selected_unknown_option_cost,
            "dep_stale_option_term": -w_stale_option * selected_stale_option_cost,
            "dep_low_quality_term": -w_low_quality * selected_low_quality_option_cost,
            "dep_latency_guard_penalty_term": -penalty_latency_guard * latency_guard_penalty_cost,
            "dep_feedback_timeout_term": -penalty_feedback_timeout * feedback_timeout_penalty_cost,
        }
        return {
            "dep_latency_cost": float(latency_cost),
            "feedback_timeout_penalty_cost": float(feedback_timeout_penalty_cost),
            "active_pair_hotspot_cost": float(aux.get("active_pair_hotspot_cost", 0.0)),
            "executed_active_pair_hotspot_cost": float(aux.get("executed_active_pair_hotspot_cost", 0.0)),
            "selected_runtime_risk_cost": selected_runtime_risk_cost,
            "selected_unknown_option_cost": selected_unknown_option_cost,
            "selected_stale_option_cost": selected_stale_option_cost,
            "selected_low_quality_option_cost": selected_low_quality_option_cost,
            **terms,
            "reward": self._sum_reward_terms(terms),
        }

    def _compute_deployment_reward(self, metrics, aux) -> float:
        return self._compute_deployment_reward_breakdown(metrics, aux)["reward"]

    def _deployment_offline_update_kwargs(self) -> Dict[str, Any]:
        cfg = self.training_cfg.deployment_offline_rl
        return {
            "action_target": cfg.action_target,
            "advantage_temperature": cfg.advantage_temperature,
            "min_advantage_weight": cfg.min_advantage_weight,
            "max_advantage_weight": cfg.max_advantage_weight,
            "actor_bc_coef": cfg.actor_bc_coef,
            "negative_bc_coef": cfg.negative_bc_coef,
            "raw_removed_negative_coef": cfg.raw_removed_negative_coef,
            "value_coef": cfg.value_coef,
            "entropy_coef": cfg.entropy_coef,
            "bootstrap_current_value": cfg.bootstrap_current_value,
        }

    def _sample_deployment_online_replay_batch(self) -> List[dict]:
        cfg = self.training_cfg.deployment_offline_rl
        total_batch = max(1, int(cfg.batch_size))
        with self._data_lock:
            online_pool = list(self.deployment_transitions)

        offline_count = 0
        if self.deployment_offline_dataset is not None and len(self.deployment_offline_dataset) > 0:
            offline_count = int(round(total_batch * cfg.offline_replay_ratio))
        offline_count = min(total_batch, max(0, offline_count))
        online_count = max(0, total_batch - offline_count)

        batch: List[dict] = []
        if offline_count > 0 and self.deployment_offline_dataset is not None:
            batch.extend(self.deployment_offline_dataset.sample(offline_count))
        if online_pool and online_count > 0:
            if len(online_pool) >= online_count:
                batch.extend(random.sample(online_pool, online_count))
            else:
                batch.extend(random.choice(online_pool) for _ in range(online_count))
        if not batch and online_pool:
            batch.extend(random.sample(online_pool, min(len(online_pool), total_batch)))
        return batch

    def train_deployment_offline(self):
        dataset = self.deployment_offline_dataset
        if dataset is None:
            dataset = DeploymentTransitionDataset(self.training_cfg.deployment_dataset.root_dir)
            self.deployment_offline_dataset = dataset
        if len(dataset) <= 0:
            raise RuntimeError(
                "[Hedger][Train][DeploymentOffline] Empty deployment dataset. "
                f"Run stage=deployment_collect first or check root={self.training_cfg.deployment_dataset.root_dir}."
            )

        LOGGER.info(
            f"[Hedger][Train][DeploymentOffline] Start: samples={len(dataset)}, "
            f"updates={self.training_cfg.total_updates}, batch={self.training_cfg.deployment_offline_rl.batch_size}, "
            f"dataset={self.training_cfg.deployment_dataset.root_dir}"
        )
        self.dep_update_recorder = Recorder(
            self._stage_log_path("deployment_offline_updates.csv"),
            fmt="csv",
            fieldnames=self._ppo_update_fieldnames(include_offline_batch=True),
            overwrite=True,
            flush_every=1,
        )
        try:
            while self._epoch < self.training_cfg.total_updates:
                batch = dataset.sample(self.training_cfg.deployment_offline_rl.batch_size)
                batch_quality = summarize_transition_quality(batch)
                with self._model_lock:
                    stats = self.deployment_agent.offline_update(
                        batch,
                        batch_size=len(batch),
                        **self._deployment_offline_update_kwargs(),
                    )
                if stats is not None:
                    stats.update(batch_quality)
                self._deployment_update_steps += 1
                self._epoch += 1
                self._global_update_step += 1
                self._record_ppo_update(
                    self.dep_update_recorder,
                    "deployment_offline",
                    self._deployment_update_steps,
                    used=len(batch),
                    remaining=len(dataset),
                    stats=stats,
                )
                LOGGER.info(
                    f"[Hedger][Train][DeploymentOffline] update={self._deployment_update_steps}, "
                    f"reward_mean={self._format_log_value(stats.get('reward_mean', 0.0))}, "
                    f"batch_bad_ratio={self._format_log_value(stats.get('offline_batch_bad_ratio', 0.0))}, "
                    f"policy_loss={self._format_log_value(stats.get('policy_loss', 0.0))}, "
                    f"value_loss={self._format_log_value(stats.get('value_loss', 0.0))}, "
                    f"adv_mean={self._format_log_value(stats.get('adv_mean', 0.0))}"
                )
                save_interval = self.checkpoint_cfg.save.interval_updates
                if self._epoch % save_interval == 0:
                    self.save_checkpoint(stage_step=self._epoch, is_final=False)
        finally:
            if self.dep_update_recorder is not None:
                self.dep_update_recorder.close()
                self.dep_update_recorder = None
            self.save_checkpoint(stage_step=self._epoch, is_final=True)
            LOGGER.info(
                f"[Hedger][Train][DeploymentOffline] Finished: "
                f"stage_step={self._epoch}, global_step={self._global_update_step}"
            )

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
                'deployment_train_mode': (
                    self.stage_cfg.deployment_train_mode
                    if self.stage_cfg is not None else None
                ),
                'deployment_dataset_root': (
                    self.training_cfg.deployment_dataset.root_dir
                    if self.training_cfg is not None else None
                ),
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
                    self._load_encoder_state(enc_state)
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

    def _load_encoder_state(self, state_dict: dict) -> None:
        self.shared_topology_encoder.load_state_dict(state_dict)

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
        return self.physical_topology and self.logical_topology and self.state_buffer

    def run(self):
        wait_logged = False
        while not self._ready_for_run:
            if not wait_logged:
                LOGGER.debug(
                    f"[Hedger][Lifecycle] Waiting for topology registration: "
                    f"logical_ready={self.logical_topology is not None}, "
                    f"physical_ready={self.physical_topology is not None}, "
                    f"state_buffer_ready={self.state_buffer is not None}"
                )
                wait_logged = True
            time.sleep(0.5)

        if self.mode == 'train':
            self.train_hedger()
        elif self.mode == 'inference':
            self.inference_hedger()
        else:
            raise ValueError(f'Unsupported mode {self.mode} for Hedger, only "train" and "inference" are supported.')
