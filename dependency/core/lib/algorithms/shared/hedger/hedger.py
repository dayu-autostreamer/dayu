import copy
from collections import deque
from typing import Any, Dict, List, Optional
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
    deployment_reward_min_samples: int
    deployment_feedback_timeout_s: Optional[float]


@dataclass(frozen=True)
class HedgerEncoderCfg:
    embedding_dim: int
    logical_heads: int
    physical_role_count: int
    physical_role_embedding_dim: int
    dropout: float


@dataclass(frozen=True)
class HedgerDeploymentDefaultWarmupCfg:
    enabled: bool = False
    min_intervals: int = 0
    min_feedback_samples: int = 0
    timeout_s: Optional[float] = None
    clear_feedback_window: bool = True


@dataclass(frozen=True)
class HedgerTrainingCfg:
    stage: str
    total_updates: int
    ppo_epochs: int
    deployment_rollout_len: int
    offloading_rollout_len: int
    deployment_batch_size: int
    offloading_batch_size: int
    deployment_default_warmup: HedgerDeploymentDefaultWarmupCfg = field(
        default_factory=HedgerDeploymentDefaultWarmupCfg
    )


@dataclass(frozen=True)
class HedgerInferenceCfg:
    run_deployment_worker: bool = True
    run_offloading_worker: bool = True
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
    state_snapshot_debug: bool = False
    actor_snapshot_debug: bool = False
    decision_pair_features_debug: bool = False
    decision_actor_debug: bool = False


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
        self.checkpoint_cfg = self._build_checkpoint_cfg(config)

        agents_cfg = self._require_mapping(config, "agents")
        self.deployment_agent_params = self._build_deployment_agent_params(agents_cfg)
        self.offloading_agent_params = self._build_offloading_agent_params(agents_cfg)

        self.deployment_thread_stop_event = threading.Event()
        self.offloading_thread_stop_event = threading.Event()
        self._latency_guard_trigger_event = threading.Event()

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
            physical_role_count=max(2, int(encoder["physical_role_count"])),
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

        return HedgerTrainingCfg(
            stage=stage,
            total_updates=max(0, int(training["total_updates"])),
            ppo_epochs=max(1, int(training["ppo_epochs"])),
            deployment_rollout_len=max(1, int(rollout["deployment"])),
            offloading_rollout_len=max(1, int(rollout["offloading"])),
            deployment_batch_size=max(1, int(batch_size["deployment"])),
            offloading_batch_size=max(1, int(batch_size["offloading"])),
            deployment_default_warmup=deployment_default_warmup,
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
        return HedgerRecordCfg(
            state_summary=bool(record.get("state_summary", True)),
            state_snapshot_debug=bool(record.get("state_snapshot_debug", False)),
            actor_snapshot_debug=bool(record.get("actor_snapshot_debug", False)),
            decision_pair_features_debug=bool(record.get("decision_pair_features_debug", False)),
            decision_actor_debug=bool(record.get("decision_actor_debug", False)),
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
    def _ppo_update_fieldnames() -> List[str]:
        return [
            "agent", "update", "epoch", "used", "remaining",
            "samples", "epochs", "batch_size", "minibatches",
            "reward_mean", "reward_std", "reward_min", "reward_max",
            "value_old_mean", "value_old_std", "value_new_mean",
            "return_mean", "return_std", "adv_mean", "adv_std",
            "last_value", "done_fraction",
            "policy_loss", "value_loss", "entropy", "entropy_coef", "value_coef", "approx_kl",
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
        constraints = deployment.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ValueError("Hedger config `agents.deployment.constraints` must be a mapping when provided.")
        max_edge_replicas = constraints.get("max_edge_replicas_per_device")
        if max_edge_replicas is not None:
            max_edge_replicas = int(max_edge_replicas)
            if max_edge_replicas <= 0:
                max_edge_replicas = None
        edge_memory_budget_ratio = float(constraints.get("edge_memory_budget_ratio", 1.0))
        if not math.isfinite(edge_memory_budget_ratio) or not (0.0 < edge_memory_budget_ratio <= 1.0):
            raise ValueError("Hedger config `agents.deployment.constraints.edge_memory_budget_ratio` must be in (0, 1].")
        min_edge_replicas = int(constraints.get("min_edge_replicas_per_service", 0) or 0)
        if min_edge_replicas < 0:
            raise ValueError(
                "Hedger config `agents.deployment.constraints.min_edge_replicas_per_service` must be >= 0."
            )
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
            "reward_dep_latency_transform": dep_latency_cfg["transform"],
            "reward_dep_latency_normalizer": dep_latency_cfg["normalizer"],
            "reward_dep_latency_clip": dep_latency_cfg["clip"],
            "penalty_capacity_relax": float(penalty["capacity_relax"]),
            "penalty_edge_cover_repair": float(penalty.get("edge_cover_repair", 0.0)),
            "penalty_latency_guard_trigger": float(penalty.get("latency_guard_trigger", 0.0)),
            "max_edge_replicas_per_device": max_edge_replicas,
            "edge_memory_budget_ratio": edge_memory_budget_ratio,
            "min_edge_replicas_per_service": min_edge_replicas,
            "ppo": ppo,
        }

    def _build_offloading_agent_params(self, agents_cfg: dict) -> dict:
        offloading = self._require_mapping(agents_cfg, "offloading")
        reward = self._require_mapping(offloading, "reward")
        penalty = self._require_mapping(offloading, "penalty")
        ppo = self._build_ppo_update_cfg(offloading)
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
            "reward_off_latency_weight": float(reward["latency_weight"]),
            "reward_off_slo_weight": float(reward["slo_weight"]),
            "reward_off_cloud_weight": float(reward["cloud_weight"]),
            "reward_off_latency_transform": off_latency_cfg["transform"],
            "reward_off_latency_normalizer": off_latency_cfg["normalizer"],
            "reward_off_latency_clip": off_latency_cfg["clip"],
            "penalty_relax": float(penalty["correction"]),
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
        if not self._latency_guard_enabled() or not self.latency_guard_cfg.queue_recovery_enabled:
            return self._empty_latency_guard_stats()
        if not isinstance(queue_lengths, dict):
            return self.latency_guard_status()

        normalized = {}
        for service_name, value in queue_lengths.items():
            try:
                normalized[str(service_name)] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue
        if not normalized:
            return self.latency_guard_status()

        recovered = False
        now = time.monotonic()
        with self._latency_guard_lock:
            self._latency_guard_queue_observations[str(device)] = {
                "values": normalized,
                "ts": now,
            }
            if self._latency_guard_active:
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
            self._latency_guard_last_stats = self._compute_latency_guard_stats_locked()
            stats = copy.deepcopy(self._latency_guard_last_stats)

        if recovered:
            self._handle_latency_guard_recovered(
                stats,
                task_version=None,
                deployment_version=None,
                reason="queue_drain",
            )
        else:
            self._log_latency_guard_if_active(stats)
        return stats

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
        stage_name = training_cfg.stage if training_cfg is not None else "none"
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
            f"max_edge_replicas_per_device={dep_params.get('max_edge_replicas_per_device', 'na')}, "
            f"edge_memory_budget_ratio={dep_params.get('edge_memory_budget_ratio', 'na')}, "
            f"min_edge_replicas_per_service={dep_params.get('min_edge_replicas_per_service', 'na')}, "
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
            f"latest_edge_bw={self._format_log_value(latest_edge_bw)}, "
            f"latest_cloud_bw={self._format_log_value(latest_cloud_bw)}, "
            f"latest_gpu_util={self._format_utilization_for_log(latest_gpu_util)}, "
            f"latest_mem_util={self._format_utilization_for_log(latest_mem_util)}"
        )
        if metrics:
            base += f", metrics=({self._summarize_metrics(metrics)})"
        return base

    @staticmethod
    def _json_for_record(value) -> str:
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
        return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

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
    ) -> Dict[str, List[float]]:
        value = logic_feats.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return {}
        value = value.detach().float().cpu()
        if value.dim() != 3 or service_idx < 0 or service_idx >= value.size(0):
            return {}
        return {
            self._device_name(device_idx): [float(item) for item in value[service_idx, device_idx].tolist()]
            for device_idx in range(value.size(1))
        }

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
        pair_features = state_debug.get("pair_feature_snapshot", {}) or {}
        latency_pair_snapshot = state_debug.get("latency_pair_snapshot", {}) or {}
        queue_pair_snapshot = state_debug.get("queue_pair_snapshot", {}) or {}
        row: Dict[str, Any] = {}
        if self.record_cfg.state_summary:
            service_count = int(logic_feats["model_flops"].numel()) if "model_flops" in logic_feats else 0
            device_count = int(phys_feats["gpu_flops"].numel()) if "gpu_flops" in phys_feats else 0

            latest_complexity = 0.0
            complexity_seq = logic_feats.get("task_complexity_seq")
            if isinstance(complexity_seq, torch.Tensor) and complexity_seq.numel():
                latest_complexity = float(complexity_seq[:, -1].float().mean().item())

            latest_gpu_util = 0.0
            latest_gpu_util_max = 0.0
            gpu_util_seq = phys_feats.get("gpu_util_seq")
            if isinstance(gpu_util_seq, torch.Tensor) and gpu_util_seq.numel():
                latest_gpu_util = float(gpu_util_seq[:, -1].float().mean().item())
                latest_gpu_util_max = float(gpu_util_seq[:, -1].float().max().item())

            latest_mem_util = 0.0
            latest_mem_util_max = 0.0
            mem_util_seq = phys_feats.get("mem_util_seq")
            if isinstance(mem_util_seq, torch.Tensor) and mem_util_seq.numel():
                latest_mem_util = float(mem_util_seq[:, -1].float().mean().item())
                latest_mem_util_max = float(mem_util_seq[:, -1].float().max().item())

            latest_cloud_bw = 0.0
            latest_edge_bw = 0.0
            bandwidth_seq = phys_feats.get("bandwidth_seq")
            if isinstance(bandwidth_seq, torch.Tensor) and bandwidth_seq.numel():
                cloud_idx = (
                    self.physical_topology.cloud_idx
                    if self.physical_topology is not None
                    else bandwidth_seq.size(0) - 1
                )
                latest_cloud_bw = float(bandwidth_seq[cloud_idx, -1].float().item())
                edge_bandwidth = bandwidth_seq[:, -1]
                if edge_bandwidth.numel() > 1:
                    edge_bandwidth = torch.cat([edge_bandwidth[:cloud_idx], edge_bandwidth[cloud_idx + 1:]])
                else:
                    edge_bandwidth = torch.empty(0, dtype=edge_bandwidth.dtype)
                latest_edge_bw = float(edge_bandwidth.float().mean().item()) if edge_bandwidth.numel() else 0.0

            row.update({
                "state_services": service_count,
                "state_devices": device_count,
                "state_complexity": latest_complexity,
                "state_edge_bw": latest_edge_bw,
                "state_cloud_bw": latest_cloud_bw,
                "state_gpu_util_mean": latest_gpu_util,
                "state_gpu_util_max": latest_gpu_util_max,
                "state_mem_util_mean": latest_mem_util,
                "state_mem_util_max": latest_mem_util_max,
                "state_model_flops_mean": self._tensor_mean(logic_feats.get("model_flops")),
                "state_model_mem_mean": self._tensor_mean(logic_feats.get("model_mem")),
                "state_gpu_flops_mean": self._tensor_mean(phys_feats.get("gpu_flops")),
                "state_mem_capacity_mean": self._tensor_mean(phys_feats.get("mem_capacity")),
                "state_latency_pair_obs_count": self._nested_float_mean(latency_pair_snapshot.get("pair_count")),
                "state_queue_pair_obs_count": self._nested_float_mean(queue_pair_snapshot.get("pair_count")),
                "state_latency_pair_off_reliability_mean": self._nested_float_mean(
                    [
                        row[-1] for service_rows in pair_features.get("offloading_latency", [])
                        for row in service_rows
                    ]
                ),
                "state_latency_pair_dep_reliability_mean": self._nested_float_mean(
                    [
                        row[-1] for service_rows in pair_features.get("deployment_latency", [])
                        for row in service_rows
                    ]
                ),
                "state_queue_pair_off_reliability_mean": self._nested_float_mean(
                    [
                        row[-1] for service_rows in pair_features.get("offloading_queue", [])
                        for row in service_rows
                    ]
                ),
                "state_queue_pair_dep_reliability_mean": self._nested_float_mean(
                    [
                        row[-1] for service_rows in pair_features.get("deployment_queue", [])
                        for row in service_rows
                    ]
                ),
            })

        if self.record_cfg.state_snapshot_debug:
            row.update({
                "state_logic_snapshot": self._json_for_record(state_debug.get("logic_snapshot")),
                "state_phys_snapshot": self._json_for_record(state_debug.get("phys_snapshot")),
                "state_latency_pair_snapshot": self._json_for_record(latency_pair_snapshot),
                "state_queue_pair_snapshot": self._json_for_record(queue_pair_snapshot),
                "state_pair_feature_snapshot": self._json_for_record(pair_features),
            })
        return row

    @staticmethod
    def _state_record_summary_fieldnames() -> List[str]:
        return [
            "state_services", "state_devices", "state_complexity",
            "state_edge_bw", "state_cloud_bw", "state_gpu_util_mean",
            "state_gpu_util_max", "state_mem_util_mean", "state_mem_util_max",
            "state_model_flops_mean", "state_model_mem_mean",
            "state_gpu_flops_mean", "state_mem_capacity_mean",
            "state_latency_pair_obs_count", "state_queue_pair_obs_count",
            "state_latency_pair_off_reliability_mean", "state_latency_pair_dep_reliability_mean",
            "state_queue_pair_off_reliability_mean", "state_queue_pair_dep_reliability_mean",
        ]

    @staticmethod
    def _state_record_debug_fieldnames() -> List[str]:
        return [
            "state_logic_snapshot", "state_phys_snapshot",
            "state_latency_pair_snapshot", "state_queue_pair_snapshot", "state_pair_feature_snapshot",
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
            "dep_latency_guard_penalty_term",
            "e2e_latency_count", "e2e_latency_mean", "e2e_latency_latest",
            "e2e_latency_p50", "e2e_latency_p90", "e2e_latency_p95", "e2e_latency_p99",
            "e2e_slo_violation", "feedback_gate_enabled", "feedback_gate_required_samples",
            "feedback_gate_collected_samples", "feedback_gate_timed_out", "feedback_gate_guard_truncated",
            "cap_relax_cnt", "cap_relax_cost", "edge_cover_repair_cnt",
            "edge_cover_repair_cost", "edge_cover_unmet", "policy_logp", "policy_entropy",
            "value_estimate", "raw_edge_replicas", "edge_replicas", "cloud_replicas",
            "cloud_only", "cloud_only_ratio", "empty_edge_devices", "empty_edge_device_ratio",
            "raw_deployment_plan", "deployment_plan", "active_deployment_plan",
            *self._state_record_fieldnames(),
            *Hedger._latency_guard_record_fieldnames(),
            "dep_offload_weight", "dep_latency_weight", "dep_latency_transform",
            "dep_latency_normalizer", "dep_latency_clip", "dep_slo_weight",
            "dep_change_weight", "dep_cloud_only_weight", "cap_relax_weight", "edge_cover_repair_weight",
            "latency_guard_penalty_weight", "max_edge_replicas_per_device",
            "edge_memory_budget_ratio", "min_edge_replicas_per_service", "loaded_checkpoint",
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
            "cloud_fraction", "task_latency_count", "latest_task_latency", "aux_cost",
            "off_cloud_term", "correction_cnt", "correction_cost", "off_correction_term", "policy_logp",
            "policy_entropy", "value_estimate", "raw_cloud_fraction",
            "executed_cloud_fraction", "unique_targets", "feasible_targets_mean",
            "feasible_targets_min", "feasible_targets_max", "raw_offloading_plan",
            "offloading_plan", "active_deployment_plan", *self._state_record_fieldnames(),
            *Hedger._latency_guard_record_fieldnames(),
            "feedback_task_observations", "feedback_deployment_version",
            "feedback_deployment_versions", "feedback_recorded", "off_latency_weight",
            "off_latency_transform", "off_latency_normalizer", "off_latency_clip",
            "off_slo_weight", "off_cloud_weight", "correction_weight", "loaded_checkpoint",
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
            "capacity_corrected",
        ]
        if self.record_cfg.decision_pair_features_debug:
            fieldnames.extend([
                "deployment_latency_pair_features",
                "deployment_queue_pair_features",
            ])
        if self.record_cfg.decision_actor_debug:
            fieldnames.extend([
                "deployment_qk_scores", "deployment_pair_bias", "deployment_final_scores",
                "deployment_policy_probs", "deployment_static_mask",
            ])
        return fieldnames

    def _offloading_decision_fieldnames(self) -> List[str]:
        fieldnames = [
            *Hedger._decision_common_fieldnames(),
            "raw_target", "executed_target", "corrected", "raw_is_cloud", "executed_is_cloud",
            "feasible_targets", "feasible_target_count", "parent_targets",
            "target_gpu_util", "target_mem_util", "target_bandwidth",
        ]
        if self.record_cfg.decision_pair_features_debug:
            fieldnames.extend([
                "offloading_latency_pair_features",
                "offloading_queue_pair_features",
            ])
        if self.record_cfg.decision_actor_debug:
            fieldnames.extend([
                "offloading_qk_scores", "offloading_pair_bias", "offloading_final_scores",
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

    def _log_deployment_decisions(
            self,
            *,
            step: int,
            raw_deploy_mask: torch.Tensor,
            exec_deploy_mask: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            actor_debug: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.dep_decision_recorder is None or self.physical_topology is None:
            return

        cloud_idx = self.physical_topology.cloud_idx
        raw_mask = raw_deploy_mask.detach().cpu().bool()
        exec_mask = exec_deploy_mask.detach().cpu().bool()
        for service_idx in range(raw_mask.size(0)):
            raw_nodes = self._device_names_from_mask(raw_mask[service_idx])
            executed_nodes = self._device_names_from_mask(exec_mask[service_idx])
            removed_indices = torch.nonzero(raw_mask[service_idx] & ~exec_mask[service_idx], as_tuple=False)
            added_indices = torch.nonzero(~raw_mask[service_idx] & exec_mask[service_idx], as_tuple=False)
            raw_edge_count = int(raw_mask[service_idx, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
            exec_edge_count = int(exec_mask[service_idx, :cloud_idx].sum().item()) if cloud_idx > 0 else 0
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
                capacity_corrected=bool(not torch.equal(raw_mask[service_idx], exec_mask[service_idx])),
            )
            if self.record_cfg.decision_pair_features_debug:
                row.update({
                    "deployment_latency_pair_features": self._json_for_record(
                        self._pair_feature_map(logic_feats, "deployment_latency_pair_feat", service_idx)
                    ),
                    "deployment_queue_pair_features": self._json_for_record(
                        self._pair_feature_map(logic_feats, "deployment_queue_pair_feat", service_idx)
                    ),
                })
            if self.record_cfg.decision_actor_debug:
                row.update({
                    "deployment_qk_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_score", service_idx)
                    ),
                    "deployment_pair_bias": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "pair_bias", service_idx)
                    ),
                    "deployment_final_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "final_score", service_idx)
                    ),
                    "deployment_policy_probs": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "policy_prob", service_idx)
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
            raw_actions: torch.Tensor,
            executed_actions: torch.Tensor,
            static_mask: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            actor_debug: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.off_decision_recorder is None or self.physical_topology is None:
            return

        raw_actions = raw_actions.detach().cpu().long()
        executed_actions = executed_actions.detach().cpu().long()
        static_mask = static_mask.detach().cpu().bool()
        parents = [[] for _ in range(raw_actions.numel())]
        if self.logical_topology is not None:
            for parent, child in self.logical_topology.links:
                parents[child].append(parent)

        cloud_idx = self.physical_topology.cloud_idx
        for service_idx in range(raw_actions.numel()):
            raw_target_idx = int(raw_actions[service_idx].item())
            executed_target_idx = int(executed_actions[service_idx].item())
            feasible_row = static_mask[service_idx].clone()
            if not feasible_row.any():
                feasible_row[cloud_idx] = True
            feasible_indices = torch.nonzero(feasible_row, as_tuple=False).flatten().tolist()
            parent_target_names = [
                self._device_name(int(executed_actions[parent].item()))
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
                raw_target=self._device_name(raw_target_idx),
                executed_target=self._device_name(executed_target_idx),
                corrected=bool(raw_target_idx != executed_target_idx),
                raw_is_cloud=bool(raw_target_idx == cloud_idx),
                executed_is_cloud=bool(executed_target_idx == cloud_idx),
                feasible_targets=self._json_for_record(self._device_names_from_indices(feasible_indices)),
                feasible_target_count=len(feasible_indices),
                parent_targets=self._json_for_record(parent_target_names),
                target_gpu_util=self._latest_feature_value(phys_feats, "gpu_util_seq", executed_target_idx),
                target_mem_util=self._latest_feature_value(phys_feats, "mem_util_seq", executed_target_idx),
                target_bandwidth=self._latest_feature_value(phys_feats, "bandwidth_seq", executed_target_idx),
            )
            if self.record_cfg.decision_pair_features_debug:
                row.update({
                    "offloading_latency_pair_features": self._json_for_record(
                        self._pair_feature_map(logic_feats, "offloading_latency_pair_feat", service_idx)
                    ),
                    "offloading_queue_pair_features": self._json_for_record(
                        self._pair_feature_map(logic_feats, "offloading_queue_pair_feat", service_idx)
                    ),
                })
            if self.record_cfg.decision_actor_debug:
                row.update({
                    "offloading_qk_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "qk_score", service_idx)
                    ),
                    "offloading_pair_bias": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "pair_bias", service_idx)
                    ),
                    "offloading_final_scores": self._json_for_record(
                        self._actor_debug_row_map(actor_debug, "final_score", service_idx)
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
            return time.time(), "interval", int(last_guard_trigger_seq)

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
                return time.time(), "latency_guard_trigger", int(current_seq)

            remaining_s = target_t - time.time()
            if remaining_s <= 0.0:
                return time.time(), "interval", last_guard_trigger_seq

            self._latency_guard_trigger_event.wait(timeout=min(remaining_s, poll_s))
            if self._latency_guard_trigger_event.is_set() and \
                    self._latency_guard_trigger_seq_value() <= last_guard_trigger_seq:
                self._latency_guard_trigger_event.clear()

        return time.time(), "stop", last_guard_trigger_seq

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
            "deployment_latency_pair_feat": torch.tensor(
                state_debug["pair_feature_snapshot"]["deployment_latency"],
                dtype=torch.float32,
            ),
            "offloading_latency_pair_feat": torch.tensor(
                state_debug["pair_feature_snapshot"]["offloading_latency"],
                dtype=torch.float32,
            ),
            "deployment_queue_pair_feat": torch.tensor(
                state_debug["pair_feature_snapshot"]["deployment_queue"],
                dtype=torch.float32,
            ),
            "offloading_queue_pair_feat": torch.tensor(
                state_debug["pair_feature_snapshot"]["offloading_queue"],
                dtype=torch.float32,
            ),
        }
        phys_feats = {
            "gpu_flops": torch.tensor(phys_feats_raw["gpu_flops"], dtype=torch.float32),
            "role_id": torch.tensor(phys_feats_raw["role_id"], dtype=torch.long),
            "mem_capacity": torch.tensor(phys_feats_raw["mem_capacity"], dtype=torch.float32),
            "bandwidth_seq": torch.tensor(phys_feats_raw["bandwidth_seq"], dtype=torch.float32),
            "gpu_util_seq": torch.tensor(phys_feats_raw["gpu_util_seq"], dtype=torch.float32),
            "mem_util_seq": torch.tensor(phys_feats_raw["mem_util_seq"], dtype=torch.float32),
        }
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
            "feedback_sample_shortfall": max(0, required_samples - actual_count),
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
            if self.state_cfg.deployment_feedback_timeout_s is not None:
                wait_timeout_s = min(5.0, max(0.5, self.state_cfg.deployment_feedback_timeout_s))
            count = self.state_buffer.wait_for_offloading_rewards(
                min_samples,
                timeout_s=wait_timeout_s,
                deployment_version=deployment_version,
            )
            if count >= min_samples:
                return HedgerDeploymentFeedbackWaitResult(ok=True, count=count)

            LOGGER.warning(
                f"[Hedger][Train][Deployment] Waiting for fresh offloading feedback: "
                f"version={deployment_version}, samples={count}/{min_samples}, "
                f"timeout={self._format_log_value(self.state_cfg.deployment_feedback_timeout_s, 2)}s"
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
            f"offloading={self.inference_cfg.run_offloading_worker}"
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
            f"interval={self._format_log_value(self.deployment_interval, 2)}s"
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

        step = 0
        while not self.deployment_thread_stop_event.is_set():
            try:
                current_decision_reason = next_decision_reason
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
                            deterministic=True,
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
                    self._pending_deployment_force_serve = current_decision_reason == "latency_guard_trigger"
                    self._pending_deployment_reason = current_decision_reason
                    decision_version = self._mark_deployment_decision_pending()
                if not self._wait_for_inference_deployment_served(decision_version):
                    break
                (
                    deployment_time_ticket,
                    next_decision_reason,
                    last_guard_trigger_seq,
                ) = self._sleep_until_next_inference_deployment_decision(
                    deployment_time_ticket,
                    self.deployment_interval,
                    last_guard_trigger_seq,
                )

                served_version = self.get_active_deployment_version()
                if next_decision_reason != "latency_guard_trigger":
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
                        dep_latency_guard_penalty_term=dep_reward_breakdown["dep_latency_guard_penalty_term"],
                        e2e_latency_count=metrics["e2e_latency_count"],
                        e2e_latency_mean=metrics["e2e_latency_mean"],
                        e2e_latency_latest=metrics["e2e_latency_latest"],
                        e2e_latency_p50=metrics["e2e_latency_p50"],
                        e2e_latency_p90=metrics["e2e_latency_p90"],
                        e2e_latency_p95=metrics["e2e_latency_p95"],
                        e2e_latency_p99=metrics["e2e_latency_p99"],
                        e2e_slo_violation=metrics["e2e_slo_violation"],
                        feedback_gate_enabled=int(bool(feedback_gate.get("enabled", False))),
                        feedback_gate_required_samples=int(feedback_gate.get("required", 0) or 0),
                        feedback_gate_collected_samples=int(feedback_gate.get("count", 0) or 0),
                        feedback_gate_timed_out=int(bool(feedback_gate.get("timed_out", False))),
                        feedback_gate_guard_truncated=int(bool(feedback_gate.get("guard_truncated", False))),
                        cap_relax_cnt=aux["capacity_relax_cnt"],
                        cap_relax_cost=aux["capacity_relax_cost"],
                        edge_cover_repair_cnt=aux.get("edge_cover_repair_cnt", 0),
                        edge_cover_repair_cost=aux.get("edge_cover_repair_cost", 0.0),
                        edge_cover_unmet=aux.get("edge_cover_unmet", 0),
                        policy_logp=self._scalar_value(logp),
                        policy_entropy=self._scalar_value(ent),
                        value_estimate=self._scalar_value(value),
                        raw_edge_replicas=raw_edge_replicas,
                        edge_replicas=edge_replicas,
                        cloud_replicas=cloud_replicas,
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
                        latency_guard_penalty_weight=self.deployment_agent_params["penalty_latency_guard_trigger"],
                        max_edge_replicas_per_device=self.deployment_agent_params["max_edge_replicas_per_device"],
                        edge_memory_budget_ratio=self.deployment_agent_params["edge_memory_budget_ratio"],
                        min_edge_replicas_per_service=self.deployment_agent_params["min_edge_replicas_per_service"],
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
                    actor_debug=aux.get("actor_debug"),
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
            f"interval={self._format_log_value(self.offloading_interval, 2)}s"
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
                            deterministic=True,
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

                off_reward_breakdown = self._compute_offloading_reward_breakdown(metrics, aux)
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
                raw_actions_cpu = aux["raw_actions"].detach().cpu()
                actions_cpu = actions.detach().cpu()
                raw_offloading_plan = self._map_offloading_mask_to_offloading_plan(raw_actions_cpu)
                raw_cloud_fraction = float((raw_actions_cpu == cloud_idx).float().mean().item())
                executed_cloud_fraction = float((actions_cpu == cloud_idx).float().mean().item())
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
                        aux_cost=aux["aux_cost"],
                        off_cloud_term=off_reward_breakdown["off_cloud_term"],
                        correction_cnt=aux["correction_cnt"],
                        correction_cost=aux["correction_cost"],
                        off_correction_term=off_reward_breakdown["off_correction_term"],
                        policy_logp=self._scalar_value(logp),
                        policy_entropy=self._scalar_value(ent),
                        value_estimate=self._scalar_value(value),
                        raw_cloud_fraction=raw_cloud_fraction,
                        executed_cloud_fraction=executed_cloud_fraction,
                        unique_targets=unique_targets,
                        feasible_targets_mean=feasible_targets_mean,
                        feasible_targets_min=feasible_targets_min,
                        feasible_targets_max=feasible_targets_max,
                        raw_offloading_plan=self._json_for_record(raw_offloading_plan),
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
                        correction_weight=self.offloading_agent_params["penalty_relax"],
                        loaded_checkpoint=self._loaded_checkpoint_path,
                    )
                    if self.record_cfg.actor_snapshot_debug:
                        row["state_offloading_actor_snapshot"] = self._json_for_record(aux.get("actor_debug"))
                    self.off_recorder.log_dict(row)
                self._log_offloading_decisions(
                    step=step,
                    raw_actions=raw_actions_cpu,
                    executed_actions=actions_cpu,
                    static_mask=static_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                    actor_debug=aux.get("actor_debug"),
                )
                LOGGER.debug(
                    f"[Hedger][Inference][Offloading] step={step}, "
                    f"decision_overhead={self._format_log_value(offloading_decision_overhead_s, 6)}s, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(new_logic_feats, new_phys_feats, metrics)}, "
                    f"correction_cnt={aux['correction_cnt']}, "
                    f"correction_cost={self._format_log_value(aux['correction_cost'])}, "
                    f"aux_cost={self._format_log_value(aux['aux_cost'])}"
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
                        del self.offloading_transitions[:self.training_cfg.offloading_rollout_len]
                        off_remaining = len(self.offloading_transitions)

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

                # Run a PPO update for the deployment agent.
                if self.stage_cfg.update_deployment_policy and \
                        len(self.deployment_transitions) >= self.training_cfg.deployment_rollout_len:
                    with self._data_lock:
                        dep_transitions = self.deployment_transitions[:self.training_cfg.deployment_rollout_len]
                        del self.deployment_transitions[:self.training_cfg.deployment_rollout_len]
                        dep_remaining = len(self.deployment_transitions)

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

        dep_train_fieldnames = ["step", "epoch", "decision_version", "dep_updates", "dep_reward",
                                "avg_off_reward", "off_reward_std", "off_reward_count", "dep_change_cost",
                                "dep_latency_cost", "dep_offload_term", "dep_latency_term",
                                "dep_slo_term", "dep_change_term", "dep_cloud_only_term",
                                "dep_capacity_relax_term", "dep_edge_cover_repair_term",
                                "dep_latency_guard_penalty_term",
                                "e2e_latency_count", "e2e_latency_mean", "e2e_latency_latest",
                                "e2e_latency_p50", "e2e_latency_p90", "e2e_latency_p95",
                                "e2e_latency_p99", "e2e_slo_violation",
                                "feedback_required_samples", "feedback_sample_shortfall",
                                "feedback_guard_interrupted",
                                "latency_guard_trigger_seq", "latency_guard_bad_ratio",
                                "latency_guard_bad_count", "latency_guard_sample_count",
                                "latency_guard_max_queue", "latency_guard_penalty_cost",
                                "cap_relax_cnt", "cap_relax_cost",
                                "edge_cover_repair_cnt", "edge_cover_repair_cost", "edge_cover_unmet",
                                "policy_logp", "policy_entropy", "value_estimate", "next_value",
                                "raw_edge_replicas", "edge_replicas", "cloud_replicas",
                                "cloud_only", "cloud_only_ratio",
                                "empty_edge_devices", "empty_edge_device_ratio",
                                "transition_buffer", "raw_deployment_plan", "deployment_plan",
                                *self._state_record_fieldnames()]
        if self.record_cfg.actor_snapshot_debug:
            dep_train_fieldnames.append("state_deployment_actor_snapshot")
        dep_train_fieldnames.extend([
            "dep_offload_weight", "dep_latency_weight", "dep_latency_transform",
            "dep_latency_normalizer", "dep_latency_clip", "dep_slo_weight",
            "dep_change_weight", "dep_cloud_only_weight", "cap_relax_weight", "edge_cover_repair_weight",
            "latency_guard_penalty_weight",
            "max_edge_replicas_per_device", "edge_memory_budget_ratio",
            "min_edge_replicas_per_service",
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

        while not self.deployment_thread_stop_event.is_set():
            try:
                if self._sleep_while_latency_guard_active("deployment worker"):
                    prev_deploy_mask = self._current_deploy_mask()
                    logic_feats, phys_feats, _, _, state_debug = self._collect_deployment_state(
                        prev_deploy_mask=prev_deploy_mask
                    )
                    continue

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
                    deploy_mask, logp, ent, value, aux = self.deployment_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        topo_order=None,  # Derived internally from the logical graph.
                        prev_deploy_mask=prev_deploy_mask_dev
                    )
                deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                with self._data_lock:
                    self.pending_deployment_plan = deploy_plan
                    self.pending_deploy_mask = deploy_mask.detach().cpu()
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

                    self._sleep_until_next_tick(
                        0,
                        self.deployment_interval,
                    )
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
                        "feedback_guard_interrupted": bool(feedback_result.guard_interrupted),
                        "latency_guard_penalty_cost": float(metrics["latency_guard_penalty_cost"]),
                    }

                    with self._data_lock:
                        self.deployment_transitions.append(tr)
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
                    dep_latency_guard_penalty_term=dep_reward_breakdown["dep_latency_guard_penalty_term"],
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
                    feedback_guard_interrupted=metrics["feedback_guard_interrupted"],
                    latency_guard_trigger_seq=metrics["latency_guard_trigger_seq"],
                    latency_guard_bad_ratio=metrics["latency_guard_bad_ratio"],
                    latency_guard_bad_count=metrics["latency_guard_bad_count"],
                    latency_guard_sample_count=metrics["latency_guard_sample_count"],
                    latency_guard_max_queue=metrics["latency_guard_max_queue"],
                    latency_guard_penalty_cost=metrics["latency_guard_penalty_cost"],
                    cap_relax_cnt=aux["capacity_relax_cnt"],
                    cap_relax_cost=aux["capacity_relax_cost"],
                    edge_cover_repair_cnt=aux.get("edge_cover_repair_cnt", 0),
                    edge_cover_repair_cost=aux.get("edge_cover_repair_cost", 0.0),
                    edge_cover_unmet=aux.get("edge_cover_unmet", 0),
                    policy_logp=float(logp.detach().cpu().item()),
                    policy_entropy=float(ent.detach().cpu().item()),
                    value_estimate=float(value.detach().cpu().item()),
                    next_value=float(next_value),
                    raw_edge_replicas=raw_edge_replicas,
                    edge_replicas=edge_replicas,
                    cloud_replicas=cloud_replicas,
                    cloud_only=cloud_only,
                    cloud_only_ratio=cloud_only_ratio,
                    empty_edge_devices=empty_edge_devices,
                    empty_edge_device_ratio=empty_edge_device_ratio,
                    transition_buffer=transition_count,
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
                    latency_guard_penalty_weight=self.deployment_agent_params["penalty_latency_guard_trigger"],
                    max_edge_replicas_per_device=self.deployment_agent_params["max_edge_replicas_per_device"],
                    edge_memory_budget_ratio=self.deployment_agent_params["edge_memory_budget_ratio"],
                    min_edge_replicas_per_service=self.deployment_agent_params["min_edge_replicas_per_service"],
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
                    actor_debug=aux.get("actor_debug"),
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
            f"rollout_agent={'frozen' if self._frozen_offloading_agent is not None else 'live'}"
        )

        off_train_fieldnames = ["step", "epoch", "off_updates", "off_reward", "latency", "latency_cost",
                                "off_latency_term", "off_latency_normalizer", "off_latency_clip", "off_latency_transform",
                                "slo_violation", "off_slo_term", "cloud_fraction", "off_cloud_term",
                                "aux_cost", "off_latency_weight", "off_slo_weight", "off_cloud_weight",
                                "correction_cnt", "correction_cost", "off_correction_term", "policy_logp",
                                "policy_entropy", "value_estimate",
                                "next_value", "raw_cloud_fraction", "executed_cloud_fraction",
                                "unique_targets", "feasible_targets_mean", "feasible_targets_min",
                                "feasible_targets_max", "transition_buffer", "raw_offloading_plan",
                                "offloading_plan", *self._state_record_fieldnames()]
        if self.record_cfg.actor_snapshot_debug:
            off_train_fieldnames.append("state_offloading_actor_snapshot")
        off_train_fieldnames.extend([
            "feedback_recorded", "feedback_deployment_version",
            "feedback_task_observations", "feedback_deployment_versions",
            "correction_weight",
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

                off_reward_breakdown = self._compute_offloading_reward_breakdown(metrics, aux)
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

                state_record = self._state_record_metrics(state_logic_feats, state_phys_feats, state_debug_record)
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                state_debug = new_state_debug

                cloud_idx = self.physical_topology.cloud_idx
                actions_cpu = actions.detach().cpu()
                raw_actions_cpu = aux["raw_actions"].detach().cpu()
                raw_offloading_plan = self._map_offloading_mask_to_offloading_plan(raw_actions_cpu)
                raw_cloud_fraction = float((raw_actions_cpu == cloud_idx).float().mean().item())
                executed_cloud_fraction = float((actions_cpu == cloud_idx).float().mean().item())
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
                    aux_cost=aux["aux_cost"],
                    off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                    off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                    off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                    correction_cnt=aux["correction_cnt"],
                    correction_cost=aux["correction_cost"],
                    off_correction_term=off_reward_breakdown["off_correction_term"],
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
                    raw_offloading_plan=self._json_for_record(raw_offloading_plan),
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
                    correction_weight=self.offloading_agent_params["penalty_relax"],
                )
                if self.record_cfg.actor_snapshot_debug:
                    row["state_offloading_actor_snapshot"] = self._json_for_record(aux.get("actor_debug"))
                self.off_recorder.log_dict(row)
                self._log_offloading_decisions(
                    step=step,
                    raw_actions=raw_actions_cpu,
                    executed_actions=actions_cpu,
                    static_mask=static_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                    actor_debug=aux.get("actor_debug"),
                )
                LOGGER.debug(
                    f"[Hedger][Train][{log_scope}] step={step}, reward={self._format_log_value(reward)}, "
                    f"{self._summarize_offloading_plan(self.offloading_plan)}, "
                    f"{self._summarize_state_snapshot(logic_feats, phys_feats, metrics)}, "
                    f"correction_cnt={aux['correction_cnt']}, "
                    f"correction_cost={self._format_log_value(aux['correction_cost'])}, "
                    f"aux_cost={self._format_log_value(aux['aux_cost'])}, "
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

    def _compute_offloading_reward_breakdown(self, metrics, aux) -> Dict[str, float]:
        """Named reward terms for offloading, reused by logging and PPO."""
        metrics = metrics or {}
        latency = float(metrics["latency"])
        latency_cost = self._compute_offloading_latency_cost(latency)
        slo_v = float(metrics["slo_violation"])
        cloud_frac = float(metrics["cloud_fraction"])

        w_lat = float(self.offloading_agent_params["reward_off_latency_weight"])
        w_slo = float(self.offloading_agent_params["reward_off_slo_weight"])
        w_cloud = float(self.offloading_agent_params["reward_off_cloud_weight"])
        correction_cost = float(aux.get("aux_cost", 0.0))

        terms = {
            "off_latency_term": -w_lat * latency_cost,
            "off_slo_term": -w_slo * slo_v,
            "off_cloud_term": -w_cloud * cloud_frac,
            "off_correction_term": -correction_cost,
        }
        return {
            "latency_cost": float(latency_cost),
            **terms,
            "reward": self._sum_reward_terms(terms),
        }

    def _compute_offloading_reward(self, metrics, aux) -> float:
        return self._compute_offloading_reward_breakdown(metrics, aux)["reward"]

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
        latency_guard_penalty_cost = float(metrics.get("latency_guard_penalty_cost", 0.0))

        w_off = float(self.deployment_agent_params["reward_dep_offload_weight"])
        w_lat = float(self.deployment_agent_params.get("reward_dep_latency_weight", 0.0))
        w_slo = float(self.deployment_agent_params.get("reward_dep_slo_weight", 0.0))
        w_change = float(self.deployment_agent_params["reward_dep_change_weight"])
        w_cloud_only = float(self.deployment_agent_params.get("reward_dep_cloud_only_weight", 0.0))
        penalty_capacity_relax = float(self.deployment_agent_params["penalty_capacity_relax"])
        penalty_edge_cover_repair = float(self.deployment_agent_params.get("penalty_edge_cover_repair", 0.0))
        penalty_latency_guard = float(self.deployment_agent_params.get("penalty_latency_guard_trigger", 0.0))

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
            "dep_latency_guard_penalty_term": -penalty_latency_guard * latency_guard_penalty_cost,
        }
        return {
            "dep_latency_cost": float(latency_cost),
            **terms,
            "reward": self._sum_reward_terms(terms),
        }

    def _compute_deployment_reward(self, metrics, aux) -> float:
        return self._compute_deployment_reward_breakdown(metrics, aux)["reward"]

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
