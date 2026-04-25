import threading
import time

from core.lib.common import LOGGER, Recorder
from core.lib.algorithms.shared.hedger import Hedger

from .offloading_support import HedgerHeuristicOffloadingMixin


class HedgerDeploymentOnly(HedgerHeuristicOffloadingMixin, Hedger):
    """Train Hedger deployment while replacing offloading PPO with a heuristic."""

    def _current_offloading_rollout_agent(self):
        return None

    def inference_hedger(self):
        assert self.logical_topology is not None and self.physical_topology is not None and self.state_buffer is not None
        if self.checkpoint_cfg.load.enabled and self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "[HedgerDeploymentOnly][Inference] Checkpoint loading was enabled but no checkpoint was loaded."
            )
        if not self.checkpoint_cfg.load.enabled:
            raise RuntimeError(
                "[HedgerDeploymentOnly][Inference] checkpoint.load.enabled must be true because "
                "the deployment policy is learned."
            )

        LOGGER.info(
            f"[HedgerDeploymentOnly][Inference] Start: "
            f"{self._summarize_runtime_config()}, {self._summarize_topology()}"
        )
        self.set_seed()
        self.shared_topology_encoder.eval()
        self.deployment_agent.eval()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        if self.cur_deploy_mask is None:
            if self.deployment_plan is not None:
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(self.deployment_plan).detach().cpu()
            elif self.initial_deployment_plan is not None:
                self.deployment_plan = self.initial_deployment_plan.copy()
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.initial_deployment_plan
                ).detach().cpu()
            else:
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask({})
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)

        worker = threading.Thread(target=self.inference_deployment_agent, daemon=True)
        worker.start()
        while not self.deployment_thread_stop_event.is_set():
            if not worker.is_alive():
                LOGGER.warning("[HedgerDeploymentOnly][Inference] Deployment worker stopped unexpectedly.")
                break
            time.sleep(0.5)
        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()

    def train_offloading_agent(self):
        if not self.stage_cfg.run_offloading_worker:
            LOGGER.info("[HedgerAblation][Deployment][OffloadingHeuristic] Worker disabled by stage.")
            return
        assert self.logical_topology is not None and self.physical_topology is not None

        log_scope = "OffloadingHeuristic"
        LOGGER.info(
            f"[HedgerAblation][Deployment][{log_scope}] Worker started: "
            f"interval={self._format_log_value(self.offloading_interval, 2)}s"
        )
        self.off_recorder = Recorder(
            self._stage_log_path("offloading_train.csv"),
            fmt="csv",
            fieldnames=["step", "epoch", "off_updates", "off_reward", "latency", "latency_cost",
                        "off_latency_term", "off_latency_normalizer", "off_latency_clip",
                        "off_latency_transform", "slo_violation", "off_slo_term", "cloud_fraction",
                        "off_cloud_term", "aux_cost", "off_latency_weight", "off_slo_weight", "off_cloud_weight",
                        "correction_cnt", "correction_cost", "off_correction_term",
                        "policy_logp", "policy_entropy", "value_estimate",
                        "next_value", "raw_cloud_fraction", "executed_cloud_fraction",
                        "unique_targets", "feasible_targets_mean", "feasible_targets_min",
                        "feasible_targets_max", "transition_buffer", "raw_offloading_plan",
                        "offloading_plan", *self._state_record_fieldnames(),
                        "feedback_recorded", "feedback_deployment_version",
                        "feedback_task_observations", "feedback_deployment_versions",
                        "correction_weight"],
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
        step = 0
        offloading_time_ticket = 0
        logic_feats, phys_feats, _, _ = self._collect_offloading_state()
        last_reward_task_version = self.state_buffer.get_task_observation_version() if self.state_buffer is not None else 0
        while not self.offloading_thread_stop_event.is_set():
            try:
                if self._sleep_while_latency_guard_active(f"{log_scope} worker"):
                    logic_feats, phys_feats, _, _ = self._collect_offloading_state()
                    last_reward_task_version = (
                        self.state_buffer.get_task_observation_version()
                        if self.state_buffer is not None else last_reward_task_version
                    )
                    continue

                static_mask = self._current_deploy_mask()
                actions, aux = self.heuristic_offloading_actions(
                    logic_edge_index=logic_edge_index,
                    logic_feats=logic_feats,
                    phys_feats=phys_feats,
                    static_mask=static_mask,
                )
                offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                with self._data_lock:
                    self.offloading_plan = offloading_plan

                offloading_time_ticket = self._sleep_until_next_tick(offloading_time_ticket, self.offloading_interval)
                if self.is_latency_guard_active():
                    logic_feats, phys_feats, _, _ = self._collect_offloading_state()
                    last_reward_task_version = (
                        self.state_buffer.get_task_observation_version()
                        if self.state_buffer is not None else last_reward_task_version
                    )
                    continue

                task_feedback_summary = self.state_buffer.get_task_observation_deployment_summary(last_reward_task_version)
                current_task_version = int(task_feedback_summary["current_version"])
                feedback_deployment_version = None
                if task_feedback_summary.get("all_same_deployment_version") and task_feedback_summary.get("count", 0) > 0:
                    feedback_deployment_version = task_feedback_summary.get("dominant_deployment_version")
                new_logic_feats, new_phys_feats, metrics, _ = self._collect_offloading_state(
                    since_task_version=last_reward_task_version,
                    deployment_version=feedback_deployment_version,
                )
                if current_task_version <= last_reward_task_version:
                    logic_feats = new_logic_feats
                    phys_feats = new_phys_feats
                    continue
                last_reward_task_version = current_task_version

                off_reward_breakdown = self._compute_offloading_reward_breakdown(metrics, aux)
                latency_cost = off_reward_breakdown["latency_cost"]
                reward = off_reward_breakdown["reward"]
                feedback_recorded = False
                if feedback_deployment_version is not None:
                    feedback_recorded = self.state_buffer.add_offloading_reward(
                        reward,
                        task_version=current_task_version,
                        deployment_version=feedback_deployment_version,
                    )

                state_logic_feats = logic_feats
                state_phys_feats = phys_feats
                state_record = self._state_record_metrics(state_logic_feats, state_phys_feats)
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats

                cloud_idx = self.physical_topology.cloud_idx
                raw_actions_cpu = actions.detach().cpu()
                cloud_fraction = float((raw_actions_cpu == cloud_idx).float().mean().item()) if raw_actions_cpu.numel() else 0.0
                feasible_counts = static_mask.detach().cpu().float().sum(dim=1)
                transition_count = len(self.offloading_transitions)
                self.off_recorder.log(
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
                    correction_cnt=0,
                    correction_cost=0.0,
                    off_correction_term=off_reward_breakdown["off_correction_term"],
                    policy_logp=0.0,
                    policy_entropy=0.0,
                    value_estimate=0.0,
                    next_value=0.0,
                    raw_cloud_fraction=cloud_fraction,
                    executed_cloud_fraction=cloud_fraction,
                    unique_targets=int(raw_actions_cpu.unique().numel()) if raw_actions_cpu.numel() else 0,
                    feasible_targets_mean=float(feasible_counts.mean().item()) if feasible_counts.numel() else 0.0,
                    feasible_targets_min=float(feasible_counts.min().item()) if feasible_counts.numel() else 0.0,
                    feasible_targets_max=float(feasible_counts.max().item()) if feasible_counts.numel() else 0.0,
                    transition_buffer=transition_count,
                    raw_offloading_plan=self._json_for_record(offloading_plan),
                    offloading_plan=self._json_for_record(offloading_plan),
                    **state_record,
                    feedback_recorded=feedback_recorded,
                    feedback_deployment_version=feedback_deployment_version,
                    feedback_task_observations=task_feedback_summary.get("count", 0),
                    feedback_deployment_versions=self._json_for_record(
                        {str(k): v for k, v in task_feedback_summary.get("deployment_version_counts", {}).items()}
                    ),
                    correction_weight=self.offloading_agent_params["penalty_relax"],
                )
                self._log_offloading_decisions(
                    step=step,
                    raw_actions=raw_actions_cpu,
                    executed_actions=raw_actions_cpu,
                    static_mask=static_mask,
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                )
                step += 1
            except Exception as exc:
                LOGGER.exception(f"[HedgerAblation][Deployment][{log_scope}] Worker loop error: {exc}")
                time.sleep(0.5)

        self.off_recorder.close()
        self.off_decision_recorder.close()
        LOGGER.info(f"[HedgerAblation][Deployment][{log_scope}] Worker stopped.")
