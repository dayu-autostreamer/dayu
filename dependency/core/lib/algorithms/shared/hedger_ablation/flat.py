import copy
import os
import threading
import time

import torch

from core.lib.common import LOGGER, Recorder
from core.lib.algorithms.shared.hedger import Hedger
from core.lib.algorithms.shared.hedger.hedger_config import (
    DeploymentConstraintCfg,
    from_partial_dict,
)
from .flat_ppo_agent import HedgerFlatPPO


class HedgerFlat(Hedger):
    """Flat ablation with one PPO module deciding deployment and offloading together."""

    def register_deployment_agent(self):
        return

    def register_offloading_agent(self):
        if getattr(self, "flat_agent", None) is not None:
            return
        assert self.shared_topology_encoder, 'Shared topology encoder must be registered before flat agent.'
        self.flat_agent = HedgerFlatPPO(
            encoder=self.shared_topology_encoder,
            d_model=self.encoder_cfg.embedding_dim,
            actor_lr=min(self.deployment_agent_params['actor_lr'], self.offloading_agent_params['actor_lr']),
            critic_lr=max(self.deployment_agent_params['critic_lr'], self.offloading_agent_params['critic_lr']),
            gamma=self.offloading_agent_params['gamma'],
            lamda=self.offloading_agent_params['lamda'],
            clip_eps=min(self.deployment_agent_params['clip_eps'], self.offloading_agent_params['clip_eps']),
            update_encoder=self.offloading_agent_params['update_encoder'],
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
            deployment_constraint_cfg=from_partial_dict(DeploymentConstraintCfg, self.deployment_agent_params),
        ).to(self.device)
        self.deployment_agent = self.flat_agent
        self.offloading_agent = self.flat_agent
        self._sync_agent_topology_bindings()

    def _sync_agent_topology_bindings(self):
        super()._sync_agent_topology_bindings()
        flat_agent = getattr(self, "flat_agent", None)
        if flat_agent is not None and self.physical_topology is not None:
            flat_agent.cloud_idx = self.physical_topology.cloud_idx

    def _build_checkpoint_payload(self, stage_step: int) -> dict:
        return {
            'encoder': self.shared_topology_encoder.state_dict(),
            'flat_agent': self.flat_agent.state_dict(),
            'flat_actor_opt': self.flat_agent.actor_opt.state_dict(),
            'flat_critic_opt': self.flat_agent.critic_opt.state_dict(),
            'meta': {
                'schema_version': 3,
                'ablation': 'hedger-flat',
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

    def load_checkpoint(self):
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
            LOGGER.warning(f"[HedgerFlat][Checkpoint] No checkpoint found: path={target_path}")
            return
        with self._model_lock:
            ckpt = torch.load(target_path, map_location=self.device)
            meta = ckpt.get('meta', {})
            self._loaded_checkpoint_path = target_path
            if self.checkpoint_cfg.load.restore_encoder and ckpt.get('encoder') is not None:
                self._load_encoder_state(ckpt['encoder'])
            flat_state = ckpt.get('flat_agent')
            if flat_state is not None:
                self._load_state_dict_compatible(self.flat_agent, flat_state, "flat_agent")
            else:
                self._load_flat_agent_from_hierarchical_checkpoint(ckpt)
            if self.checkpoint_cfg.load.restore_optimizer:
                if 'flat_actor_opt' in ckpt:
                    try:
                        self.flat_agent.actor_opt.load_state_dict(ckpt['flat_actor_opt'])
                        self._move_optimizer_state(self.flat_agent.actor_opt, self.device)
                    except ValueError as exc:
                        LOGGER.warning(f"[HedgerFlat][Checkpoint] Skip actor optimizer: {exc}")
                if 'flat_critic_opt' in ckpt:
                    try:
                        self.flat_agent.critic_opt.load_state_dict(ckpt['flat_critic_opt'])
                        self._move_optimizer_state(self.flat_agent.critic_opt, self.device)
                    except ValueError as exc:
                        LOGGER.warning(f"[HedgerFlat][Checkpoint] Skip critic optimizer: {exc}")

        same_stage_resume = meta.get('training_stage') == current_stage
        self._global_update_step = int(meta.get('global_step', meta.get('stage_step', 0)))
        if self.checkpoint_cfg.load.reset_stage_counters or not same_stage_resume:
            self._deployment_update_steps = 0
            self._offloading_update_steps = 0
            self._epoch = 0
        else:
            self._deployment_update_steps = int(meta.get('deployment_updates', 0))
            self._offloading_update_steps = int(meta.get('offloading_updates', 0))
            self._epoch = int(meta.get('stage_step', self._epoch))

    def _load_flat_agent_from_hierarchical_checkpoint(self, ckpt: dict) -> None:
        loaded_any = False
        if self.checkpoint_cfg.load.restore_deployment_agent:
            dep_state = ckpt.get('deployment_agent')
            if dep_state is not None:
                self._load_state_dict_compatible(
                    self.flat_agent,
                    dep_state,
                    "flat_agent_from_deployment_agent",
                )
                loaded_any = True
            else:
                LOGGER.warning("[HedgerFlat][Checkpoint] Missing deployment_agent state in hierarchical checkpoint.")

        if self.checkpoint_cfg.load.restore_offloading_agent:
            off_state = ckpt.get('offloading_agent')
            if off_state is not None:
                mapped_state = self._map_offloading_agent_state_to_flat(
                    off_state,
                    include_shared=not loaded_any,
                )
                self._load_state_dict_compatible(
                    self.flat_agent,
                    mapped_state,
                    "flat_agent_from_offloading_agent",
                )
                loaded_any = True
            else:
                LOGGER.warning("[HedgerFlat][Checkpoint] Missing offloading_agent state in hierarchical checkpoint.")

        if not loaded_any:
            LOGGER.warning(
                "[HedgerFlat][Checkpoint] Missing flat_agent state and no compatible hierarchical agent "
                "state was loaded."
            )

    @staticmethod
    def _map_offloading_agent_state_to_flat(state_dict: dict, include_shared: bool = False) -> dict:
        mapped = {}
        shared_prefixes = ("encoder.", "service_adapter.", "device_adapter.", "critic.")
        for key, value in (state_dict or {}).items():
            if key.startswith("actor."):
                mapped[f"offload_actor.{key[len('actor.'):]}"] = value
            elif include_shared and key.startswith(shared_prefixes):
                mapped[key] = value
        return mapped

    def inference_hedger(self):
        assert self.logical_topology is not None and self.physical_topology is not None and self.state_buffer is not None
        if self.checkpoint_cfg.load.enabled and self._loaded_checkpoint_path is None:
            raise RuntimeError("[HedgerFlat][Inference] Checkpoint loading was enabled but no checkpoint was loaded.")
        LOGGER.info(f"[HedgerFlat][Inference] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        self.flat_agent.eval()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()
        worker = threading.Thread(target=self.inference_flat_agent, daemon=True)
        worker.start()
        while not self.deployment_thread_stop_event.is_set():
            if not worker.is_alive():
                LOGGER.warning("[HedgerFlat][Inference] Worker stopped unexpectedly.")
                break
            time.sleep(0.5)
        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()

    def inference_flat_agent(self):
        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)
        tick = 0
        prev_deploy_mask = self._current_deploy_mask()
        logic_feats, phys_feats, _, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
        while not self.deployment_thread_stop_event.is_set():
            try:
                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None
                with self._model_lock, torch.inference_mode():
                    deploy_mask, actions, _, _, _, _ = self.flat_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        prev_deploy_mask=prev_dev,
                        deterministic=True,
                    )
                deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                with self._data_lock:
                    self.pending_deployment_plan = deploy_plan
                    self.pending_deploy_mask = deploy_mask.detach().cpu()
                    self.offloading_plan = offloading_plan
                    self._mark_deployment_decision_pending()
                tick = self._sleep_until_next_tick(tick, self.deployment_interval)
                prev_deploy_mask = self._current_deploy_mask()
                logic_feats, phys_feats, _, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
            except Exception as exc:
                LOGGER.exception(f"[HedgerFlat][Inference] Worker loop error: {exc}")
                time.sleep(0.5)

    def train_hedger(self):
        assert self.logical_topology is not None and self.physical_topology is not None
        LOGGER.info(f"[HedgerFlat][Train] Start: {self._summarize_runtime_config()}, {self._summarize_topology()}")
        self.set_seed()
        self.flat_agent.train()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()
        self.flat_transitions = []

        if self.cur_deploy_mask is None:
            self.cur_deploy_mask = torch.zeros((len(self.logical_topology), len(self.physical_topology)), dtype=torch.bool)
            self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)

        self.flat_recorder = Recorder(
            self._stage_log_path("flat_train.csv"),
            fmt="csv",
            fieldnames=["step", "epoch", "update", "decision_version", "reward", "latency",
                        "slo_violation", "cloud_fraction", "dep_change_cost", "cap_relax_cnt",
                        "cap_relax_cost", "edge_cover_repair_cnt", "edge_cover_unmet",
                        "policy_logp", "policy_entropy", "value_estimate", "next_value", "transition_buffer",
                        "deployment_plan", "offloading_plan"],
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
        self.off_decision_recorder = Recorder(
            self._stage_log_path("offloading_decisions.csv"),
            fmt="csv",
            fieldnames=self._offloading_decision_fieldnames(),
            overwrite=True,
            flush_every=50,
        )
        self.flat_update_recorder = Recorder(
            self._stage_log_path("flat_ppo_updates.csv"),
            fmt="csv",
            fieldnames=self._ppo_update_fieldnames(),
            overwrite=True,
            flush_every=1,
        )

        worker = threading.Thread(target=self.train_flat_agent, daemon=True)
        worker.start()
        while True:
            try:
                if not worker.is_alive():
                    LOGGER.warning("[HedgerFlat][Train] Worker stopped unexpectedly.")
                    break
                if self.is_latency_guard_active():
                    if self.latency_guard_cfg.clear_transition_buffers:
                        with self._data_lock:
                            self.flat_transitions.clear()
                    self._sleep_while_latency_guard_active("flat coordinator")
                    continue
                if len(self.flat_transitions) >= self.training_cfg.deployment_rollout_len:
                    with self._data_lock:
                        transitions = self.flat_transitions[:self.training_cfg.deployment_rollout_len]
                        del self.flat_transitions[:self.training_cfg.deployment_rollout_len]
                        remaining = len(self.flat_transitions)
                    with self._model_lock:
                        ppo_cfg = self.deployment_agent_params["ppo"]
                        entropy_coef = self._scheduled_entropy_coef(ppo_cfg, self._epoch + 1)
                        stats = self.flat_agent.ppo_update(
                            transitions,
                            epochs=self.training_cfg.ppo_epochs,
                            batch_size=self.training_cfg.deployment_batch_size,
                            entropy_coef=entropy_coef,
                            value_coef=ppo_cfg.value_coef,
                        )
                    self._deployment_update_steps += 1
                    self._offloading_update_steps += 1
                    self._epoch += 1
                    self._global_update_step += 1
                    self._record_ppo_update(
                        self.flat_update_recorder,
                        "flat",
                        self._epoch,
                        used=len(transitions),
                        remaining=remaining,
                        stats=stats,
                    )
                    if self._epoch % self.checkpoint_cfg.save.interval_updates == 0:
                        self.save_checkpoint(stage_step=self._epoch, is_final=False)
                if self._epoch >= self.training_cfg.total_updates:
                    break
                time.sleep(0.5)
            except Exception as exc:
                LOGGER.exception(f"[HedgerFlat][Train] Coordinator loop error: {exc}")

        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()
        try:
            self.save_checkpoint(stage_step=self._epoch, is_final=True)
        except Exception as exc:
            LOGGER.exception(f"[HedgerFlat][Train] Failed to save final checkpoint: {exc}")
        self.flat_recorder.close()
        self.dep_decision_recorder.close()
        self.off_decision_recorder.close()
        self.flat_update_recorder.close()

    def train_flat_agent(self):
        logic_edge_index = self._build_edge_index(self.logical_topology.links)
        phys_edge_index = self._build_edge_index(self.physical_topology.links)
        step = 0
        deployment_time_ticket = 0
        prev_deploy_mask = self._current_deploy_mask()
        logic_feats, phys_feats, _, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
        last_task_version = self.state_buffer.get_task_observation_version() if self.state_buffer is not None else 0
        while not self.deployment_thread_stop_event.is_set():
            try:
                if self._sleep_while_latency_guard_active("flat worker"):
                    prev_deploy_mask = self._current_deploy_mask()
                    logic_feats, phys_feats, _, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
                    last_task_version = self.state_buffer.get_task_observation_version()
                    continue

                logic_feats_dev = {k: v.to(self.device) for k, v in logic_feats.items()}
                phys_feats_dev = {k: v.to(self.device) for k, v in phys_feats.items()}
                prev_dev = prev_deploy_mask.to(self.device) if prev_deploy_mask is not None else None
                with self._model_lock, torch.no_grad():
                    deploy_mask, actions, logp, ent, value, aux = self.flat_agent.policy(
                        logic_edge_index=logic_edge_index,
                        logic_feats=logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=phys_feats_dev,
                        prev_deploy_mask=prev_dev,
                    )
                deploy_plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)
                offloading_plan = self._map_offloading_mask_to_offloading_plan(actions)
                with self._data_lock:
                    self.pending_deployment_plan = deploy_plan
                    self.pending_deploy_mask = deploy_mask.detach().cpu()
                    self.offloading_plan = offloading_plan
                    decision_version = self._mark_deployment_decision_pending()

                if not self._wait_for_deployment_decision_served(decision_version, abort_on_guard=False):
                    break
                deployment_time_ticket = self._sleep_until_next_tick(
                    deployment_time_ticket,
                    self.deployment_interval,
                )
                if self.is_latency_guard_active():
                    continue

                task_feedback_summary = self.state_buffer.get_task_observation_deployment_summary(last_task_version)
                current_task_version = int(task_feedback_summary["current_version"])
                if current_task_version <= last_task_version:
                    logic_feats, phys_feats, _, _, _ = self._collect_deployment_state(
                        prev_deploy_mask=deploy_mask.detach().cpu()
                    )
                    continue
                feedback_version = decision_version if task_feedback_summary.get("count", 0) > 0 else None
                new_logic_feats, new_phys_feats, metrics, done, _ = self._collect_offloading_state(
                    since_task_version=last_task_version,
                    deployment_version=feedback_version,
                )
                last_task_version = current_task_version
                off_reward_breakdown = self._compute_offloading_reward_breakdown(metrics)
                off_reward = off_reward_breakdown["reward"]

                exec_deploy_mask = deploy_mask.detach().cpu().bool()
                layout_metrics = self._deployment_layout_metrics(exec_deploy_mask)
                aux["cloud_only_ratio"] = float(layout_metrics["cloud_only_ratio"])
                aux["empty_edge_device_ratio"] = float(layout_metrics["empty_edge_device_ratio"])
                dep_metrics = {
                    "avg_offloading_reward": off_reward,
                    "deploy_change_cost": self._compute_deploy_change_cost(prev_deploy_mask),
                    "e2e_latency_count": metrics.get("task_latency_count", 0),
                    "e2e_latency_mean": metrics.get("latency", 0.0),
                    "e2e_slo_violation": metrics.get("slo_violation", 0.0),
                    "latency_guard_penalty_cost": 0.0,
                }
                dep_reward = self._compute_deployment_reward(dep_metrics, aux)
                reward = 0.5 * off_reward + 0.5 * dep_reward

                new_logic_feats_dev = {k: v.to(self.device) for k, v in new_logic_feats.items()}
                new_phys_feats_dev = {k: v.to(self.device) for k, v in new_phys_feats.items()}
                with self._model_lock, torch.no_grad():
                    next_value = float(self.flat_agent.estimate_value(
                        logic_edge_index=logic_edge_index,
                        logic_feats=new_logic_feats_dev,
                        phys_edge_index=phys_edge_index,
                        phys_feats=new_phys_feats_dev,
                        prev_deploy_mask=deploy_mask.detach(),
                    ).detach().cpu().item())

                tr = {
                    "logic_edge_index": logic_edge_index.cpu(),
                    "logic_feats": {k: v.cpu() for k, v in logic_feats_dev.items()},
                    "phys_edge_index": phys_edge_index.cpu(),
                    "phys_feats": {k: v.cpu() for k, v in phys_feats_dev.items()},
                    "raw_deploy_mask": aux["raw_deploy_mask"].detach().cpu(),
                    "actions": actions.detach().cpu(),
                    "static_mask": deploy_mask.detach().cpu(),
                    "prev_deploy_mask": prev_deploy_mask.cpu() if prev_deploy_mask is not None else None,
                    "topo_order": None,
                    "logp": logp.detach().cpu(),
                    "value": value.detach().cpu(),
                    "next_value": float(next_value),
                    "reward": float(reward),
                    "done": bool(done),
                }
                with self._data_lock:
                    self.flat_transitions.append(tr)
                    transition_count = len(self.flat_transitions)

                state_logic_feats = logic_feats
                state_phys_feats = phys_feats
                logic_feats = new_logic_feats
                phys_feats = new_phys_feats
                prev_deploy_mask = deploy_mask.detach().cpu()
                self.flat_recorder.log(
                    step=step,
                    epoch=self._epoch,
                    update=self._epoch,
                    decision_version=decision_version,
                    reward=reward,
                    latency=metrics["latency"],
                    slo_violation=metrics["slo_violation"],
                    cloud_fraction=metrics["cloud_fraction"],
                    dep_change_cost=dep_metrics["deploy_change_cost"],
                    cap_relax_cnt=aux["capacity_relax_cnt"],
                    cap_relax_cost=aux["capacity_relax_cost"],
                    edge_cover_repair_cnt=aux.get("edge_cover_repair_cnt", 0),
                    edge_cover_unmet=aux.get("edge_cover_unmet", 0),
                    policy_logp=float(logp.detach().cpu().item()),
                    policy_entropy=float(ent.detach().cpu().item()),
                    value_estimate=float(value.detach().cpu().item()),
                    next_value=next_value,
                    transition_buffer=transition_count,
                    deployment_plan=self._json_for_record(deploy_plan),
                    offloading_plan=self._json_for_record(offloading_plan),
                )
                self._log_deployment_decisions(
                    step=step,
                    raw_deploy_mask=aux["raw_deploy_mask"].detach().cpu(),
                    exec_deploy_mask=exec_deploy_mask,
                    logic_feats=state_logic_feats,
                )
                self._log_offloading_decisions(
                    step=step,
                    actions=actions.detach().cpu(),
                    static_mask=deploy_mask.detach().cpu(),
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                )
                step += 1
            except Exception as exc:
                LOGGER.exception(f"[HedgerFlat][Train] Worker loop error: {exc}")
                time.sleep(0.5)
