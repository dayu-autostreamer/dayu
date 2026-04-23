import copy
import math
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import torch

from core.lib.common import LOGGER, Recorder

from .hedger import Hedger
from .hedger_config import from_partial_dict, DeploymentConstraintCfg, OffloadingConstraintCfg
from .ppo_agent import HedgerFlatPPO


class HedgerHeuristicMixin:
    """Shared heuristic decisions used by Hedger ablation controllers."""

    def _edge_count_target(self) -> int:
        return max(1, int(self.deployment_agent_params.get("min_edge_replicas_per_service", 1) or 1))

    def _max_edge_replicas_per_device(self) -> Optional[int]:
        value = self.deployment_agent_params.get("max_edge_replicas_per_device")
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    @staticmethod
    def _latest_seq_value(feats: Dict[str, torch.Tensor], key: str, idx: int, default: float = 0.0) -> float:
        value = feats.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            return float(default)
        try:
            if value.dim() >= 2:
                return float(value[idx, -1].item())
            return float(value[idx].item())
        except Exception:
            return float(default)

    @staticmethod
    def _topological_order(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
        row, col = edge_index
        indeg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        indeg.scatter_add_(0, col, torch.ones_like(col))
        queue = [i for i in range(num_nodes) if int(indeg[i].item()) == 0]
        order = []
        adj = [[] for _ in range(num_nodes)]
        for u, v in zip(row.tolist(), col.tolist()):
            adj[int(u)].append(int(v))
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in adj[node]:
                indeg[child] -= 1
                if int(indeg[child].item()) == 0:
                    queue.append(child)
        if len(order) < num_nodes:
            seen = set(order)
            order += [idx for idx in range(num_nodes) if idx not in seen]
        return order

    @staticmethod
    def _parents(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        row, col = edge_index
        parents = [[] for _ in range(num_nodes)]
        for u, v in zip(row.tolist(), col.tolist()):
            parents[int(v)].append(int(u))
        return parents

    def _fallback_edge_deployment_plan(self, info: Optional[dict] = None, default_deployment=None) -> dict:
        if isinstance(default_deployment, dict) and default_deployment:
            return copy.deepcopy(default_deployment)
        if self.logical_topology is None:
            return {}
        edge_nodes = []
        if isinstance(info, dict):
            edge_nodes = list(info.get("node_set") or [])
        if not edge_nodes and self.physical_topology is not None:
            edge_nodes = list(self.physical_topology.nodes[:self.physical_topology.cloud_idx])
        if not edge_nodes:
            return {service: [] for service in self.logical_topology.service_list}
        plan = {}
        for idx, service in enumerate(self.logical_topology.service_list):
            plan[service] = [edge_nodes[idx % len(edge_nodes)]]
        return plan

    def _heuristic_deployment_mask(self) -> Optional[torch.Tensor]:
        if self.logical_topology is None or self.physical_topology is None or self.state_buffer is None:
            return None
        try:
            prev_deploy_mask = self._current_deploy_mask()
            logic_feats, phys_feats = self._collect_graph_state(self.state_cfg.deployment_seq_len)
        except Exception as exc:
            LOGGER.warning(f"[HedgerAblation][DeploymentHeuristic] State unavailable, use fallback: {exc}")
            return None

        num_services = len(self.logical_topology)
        num_devices = len(self.physical_topology)
        cloud_idx = self.physical_topology.cloud_idx
        if num_services <= 0 or cloud_idx <= 0:
            return None

        model_mem = logic_feats["model_mem"].float()
        if self.deployment_agent is not None and hasattr(self.deployment_agent, "_initial_residual_mem"):
            residual = self.deployment_agent._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask).float()
            static_allowed = self.deployment_agent._static_allowed_mask(phys_feats, logic_feats).bool()
        else:
            cap = phys_feats["mem_capacity"].float()
            util = phys_feats["mem_util_seq"][:, -1].float()
            residual = cap * torch.clamp(1.0 - util, min=0.0)
            static_allowed = model_mem.view(num_services, 1) <= residual.view(1, num_devices)
            static_allowed[:, cloud_idx] = True

        target_edges = self._edge_count_target()
        max_per_device = self._max_edge_replicas_per_device()
        used_count = torch.zeros(cloud_idx, dtype=torch.long)
        used_mem = torch.zeros(cloud_idx, dtype=torch.float32)
        deploy_mask = torch.zeros((num_services, num_devices), dtype=torch.bool)
        deploy_mask[:, cloud_idx] = True

        service_order = sorted(
            range(num_services),
            key=lambda s_idx: (
                -float(model_mem[s_idx].item()),
                -self._latest_seq_value(logic_feats, "hist_latency_seq", s_idx, 0.0),
                int(s_idx),
            ),
        )

        for service_idx in service_order:
            service_mem = float(model_mem[service_idx].item())
            candidates = []
            for device_idx in range(cloud_idx):
                if not bool(static_allowed[service_idx, device_idx].item()):
                    continue
                if max_per_device is not None and int(used_count[device_idx].item()) >= max_per_device:
                    continue
                remaining_after = float(residual[device_idx].item()) - float(used_mem[device_idx].item()) - service_mem
                if remaining_after < -1e-6:
                    continue
                gpu_util = self._latest_seq_value(phys_feats, "gpu_util_seq", device_idx, 0.0)
                mem_util = self._latest_seq_value(phys_feats, "mem_util_seq", device_idx, 0.0)
                bandwidth = max(1.0, self._latest_seq_value(phys_feats, "bandwidth_seq", device_idx, 1.0))
                was_selected = bool(prev_deploy_mask[service_idx, device_idx].item())
                device_empty = int(used_count[device_idx].item()) == 0
                cost = (
                    0.45 * gpu_util
                    + 0.45 * mem_util
                    - 0.05 * math.log1p(bandwidth)
                    - (0.15 if was_selected else 0.0)
                    - (0.05 if device_empty else 0.0)
                    - 0.02 * remaining_after
                )
                candidates.append((cost, int(used_count[device_idx].item()), int(device_idx), remaining_after))

            for _, _, device_idx, _ in sorted(candidates)[:target_edges]:
                deploy_mask[service_idx, device_idx] = True
                used_count[device_idx] += 1
                used_mem[device_idx] += service_mem

        return deploy_mask

    def set_heuristic_deployment_plan(
            self,
            info: Optional[dict] = None,
            default_deployment=None,
            mark_version: bool = False,
    ) -> dict:
        deploy_mask = self._heuristic_deployment_mask()
        if deploy_mask is None:
            plan = self._fallback_edge_deployment_plan(info=info, default_deployment=default_deployment)
            if self.logical_topology is not None and self.physical_topology is not None:
                deploy_mask = self._map_deployment_plan_to_deployment_mask(plan).detach().cpu()
            else:
                deploy_mask = None
        else:
            plan = self._map_deployment_mask_to_deployment_plan(deploy_mask)

        with self._data_lock:
            self.deployment_plan = copy.deepcopy(plan)
            if deploy_mask is not None:
                self.cur_deploy_mask = deploy_mask.detach().cpu()
            if mark_version:
                self.pending_deployment_plan = None
                self.pending_deploy_mask = None

        if mark_version:
            self._mark_deployment_decision_pending()
            self._mark_deployment_decision_served()

        return copy.deepcopy(plan)

    def heuristic_offloading_actions(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_services, num_devices = static_mask.size()
        cloud_idx = self.physical_topology.cloud_idx
        source_idx = self.physical_topology.source_idx
        topo_order = self._topological_order(logic_edge_index, num_services)
        parents = self._parents(logic_edge_index, num_services)
        actions = torch.full((num_services,), cloud_idx, dtype=torch.long)

        for service_idx in topo_order:
            allowed = torch.nonzero(static_mask[service_idx].bool(), as_tuple=False).flatten().tolist()
            if not allowed:
                allowed = [cloud_idx]
            if any(int(actions[parent].item()) == cloud_idx for parent in parents[service_idx]):
                actions[service_idx] = cloud_idx if cloud_idx in allowed else int(allowed[0])
                continue

            parent_targets = [int(actions[parent].item()) for parent in parents[service_idx]]
            candidates = []
            for device_idx in allowed:
                gpu_util = self._latest_seq_value(phys_feats, "gpu_util_seq", device_idx, 0.0)
                mem_util = self._latest_seq_value(phys_feats, "mem_util_seq", device_idx, 0.0)
                bandwidth = max(1.0, self._latest_seq_value(phys_feats, "bandwidth_seq", device_idx, 1.0))
                parent_switch = 0.0
                if parent_targets:
                    parent_switch = sum(1 for target in parent_targets if target != device_idx) / float(len(parent_targets))
                elif device_idx != source_idx:
                    parent_switch = 1.0
                cloud_penalty = 0.6 if device_idx == cloud_idx else 0.0
                cost = (
                    0.35 * gpu_util
                    + 0.35 * mem_util
                    + 0.20 * parent_switch
                    + cloud_penalty
                    - 0.05 * math.log1p(bandwidth)
                )
                candidates.append((cost, int(device_idx)))
            actions[service_idx] = min(candidates)[1]

        switches = 0
        for service_idx, parent_indices in enumerate(parents):
            cur = int(actions[service_idx].item())
            if not parent_indices:
                switches += 0 if cur == source_idx else 1
            else:
                switches += sum(1 for parent in parent_indices if int(actions[parent].item()) != cur)

        cloud_fraction = float((actions == cloud_idx).float().mean().item()) if actions.numel() else 0.0
        return actions, {
            "switches": int(switches),
            "correction_cnt": 0,
            "correction_cost": 0.0,
            "aux_cost": float(self.offloading_agent_params.get("penalty_switch", 0.0)) * float(switches),
            "cloud_fraction": cloud_fraction,
            "raw_actions": actions.detach().clone(),
        }

    def get_heuristic_offloading_plan(self, default_offloading=None) -> dict:
        if self.logical_topology is None or self.physical_topology is None or self.state_buffer is None:
            return copy.deepcopy(default_offloading) if isinstance(default_offloading, dict) else {}
        try:
            logic_edge_index = self._build_edge_index(self.logical_topology.links)
            logic_feats, phys_feats, _, _ = self._collect_offloading_state()
            actions, _ = self.heuristic_offloading_actions(
                logic_edge_index=logic_edge_index,
                logic_feats=logic_feats,
                phys_feats=phys_feats,
                static_mask=self._current_deploy_mask(),
            )
            return self._map_offloading_mask_to_offloading_plan(actions)
        except Exception as exc:
            LOGGER.warning(f"[HedgerAblation][OffloadingHeuristic] Fall back to default offloading: {exc}")
            return copy.deepcopy(default_offloading) if isinstance(default_offloading, dict) else {}


class HedgerDeploymentAblation(HedgerHeuristicMixin, Hedger):
    """Train Hedger deployment while replacing offloading PPO with a heuristic."""

    def _current_offloading_rollout_agent(self):
        return None

    def inference_hedger(self):
        assert self.logical_topology is not None and self.physical_topology is not None and self.state_buffer is not None
        if self.checkpoint_cfg.load.enabled and self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "[HedgerDeploymentAblation][Inference] Checkpoint loading was enabled but no checkpoint was loaded."
            )
        if not self.checkpoint_cfg.load.enabled:
            raise RuntimeError(
                "[HedgerDeploymentAblation][Inference] checkpoint.load.enabled must be true because "
                "the deployment policy is learned."
            )

        LOGGER.info(
            f"[HedgerDeploymentAblation][Inference] Start: "
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
                self.deployment_plan = copy.deepcopy(self.initial_deployment_plan)
                self.cur_deploy_mask = self._map_deployment_plan_to_deployment_mask(
                    self.initial_deployment_plan
                ).detach().cpu()
            else:
                self.cur_deploy_mask = torch.zeros(
                    (len(self.logical_topology), len(self.physical_topology)),
                    dtype=torch.bool,
                )
                self.cur_deploy_mask[:, self.physical_topology.cloud_idx] = True
                self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self.cur_deploy_mask)

        worker = threading.Thread(target=self.inference_deployment_agent, daemon=True)
        worker.start()
        while not self.deployment_thread_stop_event.is_set():
            if not worker.is_alive():
                LOGGER.warning("[HedgerDeploymentAblation][Inference] Deployment worker stopped unexpectedly.")
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
                        "latency_normalizer", "latency_clip", "slo_violation", "cloud_fraction",
                        "aux_cost", "off_latency_weight", "off_slo_weight", "off_cloud_weight",
                        "switch_cnt", "correction_cnt",
                        "correction_cost", "policy_logp", "policy_entropy", "value_estimate",
                        "next_value", "raw_cloud_fraction", "executed_cloud_fraction",
                        "unique_targets", "feasible_targets_mean", "feasible_targets_min",
                        "feasible_targets_max", "transition_buffer", "raw_offloading_plan",
                        "offloading_plan", *self._state_record_fieldnames(),
                        "feedback_recorded", "feedback_deployment_version",
                        "feedback_task_observations", "feedback_deployment_versions",
                        "switch_weight", "correction_weight"],
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

                latency_cost = self._compute_offloading_latency_cost(metrics["latency"])
                reward = self._compute_offloading_reward(metrics, aux)
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
                    latency_normalizer=self.offloading_agent_params["reward_off_latency_normalizer"],
                    latency_clip=self.offloading_agent_params["reward_off_latency_clip"],
                    slo_violation=metrics["slo_violation"],
                    cloud_fraction=metrics["cloud_fraction"],
                    aux_cost=aux["aux_cost"],
                    off_latency_weight=self.offloading_agent_params["reward_off_latency_weight"],
                    off_slo_weight=self.offloading_agent_params["reward_off_slo_weight"],
                    off_cloud_weight=self.offloading_agent_params["reward_off_cloud_weight"],
                    switch_cnt=aux["switches"],
                    correction_cnt=0,
                    correction_cost=0.0,
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
                    switch_weight=self.offloading_agent_params["penalty_switch"],
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


class HedgerOffloadingAblation(HedgerHeuristicMixin, Hedger):
    """Train Hedger offloading while replacing deployment PPO with a heuristic."""

    def get_initial_deployment_plan(self):
        return self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=False)

    def get_redeployment_plan(self):
        return self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=True)

    def inference_hedger(self):
        assert self.logical_topology is not None and self.physical_topology is not None and self.state_buffer is not None
        if self.checkpoint_cfg.load.enabled and self._loaded_checkpoint_path is None:
            raise RuntimeError(
                "[HedgerOffloadingAblation][Inference] Checkpoint loading was enabled but no checkpoint was loaded."
            )
        if not self.checkpoint_cfg.load.enabled:
            raise RuntimeError(
                "[HedgerOffloadingAblation][Inference] checkpoint.load.enabled must be true because "
                "the offloading policy is learned."
            )

        LOGGER.info(
            f"[HedgerOffloadingAblation][Inference] Start: "
            f"{self._summarize_runtime_config()}, {self._summarize_topology()}"
        )
        self.set_seed()
        self.shared_topology_encoder.eval()
        self.offloading_agent.eval()
        self.deployment_thread_stop_event.clear()
        self.offloading_thread_stop_event.clear()

        if self.cur_deploy_mask is None:
            self.set_heuristic_deployment_plan(default_deployment=self.initial_deployment_plan, mark_version=False)
        elif self.deployment_plan is None:
            self.deployment_plan = self._map_deployment_mask_to_deployment_plan(self._current_deploy_mask())

        worker = threading.Thread(target=self.inference_offloading_agent, daemon=True)
        worker.start()
        while not self.offloading_thread_stop_event.is_set():
            if not worker.is_alive():
                LOGGER.warning("[HedgerOffloadingAblation][Inference] Offloading worker stopped unexpectedly.")
                break
            time.sleep(0.5)
        self.deployment_thread_stop_event.set()
        self.offloading_thread_stop_event.set()


class HedgerFlatAblation(HedgerHeuristicMixin, Hedger):
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
            source_node_idx=self.physical_topology.source_idx if self.physical_topology is not None else 0,
            cloud_node_idx=self.physical_topology.cloud_idx if self.physical_topology is not None else -1,
            deployment_constraint_cfg=from_partial_dict(DeploymentConstraintCfg, self.deployment_agent_params),
            offloading_constraint_cfg=from_partial_dict(OffloadingConstraintCfg, self.offloading_agent_params),
        ).to(self.device)
        self.deployment_agent = self.flat_agent
        self.offloading_agent = self.flat_agent
        self._sync_agent_topology_bindings()

    def _sync_agent_topology_bindings(self):
        super()._sync_agent_topology_bindings()
        flat_agent = getattr(self, "flat_agent", None)
        if flat_agent is not None and self.physical_topology is not None:
            flat_agent.source = self.physical_topology.source_idx
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
                self.shared_topology_encoder.load_state_dict(ckpt['encoder'])
            flat_state = ckpt.get('flat_agent')
            if flat_state is not None:
                self._load_state_dict_compatible(self.flat_agent, flat_state, "flat_agent")
            else:
                LOGGER.warning("[HedgerFlat][Checkpoint] Missing flat_agent state.")
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
        logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
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
                logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
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
                        "switch_cnt", "correction_cnt", "correction_cost", "policy_logp",
                        "policy_entropy", "value_estimate", "next_value", "transition_buffer",
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
        logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
        last_task_version = self.state_buffer.get_task_observation_version() if self.state_buffer is not None else 0
        while not self.deployment_thread_stop_event.is_set():
            try:
                if self._sleep_while_latency_guard_active("flat worker"):
                    prev_deploy_mask = self._current_deploy_mask()
                    logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=prev_deploy_mask)
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
                    logic_feats, phys_feats, _, _ = self._collect_deployment_state(prev_deploy_mask=deploy_mask.detach().cpu())
                    continue
                feedback_version = decision_version if task_feedback_summary.get("count", 0) > 0 else None
                new_logic_feats, new_phys_feats, metrics, done = self._collect_offloading_state(
                    since_task_version=last_task_version,
                    deployment_version=feedback_version,
                )
                last_task_version = current_task_version
                latency_cost = self._compute_offloading_latency_cost(metrics["latency"])
                off_reward = self._compute_offloading_reward(metrics, aux)

                cloud_idx = self.physical_topology.cloud_idx
                exec_deploy_mask = deploy_mask.detach().cpu().bool()
                cloud_only = int((~exec_deploy_mask[:, :cloud_idx].any(dim=1)).sum().item()) if cloud_idx > 0 else 0
                aux["cloud_only_ratio"] = float(cloud_only) / float(max(1, exec_deploy_mask.size(0)))
                if cloud_idx > 0:
                    empty = int((~exec_deploy_mask[:, :cloud_idx].any(dim=0)).sum().item())
                    aux["empty_edge_device_ratio"] = float(empty) / float(cloud_idx)
                else:
                    aux["empty_edge_device_ratio"] = 0.0
                dep_metrics = {
                    "avg_offloading_reward": off_reward,
                    "deploy_change_cost": self._compute_deploy_change_cost(prev_deploy_mask),
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
                    "raw_actions": aux["raw_actions"].detach().cpu(),
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
                    switch_cnt=aux["switches"],
                    correction_cnt=aux["correction_cnt"],
                    correction_cost=aux["correction_cost"],
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
                    raw_actions=aux["raw_actions"].detach().cpu(),
                    executed_actions=actions.detach().cpu(),
                    static_mask=deploy_mask.detach().cpu(),
                    logic_feats=state_logic_feats,
                    phys_feats=state_phys_feats,
                )
                step += 1
            except Exception as exc:
                LOGGER.exception(f"[HedgerFlat][Train] Worker loop error: {exc}")
                time.sleep(0.5)
