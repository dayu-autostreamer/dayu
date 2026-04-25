import copy
import math
from typing import Dict, Optional

import torch

from core.lib.common import LOGGER
from core.lib.algorithms.shared.hedger_ablation.utils import latest_seq_value

__all__ = ("HedgerHeuristicDeploymentMixin",)


class HedgerHeuristicDeploymentMixin:
    def _edge_count_target(self) -> int:
        return max(1, int(self.deployment_agent_params.get("min_edge_replicas_per_service", 1) or 1))

    def _max_edge_replicas_per_device(self) -> Optional[int]:
        value = self.deployment_agent_params.get("max_edge_replicas_per_device")
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

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
                -latest_seq_value(logic_feats, "task_complexity_seq", s_idx, 0.0),
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
                gpu_util = latest_seq_value(phys_feats, "gpu_util_seq", device_idx, 0.0)
                mem_util = latest_seq_value(phys_feats, "mem_util_seq", device_idx, 0.0)
                bandwidth = max(1.0, latest_seq_value(phys_feats, "bandwidth_seq", device_idx, 1.0))
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
