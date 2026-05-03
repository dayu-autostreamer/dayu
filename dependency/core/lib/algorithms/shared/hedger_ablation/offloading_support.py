import copy
import math
from typing import Dict, List, Tuple

import torch

from core.lib.common import LOGGER
from core.lib.algorithms.shared.hedger_ablation.utils import latest_seq_value

__all__ = ("HedgerHeuristicOffloadingMixin",)


class HedgerHeuristicOffloadingMixin:
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

    def heuristic_offloading_actions(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_services, _ = static_mask.size()
        cloud_idx = self.physical_topology.cloud_idx
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

            candidates = []
            for device_idx in allowed:
                gpu_util = latest_seq_value(phys_feats, "gpu_util_seq", device_idx, 0.0)
                mem_util = latest_seq_value(phys_feats, "mem_util_seq", device_idx, 0.0)
                bandwidth = max(1.0, latest_seq_value(phys_feats, "bandwidth_seq", device_idx, 1.0))
                cloud_penalty = 0.6 if device_idx == cloud_idx else 0.0
                cost = (
                    0.35 * gpu_util
                    + 0.35 * mem_util
                    + cloud_penalty
                    - 0.05 * math.log1p(bandwidth)
                )
                candidates.append((cost, int(device_idx)))
            actions[service_idx] = min(candidates)[1]

        cloud_fraction = float((actions == cloud_idx).float().mean().item()) if actions.numel() else 0.0
        return actions, {
            "cloud_fraction": cloud_fraction,
        }

    def get_heuristic_offloading_plan(self, default_offloading=None) -> dict:
        if self.logical_topology is None or self.physical_topology is None or self.state_buffer is None:
            return copy.deepcopy(default_offloading) if isinstance(default_offloading, dict) else {}
        try:
            logic_edge_index = self._build_edge_index(self.logical_topology.links)
            logic_feats, phys_feats, _, _, _ = self._collect_offloading_state()
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
