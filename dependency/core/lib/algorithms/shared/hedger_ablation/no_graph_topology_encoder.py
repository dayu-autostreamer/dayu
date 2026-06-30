from typing import Dict

import torch
import torch.nn as nn

__all__ = ("NoGraphTopologyEncoders",)


class NoGraphTopologyEncoders(nn.Module):
    """
    No-topology-encoder ablation.

    The full Hedger policy receives service/device embeddings from the shared
    topology encoder.  This ablation removes that learned representation path
    while preserving tensor shapes expected by the deployment/offloading PPO
    heads.  The downstream agents can still use their explicit candidate and
    runtime features, but they no longer receive learned topology embeddings.
    """

    def __init__(self, d_model: int = 64, num_roles: int = 2, role_emb_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)

    @staticmethod
    def _first_tensor(feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        for value in feats.values():
            if isinstance(value, torch.Tensor):
                return value
        return torch.empty(0)

    @staticmethod
    def _node_count(feats: Dict[str, torch.Tensor], preferred_keys) -> int:
        for key in preferred_keys:
            value = feats.get(key)
            if isinstance(value, torch.Tensor) and value.dim() >= 1:
                return int(value.size(0))
        return 0

    def encode(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats):
        logic_ref = self._first_tensor(logic_feats)
        phys_ref = self._first_tensor(phys_feats)
        service_count = self._node_count(
            logic_feats,
            ("service_demand_feat", "model_flops", "model_mem", "task_complexity_seq"),
        )
        device_count = self._node_count(
            phys_feats,
            ("device_capability_feat", "gpu_flops", "mem_capacity", "role_id"),
        )
        service_embeddings = torch.zeros(
            service_count,
            self.d_model,
            device=logic_ref.device,
            dtype=torch.float32,
        )
        device_embeddings = torch.zeros(
            device_count,
            self.d_model,
            device=phys_ref.device,
            dtype=torch.float32,
        )
        return service_embeddings, device_embeddings
