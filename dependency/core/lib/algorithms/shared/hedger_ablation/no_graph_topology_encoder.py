from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lib.algorithms.shared.hedger.topology_encoder import NodeTemporalEncoder
from core.lib.algorithms.shared.hedger.utils import safe_log1p

__all__ = ("NoGraphTopologyEncoders",)


class NodeFeatureFusionBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class NoGraphLogicalEncoder(nn.Module):
    """Logical encoder ablation without DAG message passing."""

    def __init__(self, d_model: int = 64, dropout: float = 0.0):
        super().__init__()
        self.lin_static = nn.Linear(2, d_model)
        self.temporal = NodeTemporalEncoder(d_dyn=2, d_model=d_model)
        self.fusion = NodeFeatureFusionBlock(d_model=d_model, dropout=dropout)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        mf = safe_log1p(feats["model_flops"].float()).unsqueeze(-1)
        mm = safe_log1p(feats["model_mem"].float()).unsqueeze(-1)
        h_static = self.lin_static(torch.cat([mf, mm], dim=-1))

        tc = safe_log1p(feats["task_complexity_seq"].float())
        ar = feats["task_arrival_rate_seq"].float()
        h_dyn = self.temporal(torch.stack([tc, ar], dim=-1))

        return self.fusion(F.relu(h_static + h_dyn))


class NoGraphPhysicalEncoder(nn.Module):
    """Physical encoder ablation without device-topology message passing."""

    def __init__(self, d_model: int = 64, num_roles: int = 2, role_emb_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        static_in = 4 + role_emb_dim
        self.lin_static = nn.Linear(static_in, d_model)
        self.fusion = NodeFeatureFusionBlock(d_model=d_model, dropout=dropout)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        gf = safe_log1p(feats["gpu_flops"].float()).unsqueeze(-1)
        mc = safe_log1p(feats["mem_capacity"].float()).unsqueeze(-1)
        role_id = feats["role_id"].long()
        role_vec = self.role_emb(role_id)
        is_cloud = (role_id == 1).float().unsqueeze(-1)
        bw = safe_log1p(feats["bandwidth_latest"].float()).unsqueeze(-1)
        h_static = self.lin_static(torch.cat([gf, mc, bw, is_cloud, role_vec], dim=-1))

        return self.fusion(F.relu(h_static))


class NoGraphTopologyEncoders(nn.Module):
    """Shared no-graph topology encoder used by the no-graph ablation."""

    def __init__(self, d_model: int = 64, num_roles: int = 2, role_emb_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.logic = NoGraphLogicalEncoder(d_model=d_model, dropout=dropout)
        self.physical = NoGraphPhysicalEncoder(
            d_model=d_model,
            num_roles=num_roles,
            role_emb_dim=role_emb_dim,
            dropout=dropout,
        )

    def encode(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats):
        return self.logic(logic_feats), self.physical(phys_feats)
