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
        self.temporal = NodeTemporalEncoder(d_dyn=1, d_model=d_model)
        self.fusion = NodeFeatureFusionBlock(d_model=d_model, dropout=dropout)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        mf = safe_log1p(feats["model_flops"].float()).unsqueeze(-1)
        mm = safe_log1p(feats["model_mem"].float()).unsqueeze(-1)
        h_static = self.lin_static(torch.cat([mf, mm], dim=-1))

        tc = feats["task_complexity_seq"].float()
        h_dyn = self.temporal(tc.unsqueeze(-1))

        return self.fusion(F.relu(h_static + h_dyn))


class NoGraphPhysicalEncoder(nn.Module):
    """Physical encoder ablation without device-topology message passing."""

    def __init__(self, d_model: int = 64, num_roles: int = 2, role_emb_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        static_in = 2 + role_emb_dim
        dyn_in = 3
        self.lin_static = nn.Linear(static_in, d_model)
        self.temporal = NodeTemporalEncoder(d_dyn=dyn_in, d_model=d_model)
        self.fusion = NodeFeatureFusionBlock(d_model=d_model, dropout=dropout)

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        gf = safe_log1p(feats["gpu_flops"].float()).unsqueeze(-1)
        mc = safe_log1p(feats["mem_capacity"].float()).unsqueeze(-1)
        role_vec = self.role_emb(feats["role_id"].long())
        h_static = self.lin_static(torch.cat([gf, mc, role_vec], dim=-1))

        bw = safe_log1p(feats["bandwidth_seq"].float())
        gu = feats["gpu_util_seq"].float()
        mu = feats["mem_util_seq"].float()
        h_dyn = self.temporal(torch.stack([bw, gu, mu], dim=-1))

        return self.fusion(F.relu(h_static + h_dyn))


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
