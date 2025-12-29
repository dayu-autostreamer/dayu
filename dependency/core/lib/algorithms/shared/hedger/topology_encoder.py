from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from .utils import safe_log1p, graph_in_out_degree, topo_levels_dag

__all__ = ('TopologyEncoders',)


class TopologyEncoders(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4, num_roles: int = 3, role_emb_dim: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.logic = LogicalEncoder(d_model=d_model, heads=heads, dropout=dropout)
        self.physical = PhysicalEncoder(d_model=d_model, num_roles=num_roles, role_emb_dim=role_emb_dim,
                                        dropout=dropout)

    def encode(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats):
        return self.logic(logic_edge_index, logic_feats), self.physical(phys_edge_index, phys_feats)


class LogicalEncoder(nn.Module):
    """
    feats:
      - model_flops: [Ms]
      - model_mem:   [Ms]   (MB)
      - task_complexity_seq: [Ms,T]
      - hist_latency_seq:    [Ms,T]
    """

    def __init__(self, d_model: int = 64, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.lin_static = nn.Linear(2, d_model)  # [log(model_flops), log(model_mem)]
        self.temporal = NodeTemporalEncoder(d_dyn=2, d_model=d_model)
        self.lin_struct = nn.Linear(3, d_model)

        self.gnn1 = GATConv(d_model, d_model, heads=heads, concat=True, dropout=dropout)
        self.gnn2 = GATConv(d_model * heads, d_model, heads=1, concat=False, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model * heads)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, edge_index: torch.Tensor, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        Ms = feats["model_flops"].shape[0]
        mf = safe_log1p(feats["model_flops"].float()).unsqueeze(-1)
        mm = safe_log1p(feats["model_mem"].float()).unsqueeze(-1)
        h_static = self.lin_static(torch.cat([mf, mm], dim=-1))

        tc = feats["task_complexity_seq"].float()
        hl = safe_log1p(feats["hist_latency_seq"].float())
        h_dyn = self.temporal(torch.stack([tc, hl], dim=-1))

        in_deg, out_deg = graph_in_out_degree(edge_index, Ms)
        in_deg = in_deg / (in_deg.max() + 1e-6)
        out_deg = out_deg / (out_deg.max() + 1e-6)
        level = topo_levels_dag(edge_index, Ms)
        h_struct = self.lin_struct(torch.stack([in_deg, out_deg, level], dim=-1))

        x0 = F.relu(h_static + h_dyn + h_struct)
        h1 = self.gnn1(x0, edge_index)
        h1 = self.norm1(F.elu(h1))
        h2 = self.gnn2(h1, edge_index)
        h2 = self.norm2(h2)
        return h2


class PhysicalEncoder(nn.Module):
    """
    feats:
      - gpu_flops:     [Np]
      - role_id:       [Np] (0=source edge / 1=other edge  / 2=cloud)
      - mem_capacity:  [Np] (MB)
      - bandwidth_seq: [Np,T]
      - gpu_util_seq:  [Np,T]
      - mem_util_seq:  [Np,T] (0~1)
    """

    def __init__(self, d_model: int = 64, num_roles: int = 3, role_emb_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.role_emb = nn.Embedding(num_roles, role_emb_dim)
        static_in = 2 + role_emb_dim  # [log(gpu_flops), log(mem_capacity), role_emb]
        dyn_in = 3  # [log(bw), gpu_util, mem_util]
        self.lin_static = nn.Linear(static_in, d_model)
        self.temporal = NodeTemporalEncoder(d_dyn=dyn_in, d_model=d_model)
        self.gnn1 = GCNConv(d_model, d_model)
        self.gnn2 = GCNConv(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, edge_index: torch.Tensor, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        gf = safe_log1p(feats["gpu_flops"].float()).unsqueeze(-1)
        mc = safe_log1p(feats["mem_capacity"].float()).unsqueeze(-1)
        role_vec = self.role_emb(feats["role_id"].long())
        h_static = self.lin_static(torch.cat([gf, mc, role_vec], dim=-1))

        bw = safe_log1p(feats["bandwidth_seq"].float())
        gu = feats["gpu_util_seq"].float()
        mu = feats["mem_util_seq"].float()
        h_dyn = self.temporal(torch.stack([bw, gu, mu], dim=-1))

        x0 = F.relu(h_static + h_dyn)
        h1 = self.gnn1(x0, edge_index)
        h1 = self.norm1(F.relu(h1))
        h1 = self.dropout(h1)
        h2 = self.gnn2(h1, edge_index)
        h2 = self.norm2(h2)
        return h2


class NodeTemporalEncoder(nn.Module):
    def __init__(self, d_dyn: int, d_model: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(input_size=d_dyn, hidden_size=d_model, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _, hT = self.rnn(x_seq)  # [L,N,d]
        h = hT[-1]  # [N,d]
        return self.norm(h)
