from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeploymentActor(nn.Module):
    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.temperature = temperature

    def forward(self, h_s: torch.Tensor, h_p: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q = self.q(h_s)
        K = self.k(h_p)
        scores = (Q @ K.t()) / math.sqrt(Q.size(-1))
        scores = scores / max(self.temperature, 1e-6)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return torch.sigmoid(scores)  # Multilabel Bernoulli probability


class OffloadActor(nn.Module):
    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.temperature = temperature

    def forward(self, h_s: torch.Tensor, h_p: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q = self.q(h_s)
        K = self.k(h_p)
        scores = (Q @ K.t()) / math.sqrt(Q.size(-1))
        scores = scores / max(self.temperature, 1e-6)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)  # One Categorical per service


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

    def forward(self, h_s: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        vs = h_s.mean(dim=0, keepdim=True)
        vp = h_p.mean(dim=0, keepdim=True)
        return self.mlp(torch.cat([vs, vp], dim=-1)).squeeze(-1)  # [1]
