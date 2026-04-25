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

    def forward(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pair_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self.q(h_s)
        K = self.k(h_p)
        scores = (Q @ K.t()) / math.sqrt(Q.size(-1))
        scores = scores / max(self.temperature, 1e-6)
        if pair_bias is not None:
            scores = scores + pair_bias
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return torch.sigmoid(scores)  # Element-wise probability for multilabel Bernoulli sampling


class OffloadActor(nn.Module):
    def __init__(self, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.temperature = temperature

    def forward(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            pair_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self.q(h_s)
        K = self.k(h_p)
        scores = (Q @ K.t()) / math.sqrt(Q.size(-1))
        scores = scores / max(self.temperature, 1e-6)
        if pair_bias is not None:
            scores = scores + pair_bias
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)  # One categorical distribution per service


class ValueHead(nn.Module):
    def __init__(self, d_model: int, context_dim: Optional[int] = None):
        super().__init__()
        self.context_dim = d_model if context_dim is None else int(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model + self.context_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vs = h_s.mean(dim=0, keepdim=True)
        vp = h_p.mean(dim=0, keepdim=True)
        if context is None:
            context = torch.zeros((1, self.context_dim), device=vs.device, dtype=vs.dtype)
        elif context.dim() == 1:
            context = context.unsqueeze(0)
        return self.mlp(torch.cat([vs, vp, context], dim=-1)).squeeze(-1)  # [1]


class FeatureAdapter(nn.Module):
    """
    Lightweight task-specific adapter:

    - LayerNorm stabilizes feature distributions
    - A small MLP adds task-specific nonlinearity
    - The residual connection keeps features aligned with the shared backbone
    """
    def __init__(self, d_model: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden_dim = max(1, int(d_model * hidden_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # `x` has shape `[..., d_model]`.
        return x + self.net(x)


class PairBiasHead(nn.Module):
    """
    Lightweight projector from explicit pair features to one additive bias.

    The shared topology encoder still learns service and device embeddings.
    This head only turns explicit service-device statistics into a small score
    correction that is added on top of the dot-product actor logits.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        if pair_features.numel() == 0:
            return torch.zeros(pair_features.shape[:-1], device=pair_features.device, dtype=pair_features.dtype)
        return self.net(pair_features).squeeze(-1)
