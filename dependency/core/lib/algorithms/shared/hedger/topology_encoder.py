from typing import Dict

import torch
import torch.nn as nn

__all__ = ("TopologyEncoders",)


class _FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class TopologyEncoders(nn.Module):
    """
    Hedger's shared local encoders.

    The current policy intentionally keeps capability matching independent from
    graph message passing. Each service and each device is encoded by the same
    MLP, so the policy remains size-agnostic without smoothing away edge-device
    capability differences.
    """

    service_input_dim = 4
    device_input_dim = 5

    def __init__(self, d_model: int = 64, dropout: float = 0.0):
        super().__init__()
        self.service = _FeatureEncoder(self.service_input_dim, d_model, dropout=dropout)
        self.device = _FeatureEncoder(self.device_input_dim, d_model, dropout=dropout)

    def encode(
            self,
            logic_edge_index,
            logic_feats: Dict[str, torch.Tensor],
            phys_edge_index,
            phys_feats: Dict[str, torch.Tensor],
    ):
        service_state = logic_feats.get("service_demand_feat")
        device_state = phys_feats.get("device_capability_feat")
        if not isinstance(service_state, torch.Tensor) or service_state.dim() != 2 \
                or service_state.size(-1) != self.service_input_dim:
            raise ValueError("Hedger state is missing `service_demand_feat` with shape [S, 4].")
        if not isinstance(device_state, torch.Tensor) or device_state.dim() != 2 \
                or device_state.size(-1) != self.device_input_dim:
            raise ValueError("Hedger state is missing `device_capability_feat` with shape [D, 5].")
        return self.service(service_state), self.device(device_state)
