from typing import Dict

import torch


def latest_seq_value(feats: Dict[str, torch.Tensor], key: str, idx: int, default: float = 0.0) -> float:
    value = feats.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return float(default)
    try:
        if value.dim() >= 2:
            return float(value[idx, -1].item())
        return float(value[idx].item())
    except Exception:
        return float(default)


__all__ = ("latest_seq_value",)
