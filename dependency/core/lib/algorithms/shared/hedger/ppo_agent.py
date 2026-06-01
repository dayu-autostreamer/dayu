from typing import Optional, List, Dict, Tuple, Sequence, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_encoder import TopologyEncoders
from .ppo_network import DeploymentActor, OffloadActor
from .hedger_config import DeploymentConstraintCfg
from .deployment_dataset import transition_quality_bucket
from .utils import compute_returns_advantages


def _move_tensor_dict_to_device(feats: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in feats.items()
    }


def _scalar_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std_or_zero(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = _mean_or_zero(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return float(math.sqrt(max(var, 0.0)))


def _tensor_std_float(values: torch.Tensor) -> float:
    values = values.detach().float()
    if values.numel() <= 1:
        return 0.0
    return float(values.std(unbiased=False).cpu().item())


def _parameters_grad_norm(parameters) -> float:
    total_sq_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        param_norm = float(param.grad.detach().data.norm(2).cpu().item())
        total_sq_norm += param_norm * param_norm
    return float(math.sqrt(total_sq_norm))


def _feature_vector(
        feats: Dict[str, torch.Tensor],
        key: str,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    value = feats.get(key)
    if not isinstance(value, torch.Tensor) or value.dim() != 1 or value.size(0) != length:
        return torch.zeros((length,), device=device, dtype=dtype)
    return value.to(device=device, dtype=dtype)


def _debug_tensor(feats: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    value = feats.get(key)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.empty(0)


SERVICE_DEMAND_FEATURE_NAMES = [
    "log_compute_demand",
    "complexity_zscore",
    "arrival_rate_short",
    "log_model_mem",
]

DEVICE_CAPABILITY_FEATURE_NAMES = [
    "log_effective_gpu_flops",
    "log_mem_capacity",
    "relative_edge_compute",
    "is_cloud",
    "cloud_bandwidth_penalty",
]

RUNTIME_PAIR_FEATURE_NAMES = [
    "queue_short",
    "queue_busy",
    "real_time_per_complexity",
    "runtime_confidence",
    "runtime_recency",
    "queue_freshness",
]

OFFLOADING_CANDIDATE_FEATURE_NAMES = [
    "qk_feature",
    "compute_gap",
    "arrival_rate_short",
    "runtime_ratio",
    "runtime_confidence",
    "runtime_recency",
    "queue_freshness",
    "speed_evidence",
    "capacity_pressure",
    "pair_load",
    "device_load",
    "service_time_factor",
    "parent_same_device",
    "cross_tier_penalty",
    "is_cloud",
]

OFFLOADING_STATIC_PRIOR_FEATURE_NAMES = [
    "qk_feature",
    "compute_gap",
    "parent_same_device",
]

DEPLOYMENT_CANDIDATE_FEATURE_NAMES = [
    "qk_feature",
    "service_pressure",
    "dependency_criticality",
    "log_edge_feasible_count",
    "log_edge_replica_count",
    "best_pair_quality",
    "quality_gap_top_second",
    "best_runtime_risk",
    "max_queue_pressure",
    "relative_edge_compute",
    "log_effective_gpu_flops",
    "log_mem_capacity",
    "static_quality",
    "pair_quality",
    "runtime_risk",
    "queue_pressure",
    "runtime_confidence",
    "runtime_relative_weakness",
    "low_quality_gap",
    "pair_memory_cost",
    "current_replica",
    "device_replica_count",
    "static_allowed",
    "active_pair_hotspot",
]

DEPLOYMENT_SERVICE_CONTEXT_FEATURE_NAMES = [
    "log_compute_demand",
    "complexity_zscore",
    "arrival_rate_short",
    "log_model_mem",
    "service_pressure",
    "dependency_criticality",
    "edge_feasible_count",
    "edge_replica_count",
    "best_pair_quality",
    "second_pair_quality",
    "quality_gap_top_second",
    "best_runtime_risk",
    "max_queue_pressure",
]


def _service_demand_features(
        logic_feats: Dict[str, torch.Tensor],
        num_services: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    existing = logic_feats.get("service_demand_feat")
    if isinstance(existing, torch.Tensor) and existing.dim() == 2 \
            and existing.size(0) == num_services and existing.size(1) >= 4:
        return existing[:, :4].to(device=device, dtype=dtype)

    tc = logic_feats.get("task_complexity_seq")
    ar = logic_feats.get("task_arrival_rate_seq")
    mf = logic_feats.get("model_flops")
    mm = logic_feats.get("model_mem")
    if not isinstance(tc, torch.Tensor) or tc.dim() != 2 or tc.size(0) != num_services:
        raise ValueError("Hedger state is missing `task_complexity_seq`.")
    if not isinstance(ar, torch.Tensor) or ar.dim() != 2 or ar.size(0) != num_services:
        raise ValueError("Hedger state is missing `task_arrival_rate_seq`.")
    if not isinstance(mf, torch.Tensor) or mf.dim() != 1 or mf.size(0) != num_services:
        raise ValueError("Hedger state is missing `model_flops`.")
    if not isinstance(mm, torch.Tensor) or mm.dim() != 1 or mm.size(0) != num_services:
        raise ValueError("Hedger state is missing `model_mem`.")
    tc = tc.to(device=device, dtype=dtype)
    ar = ar.to(device=device, dtype=dtype)
    mf = mf.to(device=device, dtype=dtype)
    mm = mm.to(device=device, dtype=dtype)

    current = tc[:, -1]
    mean = tc.mean(dim=1)
    std = tc.std(dim=1, unbiased=False)
    log_compute_demand = torch.log1p(current.clamp_min(0.0) * mf.clamp_min(0.0))
    complexity_zscore = (current - mean) / (std + 1e-6)
    arrival_rate = ar[:, -1].clamp_min(0.0)
    log_model_mem = torch.log1p(mm.clamp_min(0.0))
    return torch.stack([log_compute_demand, complexity_zscore, arrival_rate, log_model_mem], dim=-1)


def _device_capability_features(
        phys_feats: Dict[str, torch.Tensor],
        num_devices: int,
        device: torch.device,
        dtype: torch.dtype,
        cloud_idx: int,
) -> torch.Tensor:
    existing = phys_feats.get("device_capability_feat")
    if isinstance(existing, torch.Tensor) and existing.dim() == 2 \
            and existing.size(0) == num_devices and existing.size(1) >= 5:
        return existing[:, :5].to(device=device, dtype=dtype)

    gpu_flops = _feature_vector(phys_feats, "gpu_flops", num_devices, device, dtype)
    mem_capacity = _feature_vector(phys_feats, "mem_capacity", num_devices, device, dtype)
    bandwidth = _feature_vector(phys_feats, "bandwidth_latest", num_devices, device, dtype).clamp_min(1e-6)
    _, is_cloud = _role_tensors(phys_feats, num_devices, device, dtype, cloud_idx)
    edge_mask = is_cloud < 0.5

    log_gpu = torch.log1p(gpu_flops.clamp_min(0.0))
    log_mem = torch.log1p(mem_capacity.clamp_min(0.0))
    if edge_mask.any():
        edge_compute_mean = log_gpu[edge_mask].mean()
        edge_bw_ref = bandwidth[edge_mask].mean().clamp_min(1e-6)
    else:
        edge_compute_mean = log_gpu.mean()
        edge_bw_ref = bandwidth.mean().clamp_min(1e-6)
    relative_edge_compute = torch.where(edge_mask, log_gpu - edge_compute_mean, torch.zeros_like(log_gpu))
    cloud_bandwidth_penalty = is_cloud * torch.log1p(edge_bw_ref / bandwidth)
    return torch.stack([log_gpu, log_mem, relative_edge_compute, is_cloud, cloud_bandwidth_penalty], dim=-1)


def _runtime_pair_features(
        logic_feats: Dict[str, torch.Tensor],
        num_services: int,
        num_devices: int,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build the shared service-device runtime state:

    [queue_short, queue_busy, real_time_per_complexity,
     runtime_confidence, runtime_recency, queue_freshness].
    """
    existing = logic_feats.get("runtime_pair_feat")
    if isinstance(existing, torch.Tensor) and existing.dim() == 3 \
            and existing.size(0) == num_services and existing.size(1) == num_devices \
            and existing.size(2) >= len(RUNTIME_PAIR_FEATURE_NAMES):
        return existing[..., :len(RUNTIME_PAIR_FEATURE_NAMES)].to(device=device, dtype=dtype)
    raise ValueError("Hedger state is missing `runtime_pair_feat`; collect state with the current runtime builder.")


def _masked_standardize(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    mask_f = mask.float()
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (scores * mask_f).sum(dim=1, keepdim=True) / counts
    var = (((scores - mean) * mask_f) ** 2).sum(dim=1, keepdim=True) / counts
    standardized = torch.clamp((scores - mean) / torch.sqrt(var + 1e-6), min=-5.0, max=5.0)
    return standardized * mask_f


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = True) -> torch.Tensor:
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    denom = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (values * mask_f).sum(dim=dim, keepdim=keepdim) / denom


def _unit_standardize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if values.numel() <= 1:
        return torch.full_like(values, 0.5)
    mean = values.mean()
    std = values.std(unbiased=False)
    return torch.sigmoid((values - mean) / (std + 1e-6))


def _safe_logit(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    return torch.log(probs) - torch.log1p(-probs)


def _role_tensors(
        phys_feats: Dict[str, torch.Tensor],
        num_devices: int,
        device: torch.device,
        dtype: torch.dtype,
        cloud_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    role_id = phys_feats.get("role_id")
    if isinstance(role_id, torch.Tensor) and role_id.dim() == 1 and role_id.size(0) == num_devices:
        role_id = role_id.to(device=device, dtype=torch.long).clone()
        is_cloud = (role_id == 1).to(dtype=dtype)
    else:
        role_id = torch.zeros((num_devices,), device=device, dtype=torch.long)
        is_cloud = torch.zeros((num_devices,), device=device, dtype=dtype)
    if 0 <= cloud_idx < num_devices:
        is_cloud[cloud_idx] = 1.0
        role_id[cloud_idx] = 1
    return role_id, is_cloud


def _topology_context(edge_index: torch.Tensor, num_nodes: int) -> Tuple[List[List[int]], List[List[int]], torch.Tensor]:
    row, col = edge_index
    parents = [[] for _ in range(num_nodes)]
    children = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        parents[v].append(u)
        children[u].append(v)

    levels = [0 for _ in range(num_nodes)]
    indeg = [len(parents[idx]) for idx in range(num_nodes)]
    queue = [idx for idx, deg in enumerate(indeg) if deg == 0]
    while queue:
        node = queue.pop(0)
        for child in children[node]:
            levels[child] = max(levels[child], levels[node] + 1)
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    max_level = max(levels) if levels else 0
    denom = float(max(max_level, 1))
    level_tensor = torch.tensor([level / denom for level in levels], device=edge_index.device, dtype=torch.float32)
    return parents, children, level_tensor


class CandidateCostHead(nn.Module):
    """Small candidate scalar head used by deployment costs and offloading affinity."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        last_layer = self.net[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.numel() == 0:
            return torch.zeros(candidate_features.shape[:-1], device=candidate_features.device,
                               dtype=candidate_features.dtype)
        return self.net(candidate_features).squeeze(-1)


class DeploymentPairRankHead(nn.Module):
    """Predict the Bernoulli logit for each service-device edge replica."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        last_layer = self.net[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        if candidate_features.numel() == 0:
            return torch.zeros(
                candidate_features.shape[:-1],
                device=candidate_features.device,
                dtype=candidate_features.dtype,
            )
        return self.net(candidate_features.float()).squeeze(-1)


class CandidateValueHead(nn.Module):
    """Graph-size agnostic critic over pooled service, device, and candidate states."""
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        hidden_dim = max(16, d_model // 2)
        self.candidate_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
        )
        self.value = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            candidate_features: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        service_pool = h_s.mean(dim=0)
        device_pool = h_p.mean(dim=0)
        candidate_emb = self.candidate_encoder(candidate_features)
        if mask is None:
            candidate_pool = candidate_emb.mean(dim=(0, 1))
        else:
            mask_f = mask.to(device=candidate_emb.device, dtype=candidate_emb.dtype).unsqueeze(-1)
            denom = mask_f.sum().clamp_min(1.0)
            candidate_pool = (candidate_emb * mask_f).sum(dim=(0, 1)) / denom
        return self.value(torch.cat([service_pool, device_pool, candidate_pool], dim=-1).unsqueeze(0)).squeeze(-1)


class _DeploymentBackbonePPO(nn.Module):
    """
    Shared deployment-side backbone for Hedger-style PPO policies.

    This backbone centralizes deployment feasibility logic, adapters, critic
    context, and optimizer wiring so deployment-side policies can reuse the
    same constraint handling without duplicating core Hedger mechanics.
    """
    def __init__(self, encoder: TopologyEncoders, d_model=64, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lamda=0.95, clip_eps=0.2, update_encoder: bool = True, cloud_node_idx: int = -1,
                 constraint_cfg: DeploymentConstraintCfg = DeploymentConstraintCfg()):
        super().__init__()
        self.encoder = encoder
        self.actor = DeploymentActor(d_model)
        hidden_dim = max(32, d_model)
        self.deployment_pair_rank_head = DeploymentPairRankHead(
            input_dim=len(DEPLOYMENT_CANDIDATE_FEATURE_NAMES),
            hidden_dim=hidden_dim,
        )
        self.critic = CandidateValueHead(input_dim=len(DEPLOYMENT_CANDIDATE_FEATURE_NAMES), d_model=d_model)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.cloud_idx = cloud_node_idx
        self.cfg = constraint_cfg
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._update_encoder = update_encoder
        self.actor_opt = None
        self._actor_train_params = []
        self._rebuild_actor_optimizer()

    def _rebuild_actor_optimizer(self):
        params_actor = list(self.actor.parameters())
        params_actor.extend(list(self.deployment_pair_rank_head.parameters()))
        if self._update_encoder:
            params_actor.extend(list(self.encoder.parameters()))
        self.actor_opt = torch.optim.Adam(params_actor, lr=self._actor_lr)
        self._actor_train_params = params_actor
        if not hasattr(self, "critic_opt") or self.critic_opt is None:
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self._critic_lr)

    def _cloud_index(self, width: int) -> int:
        if width <= 0:
            return -1
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (width - 1)
        if cloud_idx < 0 or cloud_idx >= width:
            return width - 1
        return cloud_idx

    def _edge_memory_budget_ratio(self) -> float:
        ratio = float(getattr(self.cfg, "edge_memory_budget_ratio", 1.0))
        if not math.isfinite(ratio):
            ratio = 1.0
        return min(max(ratio, 0.0), 1.0)

    def _max_edge_replicas_per_device(self) -> Optional[int]:
        value = getattr(self.cfg, "max_edge_replicas_per_device", None)
        if value is None:
            return None
        value = int(value)
        return value if value > 0 else None

    def _queue_normalizer(self) -> float:
        value = float(getattr(self.cfg, "queue_normalizer", 8.0))
        if not math.isfinite(value):
            value = 8.0
        return max(value, 1e-6)

    def _negative_queue_threshold(self) -> float:
        value = float(getattr(self.cfg, "negative_queue_threshold", 0.65))
        if not math.isfinite(value):
            value = 0.65
        return min(max(value, 0.0), 1.0)

    def _negative_hotspot_threshold(self) -> float:
        value = float(getattr(self.cfg, "negative_hotspot_threshold", 0.08))
        if not math.isfinite(value):
            value = 0.08
        return min(max(value, 0.0), 1.0)

    def _negative_runtime_risk_threshold(self) -> float:
        value = float(getattr(self.cfg, "negative_runtime_risk_threshold", 0.50))
        if not math.isfinite(value):
            value = 0.50
        return min(max(value, 0.0), 1.0)

    def _negative_unknown_threshold(self) -> float:
        value = float(getattr(self.cfg, "negative_unknown_threshold", 0.50))
        if not math.isfinite(value):
            value = 0.50
        return min(max(value, 0.0), 1.0)

    def _negative_stale_threshold(self) -> float:
        value = float(getattr(self.cfg, "negative_stale_threshold", 0.85))
        if not math.isfinite(value):
            value = 0.85
        return min(max(value, 0.0), 1.0)

    def _positive_quality_threshold(self) -> float:
        value = float(getattr(self.cfg, "positive_quality_threshold", 0.20))
        if not math.isfinite(value):
            value = 0.20
        return min(max(value, 0.0), 1.0)

    def _option_quality_ratio(self) -> float:
        value = float(getattr(self.cfg, "option_quality_ratio", 0.65))
        if not math.isfinite(value):
            value = 0.65
        return min(max(value, 0.0), 1.0)

    def _option_quality_tolerance(self) -> float:
        value = float(getattr(self.cfg, "option_quality_tolerance", 0.12))
        if not math.isfinite(value):
            value = 0.12
        return max(value, 0.0)

    def _option_pressure_floor(self) -> float:
        value = float(getattr(self.cfg, "option_pressure_floor", 0.20))
        if not math.isfinite(value):
            value = 0.20
        return min(max(value, 0.0), 1.0)

    def _soft_target_temperature(self) -> float:
        value = float(getattr(self.cfg, "soft_target_temperature", 0.16))
        if not math.isfinite(value):
            value = 0.16
        return max(value, 1e-3)

    def _soft_target_pressure_tolerance(self) -> float:
        value = float(getattr(self.cfg, "soft_target_pressure_tolerance", 0.18))
        if not math.isfinite(value):
            value = 0.18
        return max(value, 0.0)

    def _soft_target_min(self) -> float:
        value = float(getattr(self.cfg, "soft_target_min", 0.04))
        if not math.isfinite(value):
            value = 0.04
        return min(max(value, 0.0), 1.0)

    def _soft_target_max(self) -> float:
        value = float(getattr(self.cfg, "soft_target_max", 0.92))
        if not math.isfinite(value):
            value = 0.92
        return min(max(value, self._soft_target_min()), 1.0)

    def _soft_target_unknown_penalty(self) -> float:
        value = float(getattr(self.cfg, "soft_target_unknown_penalty", 0.30))
        if not math.isfinite(value):
            value = 0.30
        return min(max(value, 0.0), 1.0)

    def _soft_target_risk_penalty(self) -> float:
        value = float(getattr(self.cfg, "soft_target_risk_penalty", 0.45))
        if not math.isfinite(value):
            value = 0.45
        return min(max(value, 0.0), 1.0)

    def _deployment_inertia_logit_bias(self) -> float:
        value = float(getattr(self.cfg, "inertia_logit_bias", 0.10))
        if not math.isfinite(value):
            value = 0.10
        return max(0.0, value)

    def _scale_edge_memory_budget(self, budget: torch.Tensor) -> torch.Tensor:
        ratio = self._edge_memory_budget_ratio()
        if ratio >= 1.0 or budget.numel() == 0:
            return budget

        scaled = budget.clone()
        cloud_idx = self._cloud_index(scaled.size(0))
        edge_mask = torch.ones_like(scaled, dtype=torch.bool)
        if cloud_idx >= 0:
            edge_mask[cloud_idx] = False
        scaled[edge_mask] = scaled[edge_mask] * ratio
        return scaled

    def _static_allowed_mask(
        self,
        phys_feats: Dict[str, torch.Tensor],
        logic_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the static deployment mask.

        This mask filters out physically impossible service-device pairs using
        model memory footprint and the configured effective device memory
        budget. Cloud remains a feasible fallback regardless of the edge
        memory budget ratio.

        Returns:
            `static_allowed`: boolean tensor of shape `[Ms, Np]`, where
            `static_allowed[i, n]` indicates whether service `i` is statically
            deployable on device `n`.
        """
        cap = phys_feats["mem_capacity"].float()  # [Np]
        cap = self._scale_edge_memory_budget(cap)
        model_mem = logic_feats["model_mem"].float()  # [Ms]
        Ms = model_mem.size(0)
        Np = cap.size(0)

        # Shape: [Ms, Np].
        static_allowed = model_mem.view(Ms, 1) <= cap.view(1, Np)

        # Always keep the cloud as a feasible fallback.
        cloud_idx = self._cloud_index(Np)
        if cloud_idx >= 0:
            static_allowed[:, cloud_idx] = True

        return static_allowed

    @staticmethod
    def topo_order(edge_index: torch.Tensor, num_nodes: int):
        if num_nodes <= 0:
            return []
        if edge_index.numel() == 0:
            return list(range(num_nodes))

        row, col = edge_index.detach().cpu()
        adj = [[] for _ in range(num_nodes)]
        indeg = [0 for _ in range(num_nodes)]
        for u_raw, v_raw in zip(row.tolist(), col.tolist()):
            u, v = int(u_raw), int(v_raw)
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adj[u].append(v)
                indeg[v] += 1

        q = [i for i, deg in enumerate(indeg) if deg == 0]
        order = []
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) < num_nodes:
            seen = set(order)
            order += [i for i in range(num_nodes) if i not in seen]
        return order

    def _initial_residual_mem(
        self,
        phys_feats: Dict[str, torch.Tensor],
        logic_feats: Optional[Dict[str, torch.Tensor]] = None,
        prev_deploy_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the deployment memory budget used by the projection layer.

        Hedger controls the processor replicas in the deployment mask, so the
        scheduler treats a new deployment decision as a fresh placement under
        the configured edge memory budget rather than feeding instantaneous
        device memory utilization into the policy.
        """
        cap = phys_feats["mem_capacity"].float()  # [Np] GB
        return self._scale_edge_memory_budget(cap)

    def _adapt_embeddings(self, h_s: torch.Tensor, h_p: torch.Tensor):
        return h_s, h_p

    def _qk_components(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_embedding = self.actor.q(h_s)
        k_embedding = self.actor.k(h_p)
        qk_scores = (q_embedding @ k_embedding.t()) / math.sqrt(q_embedding.size(-1))
        qk_scores = qk_scores / max(self.actor.temperature, 1e-6)
        qk_feature = _masked_standardize(qk_scores, mask)
        return q_embedding, k_embedding, qk_scores, qk_feature

    def _deployment_pair_context(
            self,
            logic_edge_index: Optional[torch.Tensor],
            logic_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            deploy_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        num_services, num_devices = static_allowed.shape
        device = static_allowed.device
        dtype = torch.float32
        cloud_idx = self._cloud_index(num_devices)
        edge_width = max(0, cloud_idx)

        if edge_width > 0:
            edge_allowed = static_allowed[:, :edge_width].bool()
        else:
            edge_allowed = torch.zeros((num_services, 0), device=device, dtype=torch.bool)
        edge_feasible_count = edge_allowed.float().sum(dim=1)

        if deploy_mask is None:
            mask = torch.zeros((num_services, num_devices), device=device, dtype=torch.bool)
        else:
            mask = deploy_mask.to(device=device).bool()

        edge_replica_count = mask[:, :edge_width].float().sum(dim=1) if edge_width > 0 \
            else torch.zeros((num_services,), device=device, dtype=dtype)
        device_replica_count = mask[:, :edge_width].float().sum(dim=0) if edge_width > 0 \
            else torch.zeros((0,), device=device, dtype=dtype)

        demand_feat = _service_demand_features(logic_feats, num_services, device, dtype)
        runtime_feat = _runtime_pair_features(logic_feats, num_services, num_devices, device, dtype)

        if edge_width > 0:
            edge_allowed_f = edge_allowed.float()
            queue_short = runtime_feat[:, :edge_width, 0].clamp_min(0.0)
            queue_tail = runtime_feat[:, :edge_width, 1].clamp_min(0.0)
            queue_freshness = runtime_feat[:, :edge_width, 5].clamp(0.0, 1.0)
            queue_signal = (queue_short + 0.5 * queue_tail) * torch.maximum(
                queue_freshness,
                torch.full_like(queue_freshness, 0.25),
            )
            queue_masked = torch.where(edge_allowed, queue_signal, torch.zeros_like(queue_signal))
            edge_queue_max = queue_masked.max(dim=1).values

            runtime_conf = (
                runtime_feat[:, :edge_width, 3].clamp(0.0, 1.0)
                * runtime_feat[:, :edge_width, 4].clamp(0.0, 1.0)
            )
            runtime_time = runtime_feat[:, :edge_width, 2].clamp_min(0.0)
            runtime_weight = runtime_conf * edge_allowed_f
            runtime_sum = (runtime_time * runtime_weight).sum(dim=1)
            runtime_den = runtime_weight.sum(dim=1).clamp_min(1e-6)
            runtime_service = runtime_sum / runtime_den
            runtime_service = torch.where(runtime_weight.sum(dim=1) > 0.0, runtime_service, torch.zeros_like(runtime_service))
        else:
            edge_queue_max = torch.zeros((num_services,), device=device, dtype=dtype)
            runtime_service = torch.zeros((num_services,), device=device, dtype=dtype)

        if logic_edge_index is not None and logic_edge_index.numel() > 0:
            parents, children, level_tensor = _topology_context(logic_edge_index.to(device=device), num_services)
            degree_values = torch.tensor(
                [len(parents[idx]) + len(children[idx]) for idx in range(num_services)],
                device=device,
                dtype=dtype,
            )
            degree_norm = degree_values / degree_values.max().clamp_min(1.0)
            dependency_criticality = (0.6 * degree_norm + 0.4 * level_tensor.to(device=device, dtype=dtype)).clamp(0.0, 1.0)
        else:
            dependency_criticality = torch.zeros((num_services,), device=device, dtype=dtype)

        compute_pressure = _unit_standardize(demand_feat[:, 0])
        arrival_pressure = _unit_standardize(demand_feat[:, 2])
        complexity_burst = torch.sigmoid(demand_feat[:, 1])
        runtime_pressure = _unit_standardize(runtime_service)
        runtime_pressure = torch.where(runtime_service > 0.0, runtime_pressure, torch.zeros_like(runtime_pressure))
        queue_pressure = torch.clamp(edge_queue_max / self._queue_normalizer(), min=0.0, max=1.0)
        feasible_mask = (edge_feasible_count > 0.0).float()

        service_pressure = (
            0.30 * compute_pressure
            + 0.15 * arrival_pressure
            + 0.10 * complexity_burst
            + 0.25 * queue_pressure
            + 0.15 * runtime_pressure
            + 0.05 * dependency_criticality
        ).clamp(0.0, 1.0) * feasible_mask

        active_pair_hotspot = torch.zeros((num_services, num_devices), device=device, dtype=dtype)
        if edge_width > 0:
            selected_edge = mask[:, :edge_width].float()
            hotspot_edge = (
                selected_edge
                * edge_allowed.float()
                * service_pressure.view(num_services, 1)
                * queue_signal.clamp_min(0.0).clamp(max=self._queue_normalizer()) / self._queue_normalizer()
            ).clamp(0.0, 1.0)
            active_pair_hotspot[:, :edge_width] = hotspot_edge
            active_service_hotspot = hotspot_edge.max(dim=1).values
        else:
            active_service_hotspot = torch.zeros((num_services,), device=device, dtype=dtype)

        return {
            "service_pressure": service_pressure,
            "edge_feasible_count": edge_feasible_count,
            "edge_replica_count": edge_replica_count,
            "device_replica_count": device_replica_count,
            "active_pair_hotspot": active_pair_hotspot,
            "active_pair_hotspot_cost": active_service_hotspot.mean() if active_service_hotspot.numel() > 0
            else torch.zeros((), device=device, dtype=dtype),
            "dependency_criticality": dependency_criticality,
            "queue_pressure": queue_pressure,
            "runtime_pressure": runtime_pressure,
        }

    def _deployment_runtime_context(
            self,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            qk_feature: torch.Tensor,
            static_allowed: torch.Tensor,
            pair_ctx: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Runtime and feasibility context for the deployment Bernoulli actor.

        This deliberately stays close to the observable state.  The deployment
        actor outputs a service-device 0/1 matrix; this context should inform
        that matrix, not run another option-set policy beside it.
        """
        num_services, num_devices = static_allowed.shape
        device = static_allowed.device
        dtype = torch.float32
        cloud_idx = self._cloud_index(num_devices)
        edge_width = max(0, cloud_idx)
        runtime_feat = _runtime_pair_features(logic_feats, num_services, num_devices, device, dtype)
        demand_feat = _service_demand_features(logic_feats, num_services, device, dtype)
        residual_mem = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask).to(
            device=device,
            dtype=dtype,
        )
        if prev_deploy_mask is None:
            prev_mask = torch.zeros((num_services, num_devices), device=device, dtype=dtype)
        else:
            prev_mask = prev_deploy_mask.to(device=device, dtype=dtype)

        queue_signal = (
            runtime_feat[..., 0].clamp_min(0.0)
            + 0.5 * runtime_feat[..., 1].clamp_min(0.0)
        ) * torch.maximum(
            runtime_feat[..., 5].clamp(0.0, 1.0),
            torch.full_like(runtime_feat[..., 5], 0.25),
        )
        queue_pressure = torch.clamp(queue_signal / self._queue_normalizer(), min=0.0, max=1.0)
        runtime_observed_confidence = runtime_feat[..., 3].clamp(0.0, 1.0)
        runtime_recency = runtime_feat[..., 4].clamp(0.0, 1.0)
        runtime_confidence = runtime_observed_confidence * runtime_recency
        runtime_unknown = (1.0 - runtime_observed_confidence).clamp(0.0, 1.0)
        runtime_stale = (1.0 - runtime_recency).clamp(0.0, 1.0)
        runtime_per_complexity = runtime_feat[..., 2].clamp_min(0.0)

        runtime_relative_weakness = torch.zeros_like(runtime_per_complexity)
        if edge_width > 0:
            edge_allowed = static_allowed[:, :edge_width].bool()
            edge_runtime = runtime_per_complexity[:, :edge_width]
            edge_confidence = runtime_confidence[:, :edge_width]
            known_edge = edge_allowed & (edge_confidence >= 0.20) & (edge_runtime > 1e-6)
            inf_runtime = torch.full_like(edge_runtime, float("inf"))
            best_runtime = torch.where(known_edge, edge_runtime, inf_runtime).min(dim=1, keepdim=True).values
            has_known = torch.isfinite(best_runtime)
            runtime_ratio = edge_runtime / best_runtime.clamp_min(1e-6)
            weak_edge = (torch.log(runtime_ratio.clamp_min(1.0)) / math.log(4.0)).clamp(0.0, 1.0)
            runtime_relative_weakness[:, :edge_width] = torch.where(
                known_edge & has_known,
                weak_edge,
                torch.zeros_like(weak_edge),
            )

        model_log_mem = demand_feat[:, 3].view(num_services, 1)
        mem_gap = model_log_mem - torch.log1p(residual_mem.clamp_min(0.0)).view(1, num_devices)
        pair_memory_cost = torch.sigmoid(mem_gap)
        device_load = prev_mask[:, :edge_width].sum(dim=0) if edge_width > 0 \
            else torch.zeros((0,), device=device, dtype=dtype)
        device_load_norm_edge = device_load / device_load.max().clamp_min(1.0) if edge_width > 0 else device_load
        device_load_norm = torch.zeros((num_services, num_devices), device=device, dtype=dtype)
        if edge_width > 0:
            device_load_norm[:, :edge_width] = device_load_norm_edge.view(1, edge_width).expand(num_services, -1)

        static_allowed_f = static_allowed.to(device=device, dtype=dtype)
        memory_fit = (1.0 - pair_memory_cost).clamp(0.0, 1.0)
        static_quality_score = torch.clamp(
            static_allowed_f * torch.sigmoid(qk_feature.to(device=device, dtype=dtype)) * memory_fit,
            min=0.0,
            max=1.0,
        )
        observed_quality_score = torch.clamp(
            static_allowed_f
            * (1.0 - runtime_relative_weakness).clamp(0.0, 1.0)
            * (1.0 - queue_pressure).clamp(0.0, 1.0)
            * memory_fit,
            min=0.0,
            max=1.0,
        )
        pair_quality_score = torch.clamp(
            runtime_confidence * observed_quality_score
            + (1.0 - runtime_confidence) * static_quality_score,
            min=0.0,
            max=1.0,
        )
        if cloud_idx >= 0:
            device_ids = torch.arange(num_devices, device=device)
            cloud_mask = device_ids.view(1, -1).eq(cloud_idx).expand(num_services, -1)
            static_quality_score = torch.where(cloud_mask, torch.zeros_like(static_quality_score), static_quality_score)
            pair_quality_score = torch.where(cloud_mask, torch.zeros_like(pair_quality_score), pair_quality_score)
        evidence_untrusted = torch.maximum(runtime_unknown, runtime_stale)
        quality_threshold = max(self._positive_quality_threshold(), 1e-6)
        low_quality_gap = ((quality_threshold - pair_quality_score) / quality_threshold).clamp(0.0, 1.0)

        active_hotspot = pair_ctx["active_pair_hotspot"].to(device=device, dtype=dtype)
        runtime_risk_score = torch.clamp(
            0.55 * runtime_confidence * runtime_relative_weakness
            + 0.50 * queue_pressure
            + 0.20 * runtime_confidence * low_quality_gap
            + active_hotspot,
            min=0.0,
            max=1.0,
        )

        return {
            "runtime_per_complexity": runtime_per_complexity,
            "queue_pressure": queue_pressure,
            "runtime_confidence": runtime_confidence,
            "runtime_recency": runtime_recency,
            "runtime_unknown": runtime_unknown,
            "runtime_stale": runtime_stale,
            "runtime_relative_weakness": runtime_relative_weakness,
            "pair_memory_cost": pair_memory_cost,
            "device_load_norm": device_load_norm,
            "static_quality_score": static_quality_score,
            "observed_quality_score": observed_quality_score,
            "pair_quality_score": pair_quality_score,
            "runtime_unknown_risk": runtime_unknown,
            "runtime_stale_risk": runtime_stale,
            "static_option_score": qk_feature.to(device=device, dtype=dtype),
            "runtime_risk_score": runtime_risk_score,
            "evidence_confidence": runtime_confidence,
            "evidence_untrusted": evidence_untrusted,
            "low_quality_gap": low_quality_gap,
        }

    def _deployment_candidate_features(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            qk_feature: torch.Tensor,
            static_allowed: torch.Tensor,
            pair_ctx: Dict[str, torch.Tensor],
            option_ctx: Dict[str, torch.Tensor],
            service_ctx: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_services, num_devices = qk_feature.shape
        device = qk_feature.device
        dtype = qk_feature.dtype
        cloud_idx = self._cloud_index(num_devices)
        device_feat = _device_capability_features(phys_feats, num_devices, device, dtype, cloud_idx)

        if prev_deploy_mask is None:
            prev_mask = torch.zeros((num_services, num_devices), device=device, dtype=dtype)
        else:
            prev_mask = prev_deploy_mask.to(device=device, dtype=dtype)
        device_service_count = prev_mask.sum(dim=0).clamp_min(0.0)

        static_allowed_float = static_allowed.to(device=device, dtype=dtype)
        active_pair_hotspot = pair_ctx["active_pair_hotspot"].to(device=device, dtype=dtype)

        def service_column(name: str) -> torch.Tensor:
            value = service_ctx[name].to(device=device, dtype=dtype).view(num_services, 1)
            return value.expand(-1, num_devices)

        return torch.stack(
            [
                qk_feature,
                service_column("service_pressure"),
                service_column("dependency_criticality"),
                torch.log1p(service_column("edge_feasible_count").clamp_min(0.0)),
                torch.log1p(service_column("edge_replica_count").clamp_min(0.0)),
                service_column("service_best_pair_quality"),
                service_column("service_quality_gap_top_second"),
                service_column("service_best_runtime_risk"),
                service_column("service_max_queue_pressure"),
                device_feat[:, 2].view(1, num_devices).expand(num_services, -1),
                device_feat[:, 0].view(1, num_devices).expand(num_services, -1),
                device_feat[:, 1].view(1, num_devices).expand(num_services, -1),
                option_ctx["static_quality_score"].to(device=device, dtype=dtype),
                option_ctx["pair_quality_score"].to(device=device, dtype=dtype),
                option_ctx["runtime_risk_score"].to(device=device, dtype=dtype),
                option_ctx["queue_pressure"].to(device=device, dtype=dtype),
                option_ctx["runtime_confidence"].to(device=device, dtype=dtype),
                option_ctx["runtime_relative_weakness"].to(device=device, dtype=dtype),
                option_ctx["low_quality_gap"].to(device=device, dtype=dtype),
                option_ctx["pair_memory_cost"].to(device=device, dtype=dtype),
                prev_mask,
                torch.log1p(device_service_count).view(1, num_devices).expand(num_services, -1),
                static_allowed_float,
                active_pair_hotspot,
            ],
            dim=-1,
        )

    def _deployment_service_context_features(
            self,
            logic_feats: Dict[str, torch.Tensor],
            pair_ctx: Dict[str, torch.Tensor],
            option_ctx: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_services, num_devices = static_allowed.shape
        device = static_allowed.device
        dtype = torch.float32
        cloud_idx = self._cloud_index(num_devices)
        edge_width = max(0, cloud_idx)
        demand_feat = _service_demand_features(logic_feats, num_services, device, dtype)
        service_pressure = pair_ctx["service_pressure"].to(device=device, dtype=dtype)
        dependency_criticality = pair_ctx["dependency_criticality"].to(device=device, dtype=dtype)
        edge_feasible_count = pair_ctx["edge_feasible_count"].to(device=device, dtype=dtype)
        edge_replica_count = pair_ctx["edge_replica_count"].to(device=device, dtype=dtype)

        if edge_width > 0:
            edge_allowed = static_allowed[:, :edge_width].bool()
            pair_quality_edge = option_ctx["pair_quality_score"][:, :edge_width].to(device=device, dtype=dtype)
            runtime_risk_edge = option_ctx["runtime_risk_score"][:, :edge_width].to(device=device, dtype=dtype)
            queue_edge = option_ctx["queue_pressure"][:, :edge_width].to(device=device, dtype=dtype)

            masked_quality = pair_quality_edge.masked_fill(~edge_allowed, -1.0)
            top2_quality = masked_quality.topk(k=min(2, edge_width), dim=1).values
            best_quality = top2_quality[:, 0].clamp_min(0.0)
            if edge_width >= 2:
                second_quality = top2_quality[:, 1].clamp_min(0.0)
            else:
                second_quality = torch.zeros_like(best_quality)
            quality_gap = (best_quality - second_quality).clamp_min(0.0)
            best_runtime_risk = runtime_risk_edge.masked_fill(~edge_allowed, 1.0).min(dim=1).values.clamp(0.0, 1.0)
            max_queue_pressure = queue_edge.masked_fill(~edge_allowed, 0.0).max(dim=1).values.clamp(0.0, 1.0)
        else:
            best_quality = torch.zeros((num_services,), device=device, dtype=dtype)
            second_quality = torch.zeros_like(best_quality)
            quality_gap = torch.zeros_like(best_quality)
            best_runtime_risk = torch.ones_like(best_quality)
            max_queue_pressure = torch.zeros_like(best_quality)

        service_features = torch.stack(
            [
                demand_feat[:, 0],
                demand_feat[:, 1],
                demand_feat[:, 2],
                demand_feat[:, 3],
                service_pressure,
                dependency_criticality,
                torch.log1p(edge_feasible_count.clamp_min(0.0)),
                torch.log1p(edge_replica_count.clamp_min(0.0)),
                best_quality,
                second_quality,
                quality_gap,
                best_runtime_risk,
                max_queue_pressure,
            ],
            dim=-1,
        )
        return service_features, {
            "service_pressure": service_pressure,
            "dependency_criticality": dependency_criticality,
            "edge_feasible_count": edge_feasible_count,
            "edge_replica_count": edge_replica_count,
            "service_best_pair_quality": best_quality,
            "service_second_pair_quality": second_quality,
            "service_quality_gap_top_second": quality_gap,
            "service_best_runtime_risk": best_runtime_risk,
            "service_max_queue_pressure": max_queue_pressure,
        }

    def _deployment_actor_terms(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor],
    ]:
        q_embedding, k_embedding, qk_scores, qk_feature = self._qk_components(h_s, h_p, static_allowed)
        pair_ctx = self._deployment_pair_context(
            logic_edge_index,
            logic_feats,
            static_allowed,
            deploy_mask=prev_deploy_mask,
        )
        option_ctx = self._deployment_runtime_context(
            logic_feats,
            phys_feats,
            qk_feature,
            static_allowed,
            pair_ctx,
            prev_deploy_mask=prev_deploy_mask,
        )
        service_features, service_ctx = self._deployment_service_context_features(
            logic_feats,
            pair_ctx,
            option_ctx,
            static_allowed,
        )
        candidate_features = self._deployment_candidate_features(
            logic_edge_index,
            logic_feats,
            phys_feats,
            qk_feature,
            static_allowed,
            pair_ctx,
            option_ctx,
            service_ctx,
            prev_deploy_mask=prev_deploy_mask,
        )
        cloud_idx = self._cloud_index(static_allowed.size(1))
        pair_rank_logit_raw = self.deployment_pair_rank_head(candidate_features.float())
        edge_allowed = static_allowed.bool().clone()
        if cloud_idx >= 0:
            edge_allowed[:, cloud_idx] = False
        pair_rank_mean = _masked_mean(pair_rank_logit_raw, edge_allowed, dim=1, keepdim=True)
        pair_centered_logit = torch.where(
            edge_allowed,
            pair_rank_logit_raw - pair_rank_mean,
            torch.zeros_like(pair_rank_logit_raw),
        )
        if prev_deploy_mask is None:
            current_replica = torch.zeros_like(pair_rank_logit_raw)
        else:
            current_replica = (
                prev_deploy_mask.to(device=pair_rank_logit_raw.device).bool() & edge_allowed
            ).to(dtype=pair_rank_logit_raw.dtype)
        # Deployment is represented directly as a service-device Bernoulli
        # matrix.  The absolute pair logit is the actor's deployment intent;
        # deterministic inference selects edge pairs with logit > 0 and the
        # projection step only repairs hard feasibility constraints.
        select_logits_raw = pair_rank_logit_raw + self._deployment_inertia_logit_bias() * current_replica
        select_logits = select_logits_raw
        invalid = ~static_allowed.bool()
        select_logits = torch.where(invalid, torch.full_like(select_logits, -20.0), select_logits)
        if cloud_idx >= 0:
            device_ids = torch.arange(static_allowed.size(1), device=h_s.device)
            cloud_mask = device_ids.view(1, -1).eq(cloud_idx).expand_as(static_allowed)
            select_logits = torch.where(cloud_mask, torch.full_like(select_logits, 20.0), select_logits)
        pair_ctx.update(option_ctx)
        pair_ctx.update(service_ctx)
        pair_ctx["base_score"] = qk_feature
        pair_ctx["service_context_feature"] = service_features
        pair_ctx["pair_rank_logit_raw"] = pair_rank_logit_raw
        pair_ctx["pair_centered_logit"] = pair_centered_logit
        pair_ctx["pair_utility_logit"] = select_logits
        pair_ctx["centered_score"] = pair_centered_logit
        pair_ctx["select_logit"] = select_logits
        pair_ctx["select_prob"] = torch.sigmoid(select_logits)
        return (
            q_embedding,
            k_embedding,
            qk_scores,
            qk_feature,
            select_logits_raw,
            candidate_features,
            select_logits,
            pair_ctx,
        )

    @torch.no_grad()
    def estimate_value(
            self,
            logic_edge_index,
            logic_feats,
            phys_edge_index,
            phys_feats,
            prev_deploy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Estimate the state value for rollout bootstrap.

        Hedger uses fixed-length truncated rollouts in a continuous online
        control loop, so PPO should bootstrap the final transition with the
        critic value of the successor state instead of forcing it to zero.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)
        _, _, _, _, _, candidate_features, _, _ = self._deployment_actor_terms(
            h_s,
            h_p,
            logic_edge_index,
            logic_feats,
            phys_feats,
            static_allowed,
            prev_deploy_mask=prev_deploy_mask,
        )
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        return value.squeeze(0)

    def _enforce_cloud_replica(self, deploy_mask: torch.Tensor) -> torch.Tensor:
        """Force every service to keep a cloud replica in the deployment mask."""
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (deploy_mask.size(1) - 1)
        deploy_mask = deploy_mask.clone()
        deploy_mask[:, cloud_idx] = True
        return deploy_mask

    @torch.no_grad()
    def _sample_matrix_deployment_mask(
            self,
            select_logits: torch.Tensor,
            static_allowed: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor],
            deterministic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        num_services, num_devices = select_logits.shape
        cloud_idx = self._cloud_index(num_devices)
        edge_allowed = static_allowed.to(device=select_logits.device).bool().clone()
        if cloud_idx >= 0:
            edge_allowed[:, cloud_idx] = False
        if prev_deploy_mask is None:
            prev_edge = torch.zeros_like(edge_allowed)
        else:
            prev_edge = prev_deploy_mask.to(device=select_logits.device).bool() & edge_allowed

        select_prob = torch.sigmoid(select_logits).clamp(1e-6, 1.0 - 1e-6)
        if deterministic:
            selected_edge = (select_logits > 0.0) & edge_allowed
        else:
            selected_edge = torch.bernoulli(select_prob).bool() & edge_allowed

        deploy_mask = torch.zeros(
            (num_services, num_devices),
            device=select_logits.device,
            dtype=torch.bool,
        )
        deploy_mask[:, :] = selected_edge
        if cloud_idx >= 0:
            deploy_mask[:, cloud_idx] = True

        added_mask = selected_edge & ~prev_edge
        kept_mask = selected_edge & prev_edge
        removed_mask = prev_edge & edge_allowed & ~selected_edge
        selected_edge_count = selected_edge.float().sum()
        selected_prob_mean = (
            select_prob.masked_select(selected_edge).mean()
            if bool(selected_edge.any().item())
            else torch.tensor(0.0, device=select_logits.device)
        )
        return deploy_mask, {
            "matrix_added_mask": added_mask,
            "matrix_kept_mask": kept_mask,
            "matrix_removed_mask": removed_mask,
            "pre_quality_raw_mask": deploy_mask.clone(),
            "pre_quality_added_mask": added_mask,
            "pre_quality_kept_mask": kept_mask,
            "select_prob": select_prob,
            "matrix_raw_selected_edge": selected_edge,
            "matrix_raw_selected_cnt": selected_edge_count,
            "matrix_selected_prob_mean": selected_prob_mean,
        }

    def _deployment_select_logp_entropy(
            self,
            select_logits: torch.Tensor,
            target_mask: torch.Tensor,
            static_allowed: torch.Tensor,
            topo_order: Optional[Sequence[int]] = None,
            positive_mask: Optional[torch.Tensor] = None,
            negative_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_services, num_devices = select_logits.shape
        cloud_idx = self._cloud_index(num_devices)
        if topo_order is None:
            topo_order = list(range(num_services))
        edge_allowed = static_allowed.to(device=select_logits.device).bool().clone()
        if cloud_idx >= 0:
            edge_allowed[:, cloud_idx] = False
        target_edge = target_mask.to(device=select_logits.device).bool() & edge_allowed
        prob = torch.sigmoid(select_logits).clamp(1e-6, 1.0 - 1e-6)

        if positive_mask is None:
            positive_mask_t = target_edge
        else:
            positive_mask_t = positive_mask.to(device=select_logits.device).bool() & edge_allowed
        if negative_mask is None:
            negative_mask_t = torch.zeros_like(edge_allowed)
        else:
            negative_mask_t = negative_mask.to(device=select_logits.device).bool() & edge_allowed & ~positive_mask_t

        positive_count = positive_mask_t.float().sum()
        positive_logp = torch.log(prob).masked_select(positive_mask_t).sum() / positive_count.clamp_min(1.0)
        negative_count = negative_mask_t.float().sum()
        negative_logp = torch.tensor(0.0, device=select_logits.device)
        if bool(negative_mask_t.any().item()):
            negative_logp = torch.log1p(-prob).masked_select(negative_mask_t).sum() / negative_count.clamp_min(1.0)

        ent_terms = []
        for service_idx in topo_order:
            allowed_row = edge_allowed[service_idx]
            if not bool(allowed_row.any().item()):
                continue
            row_entropy = torch.distributions.Bernoulli(probs=prob[service_idx]).entropy()
            ent_terms.append(row_entropy.masked_select(allowed_row).mean())
        entropy = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=select_logits.device)
        return (
            positive_logp + negative_logp,
            entropy,
            positive_logp,
            negative_logp,
            positive_mask_t,
            negative_mask_t,
        )

    def _deployment_effective_option_masks(
            self,
            static_allowed: torch.Tensor,
            policy_ctx: Dict[str, torch.Tensor],
            *,
            top_quality_tolerance: Optional[float] = None,
            option_quality_ratio: Optional[float] = None,
            coverage_pressure_floor: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build service-local option labels for the deployment Bernoulli matrix.

        The actor still decides each service-device bit independently.  These
        masks only define what the offline objective considers an effective
        edge option: feasible, not clearly risky, and close enough to the best
        quality available for that service.  This gives the network positive
        signal for multiple usable replicas without adding rule-based replicas
        during inference.
        """
        device = static_allowed.device
        dtype = torch.float32
        static_allowed = static_allowed.bool()
        num_services, num_devices = static_allowed.shape
        cloud_idx = self._cloud_index(num_devices)
        edge_allowed = static_allowed.clone()
        if cloud_idx >= 0:
            edge_allowed[:, cloud_idx] = False

        def ctx_matrix(name: str, default: float) -> torch.Tensor:
            value = policy_ctx.get(name)
            if isinstance(value, torch.Tensor) and value.shape == static_allowed.shape:
                return value.to(device=device, dtype=dtype)
            return torch.full(static_allowed.shape, float(default), device=device, dtype=dtype)

        pair_quality = ctx_matrix("pair_quality_score", 1.0).clamp(0.0, 1.0)
        queue_pressure = ctx_matrix("queue_pressure", 0.0).clamp(0.0, 1.0)
        hotspot = ctx_matrix("active_pair_hotspot", 0.0).clamp(0.0, 1.0)
        runtime_risk = ctx_matrix("runtime_risk_score", 0.0).clamp(0.0, 1.0)
        runtime_unknown = ctx_matrix("runtime_unknown_risk", 0.0).clamp(0.0, 1.0)
        runtime_stale = ctx_matrix("runtime_stale_risk", 0.0).clamp(0.0, 1.0)

        service_pressure = policy_ctx.get("service_pressure")
        if isinstance(service_pressure, torch.Tensor) and service_pressure.dim() == 1 \
                and service_pressure.size(0) == num_services:
            service_pressure = service_pressure.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        else:
            service_pressure = torch.zeros((num_services,), device=device, dtype=dtype)

        quality_threshold = float(self._positive_quality_threshold())
        tolerance = self._option_quality_tolerance() if top_quality_tolerance is None \
            else max(0.0, float(top_quality_tolerance))
        quality_ratio = self._option_quality_ratio() if option_quality_ratio is None \
            else min(max(0.0, float(option_quality_ratio)), 1.0)
        pressure_floor = self._option_pressure_floor() if coverage_pressure_floor is None \
            else min(max(0.0, float(coverage_pressure_floor)), 1.0)

        severe_risky = (
            (queue_pressure >= self._negative_queue_threshold())
            | (hotspot >= self._negative_hotspot_threshold())
            | (runtime_risk >= self._negative_runtime_risk_threshold())
        ) & edge_allowed
        evidence_risky = (
            (runtime_unknown >= self._negative_unknown_threshold())
            | (runtime_stale >= self._negative_stale_threshold())
        ) & edge_allowed
        low_quality = (pair_quality < quality_threshold) & edge_allowed
        quality_eligible = edge_allowed & ~severe_risky & ~evidence_risky & ~low_quality

        best_quality = pair_quality.masked_fill(~quality_eligible, -1.0).max(dim=1).values
        has_effective_candidate = best_quality >= quality_threshold
        near_top_floor = (best_quality - tolerance).clamp_min(quality_threshold)
        relative_floor = (best_quality * quality_ratio).clamp_min(quality_threshold)
        option_floor = torch.minimum(near_top_floor, relative_floor)
        option_floor = torch.where(has_effective_candidate, option_floor, torch.ones_like(option_floor))
        effective_option_mask = (
            quality_eligible
            & (pair_quality >= option_floor.view(num_services, 1))
            & has_effective_candidate.view(num_services, 1)
        )
        top_quality_mask = (
            quality_eligible
            & (pair_quality >= near_top_floor.view(num_services, 1))
            & has_effective_candidate.view(num_services, 1)
        )
        pressure_weight = torch.where(
            edge_allowed.any(dim=1),
            torch.maximum(service_pressure, torch.full_like(service_pressure, pressure_floor)),
            torch.zeros_like(service_pressure),
        )
        non_effective_option_mask = (
            edge_allowed
            & ~effective_option_mask
            & has_effective_candidate.view(num_services, 1)
        )
        clear_gap = max(0.02, 0.5 * float(tolerance))
        clear_non_effective_mask = (
            non_effective_option_mask
            & (pair_quality < (option_floor.view(num_services, 1) - clear_gap).clamp_min(0.0))
        )
        risky_option_mask = (severe_risky | evidence_risky | low_quality) & edge_allowed
        return {
            "edge_allowed": edge_allowed,
            "pair_quality": pair_quality,
            "runtime_unknown": runtime_unknown,
            "runtime_stale": runtime_stale,
            "quality_eligible": quality_eligible,
            "top_quality_mask": top_quality_mask,
            "effective_option_mask": effective_option_mask,
            "non_effective_option_mask": non_effective_option_mask,
            "clear_non_effective_mask": clear_non_effective_mask,
            "evidence_risky_mask": evidence_risky,
            "risky_option_mask": risky_option_mask,
            "option_floor": option_floor,
            "best_quality": best_quality.clamp_min(0.0),
            "service_pressure": service_pressure,
            "pressure_weight": pressure_weight,
            "has_effective_candidate": has_effective_candidate,
        }

    def _deployment_soft_option_targets(
            self,
            static_allowed: torch.Tensor,
            policy_ctx: Dict[str, torch.Tensor],
            *,
            selected_mask: Optional[torch.Tensor] = None,
            raw_selected_mask: Optional[torch.Tensor] = None,
            quality_bucket: str = "unknown",
            top_quality_tolerance: Optional[float] = None,
            option_quality_ratio: Optional[float] = None,
            coverage_pressure_floor: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build pressure-aware soft labels for the deployment Bernoulli matrix.

        Inference stays literal: an edge pair is selected when its learned logit
        is above zero.  Offline learning, however, should not reduce a service's
        alternatives to a single hard top candidate.  This target gives every
        sufficiently good service-device pair a continuous target based on its
        relative quality and the service pressure, while weak/risky/unknown
        pairs receive explicit pressure to stay below the p=0.5 boundary.
        """
        device = static_allowed.device
        dtype = torch.float32
        option_masks = self._deployment_effective_option_masks(
            static_allowed,
            policy_ctx,
            top_quality_tolerance=top_quality_tolerance,
            option_quality_ratio=option_quality_ratio,
            coverage_pressure_floor=coverage_pressure_floor,
        )
        edge_allowed = option_masks["edge_allowed"]
        pair_quality = option_masks["pair_quality"]
        best_quality = option_masks["best_quality"]
        effective_option_mask = option_masks["effective_option_mask"]
        clear_non_effective_mask = option_masks["clear_non_effective_mask"]
        risky_option_mask = option_masks["risky_option_mask"]
        evidence_risky_mask = option_masks["evidence_risky_mask"]
        pressure_weight = option_masks["pressure_weight"]

        def ctx_matrix(name: str, default: float) -> torch.Tensor:
            value = policy_ctx.get(name)
            if isinstance(value, torch.Tensor) and value.shape == static_allowed.shape:
                return value.to(device=device, dtype=dtype)
            return torch.full(static_allowed.shape, float(default), device=device, dtype=dtype)

        queue_pressure = ctx_matrix("queue_pressure", 0.0).clamp(0.0, 1.0)
        runtime_risk = ctx_matrix("runtime_risk_score", 0.0).clamp(0.0, 1.0)
        runtime_unknown = ctx_matrix("runtime_unknown_risk", 0.0).clamp(0.0, 1.0)
        runtime_stale = ctx_matrix("runtime_stale_risk", 0.0).clamp(0.0, 1.0)
        low_quality_gap = ctx_matrix("low_quality_gap", 0.0).clamp(0.0, 1.0)
        active_hotspot = ctx_matrix("active_pair_hotspot", 0.0).clamp(0.0, 1.0)

        num_services = static_allowed.size(0)
        quality_threshold = float(self._positive_quality_threshold())
        tolerance = self._option_quality_tolerance() if top_quality_tolerance is None \
            else max(0.0, float(top_quality_tolerance))
        quality_ratio = self._option_quality_ratio() if option_quality_ratio is None \
            else min(max(0.0, float(option_quality_ratio)), 1.0)
        pressure_tolerance = float(self._soft_target_pressure_tolerance())
        temp = float(self._soft_target_temperature())
        min_target = float(self._soft_target_min())
        max_target = float(self._soft_target_max())

        pressure_expand = pressure_weight.view(num_services, 1).clamp(0.0, 1.0)
        near_top_floor = (
            best_quality
            - tolerance
            - pressure_tolerance * pressure_expand.squeeze(1)
        ).clamp_min(quality_threshold)
        relative_floor = (best_quality * quality_ratio).clamp_min(quality_threshold)
        soft_floor = torch.minimum(near_top_floor, relative_floor).view(num_services, 1)

        quality_margin = (pair_quality - soft_floor) / temp
        raw_target = torch.sigmoid(quality_margin).clamp(0.0, 1.0)
        target = min_target + (max_target - min_target) * raw_target

        evidence_penalty = self._soft_target_unknown_penalty() * torch.maximum(runtime_unknown, runtime_stale)
        risk_penalty = self._soft_target_risk_penalty() * torch.clamp(
            0.45 * runtime_risk
            + 0.25 * queue_pressure
            + 0.25 * active_hotspot
            + 0.20 * low_quality_gap,
            min=0.0,
            max=1.0,
        )
        target = (target - evidence_penalty - risk_penalty).clamp(min_target, max_target)

        selected = torch.zeros_like(edge_allowed)
        if isinstance(selected_mask, torch.Tensor) and selected_mask.shape == static_allowed.shape:
            selected = selected_mask.to(device=device).bool() & edge_allowed
        raw_selected = selected
        if isinstance(raw_selected_mask, torch.Tensor) and raw_selected_mask.shape == static_allowed.shape:
            raw_selected = raw_selected_mask.to(device=device).bool() & edge_allowed

        safe_selected = selected & effective_option_mask & ~evidence_risky_mask & ~risky_option_mask
        if bool(safe_selected.any().item()):
            selected_floor = torch.full_like(target, 0.58)
            target = torch.where(safe_selected, torch.maximum(target, selected_floor), target)

        quality = str(quality_bucket or "unknown").strip().lower()
        if quality == "bad":
            selected_risky = selected & (risky_option_mask | evidence_risky_mask | clear_non_effective_mask)
            target = torch.where(selected_risky, torch.full_like(target, min_target), target * 0.65)

        target = torch.where(edge_allowed, target, torch.zeros_like(target))
        positive_mask = (target >= 0.50) & edge_allowed
        negative_mask = (target <= 0.35) & edge_allowed
        risk_weight = (
            risky_option_mask.to(dtype=dtype)
            + evidence_risky_mask.to(dtype=dtype)
            + clear_non_effective_mask.to(dtype=dtype)
        ).clamp(0.0, 1.0)
        target_weight = edge_allowed.to(dtype=dtype) * (
            0.35
            + 0.75 * pressure_expand
            + 0.50 * pair_quality
            + 0.50 * positive_mask.to(dtype=dtype)
            + 0.65 * risk_weight
            + 0.35 * raw_selected.to(dtype=dtype)
        )
        if quality == "bad":
            target_weight = target_weight + selected.to(dtype=dtype) * 0.75

        return {
            **option_masks,
            "soft_target": target,
            "soft_target_weight": target_weight,
            "soft_target_positive_mask": positive_mask,
            "soft_target_negative_mask": negative_mask,
            "soft_target_floor": soft_floor.squeeze(1),
            "soft_target_margin": quality_margin,
        }

    def _deployment_plan_level_terms(
            self,
            select_logits: torch.Tensor,
            static_allowed: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            policy_ctx: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor] = None,
            *,
            positive_logit_margin: float = 0.25,
            coverage_logit_margin: float = 0.20,
            ranking_logit_margin: float = 0.15,
            contrast_logit_margin: float = 0.30,
            top_quality_tolerance: float = 0.05,
            coverage_pressure_floor: float = 0.35,
            option_quality_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Direct-matrix diagnostics and light margin losses.

        The deployment actor's contract is now literal: each edge pair is
        selected when its Bernoulli logit is above zero.  These terms therefore
        only calibrate that matrix; they do not build a teacher deployment plan.
        """
        device = select_logits.device
        dtype = torch.float32
        static_allowed = static_allowed.to(device=device).bool()
        num_services, num_devices = static_allowed.shape
        cloud_idx = self._cloud_index(num_devices)
        option_masks = self._deployment_soft_option_targets(
            static_allowed,
            policy_ctx,
            top_quality_tolerance=top_quality_tolerance,
            option_quality_ratio=option_quality_ratio,
            coverage_pressure_floor=coverage_pressure_floor,
        )
        edge_allowed = option_masks["edge_allowed"]
        pair_quality = option_masks["pair_quality"]
        top_quality_mask = option_masks["top_quality_mask"]
        effective_option_mask = option_masks["effective_option_mask"]
        non_effective_option_mask = option_masks["non_effective_option_mask"]
        clear_non_effective_mask = option_masks["clear_non_effective_mask"]
        risky_option_mask = option_masks["risky_option_mask"]
        pressure_weight = option_masks["pressure_weight"]
        has_effective_candidate = option_masks["has_effective_candidate"]
        soft_target = option_masks["soft_target"].detach()
        soft_target_weight = option_masks["soft_target_weight"].detach()
        soft_target_positive_mask = option_masks["soft_target_positive_mask"]
        soft_target_negative_mask = option_masks["soft_target_negative_mask"]

        select_prob = torch.sigmoid(select_logits).clamp(1e-6, 1.0 - 1e-6)
        pair_suppression_logits = select_logits
        pair_suppression_prob = torch.sigmoid(pair_suppression_logits).clamp(1e-6, 1.0 - 1e-6)
        edge_prob = torch.where(edge_allowed, select_prob, torch.zeros_like(select_prob))
        has_edge = edge_allowed.any(dim=1)

        one_minus_edge = torch.where(edge_allowed, (1.0 - edge_prob).clamp(1e-6, 1.0), torch.ones_like(edge_prob))
        no_edge_prob = one_minus_edge.prod(dim=1)
        expected_quality_mass = (edge_prob * pair_quality).sum(dim=1)

        quality_threshold = float(self._positive_quality_threshold())
        quality_advantage = (pair_quality - quality_threshold).clamp_min(0.0).masked_fill(~edge_allowed, 0.0)

        coverage_score = select_logits.masked_fill(~effective_option_mask, -1.0e6).max(dim=1).values
        coverage_services = has_effective_candidate & has_edge
        coverage_weights = torch.maximum(
            pressure_weight,
            torch.full_like(pressure_weight, float(coverage_pressure_floor)),
        ) * coverage_services.to(dtype=dtype)
        coverage_margin_loss = (
            F.softplus(float(coverage_logit_margin) - coverage_score.clamp_min(-20.0))
            * coverage_weights
        ).sum() / coverage_weights.sum().clamp_min(1.0)

        option_quality_weights = top_quality_mask.to(dtype=dtype) * (
            0.5 + quality_advantage.clamp(0.0, 1.0)
        )
        quality_transition_weight = coverage_weights.view(num_services, 1)
        quality_margin_den = (
            option_quality_weights * quality_transition_weight
        ).sum().clamp_min(1.0)
        quality_margin_loss = (
            F.softplus(float(positive_logit_margin) - select_logits)
            * option_quality_weights
            * quality_transition_weight
        ).sum() / quality_margin_den

        ranking_weights = risky_option_mask.to(dtype=dtype) * torch.maximum(
            pressure_weight,
            torch.ones_like(pressure_weight) * 0.25,
        ).view(num_services, 1)
        ranking_den = ranking_weights.sum().clamp_min(1.0)
        ranking_margin_loss = (
            F.softplus(pair_suppression_logits + float(ranking_logit_margin)) * ranking_weights
        ).sum() / ranking_den

        competitor_mask = (
            (clear_non_effective_mask | risky_option_mask)
            & edge_allowed
            & has_effective_candidate.view(num_services, 1)
        )
        contrast_services = has_effective_candidate & top_quality_mask.any(dim=1) & competitor_mask.any(dim=1)
        top_competition_logit = select_logits.masked_fill(~top_quality_mask, -1.0e6).max(dim=1).values
        competitor_logit = select_logits.masked_fill(~competitor_mask, -1.0e6).max(dim=1).values
        contrast_gap = top_competition_logit - competitor_logit
        contrast_weights = torch.maximum(
            pressure_weight,
            torch.full_like(pressure_weight, float(coverage_pressure_floor)),
        ) * contrast_services.to(dtype=dtype)
        contrast_margin_loss = (
            F.softplus(float(contrast_logit_margin) - contrast_gap.clamp(-20.0, 20.0))
            * contrast_weights
        ).sum() / contrast_weights.sum().clamp_min(1.0)

        expected_edge_count = edge_prob.sum(dim=1)

        model_mem = logic_feats.get("model_mem")
        if not isinstance(model_mem, torch.Tensor):
            model_mem = torch.zeros((num_services,), device=device, dtype=dtype)
        model_mem = model_mem.to(device=device, dtype=dtype).view(num_services)
        residual_mem = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask).to(
            device=device,
            dtype=dtype,
        )
        expected_device_mem = (edge_prob * model_mem.view(num_services, 1)).sum(dim=0)
        edge_device = torch.ones((num_devices,), device=device, dtype=torch.bool)
        if cloud_idx >= 0:
            edge_device[cloud_idx] = False
        if bool(edge_device.any().item()):
            memory_overage = F.relu(expected_device_mem - residual_mem) / residual_mem.clamp_min(1e-3)
            memory_loss = memory_overage.masked_select(edge_device).mean()
            memory_overage_mean = memory_overage.masked_select(edge_device).mean()
            memory_overage_max = memory_overage.masked_select(edge_device).max()
        else:
            zero = select_logits.sum() * 0.0
            memory_loss = zero
            memory_overage_mean = zero
            memory_overage_max = zero

        if bool(edge_allowed.any().item()):
            edge_probs = select_prob.masked_select(edge_allowed)
            edge_logits = select_logits.masked_select(edge_allowed)
            edge_prob_mean = edge_probs.mean()
            edge_prob_std = edge_probs.std(unbiased=False)
            edge_logit_mean = edge_logits.mean()
            edge_logit_std = edge_logits.std(unbiased=False)
            prob_above_05_ratio = (edge_probs > 0.5).to(dtype=dtype).mean()
        else:
            zero = select_logits.sum() * 0.0
            edge_prob_mean = zero
            edge_prob_std = zero
            edge_logit_mean = zero
            edge_logit_std = zero
            prob_above_05_ratio = zero
        zero = select_logits.sum() * 0.0

        feasible_no_edge = no_edge_prob.masked_select(has_edge) if bool(has_edge.any().item()) else None
        feasible_edge_count = expected_edge_count.masked_select(has_edge) if bool(has_edge.any().item()) else None
        feasible_quality = expected_quality_mass.masked_select(has_edge) if bool(has_edge.any().item()) else None
        top_quality_probs = select_prob.masked_select(top_quality_mask)
        effective_option_probs = select_prob.masked_select(effective_option_mask)
        effective_option_logits = select_logits.masked_select(effective_option_mask)
        non_top_mask = non_effective_option_mask
        non_top_probs = select_prob.masked_select(non_top_mask)
        top_quality_count = top_quality_mask.float().sum(dim=1).masked_select(has_effective_candidate)
        effective_option_count_all = effective_option_mask.to(dtype=dtype).sum(dim=1)
        effective_option_count = effective_option_count_all.masked_select(has_effective_candidate)
        non_top_count = non_top_mask.float().sum(dim=1).masked_select(has_effective_candidate)
        effective_option_mass = (edge_prob * soft_target).sum(dim=1)
        desired_effective_option_mass = (soft_target * edge_allowed.to(dtype=dtype)).sum(dim=1)
        effective_option_shortage = (
            F.relu(soft_target - edge_prob)
            * soft_target_weight
            * edge_allowed.to(dtype=dtype)
        ).sum(dim=1) / (soft_target_weight * edge_allowed.to(dtype=dtype)).sum(dim=1).clamp_min(1.0)
        effective_option_mass_loss = (
            effective_option_shortage * coverage_weights.detach()
        ).sum() / coverage_weights.detach().sum().clamp_min(1.0)
        non_effective_option_mask = non_top_mask
        non_effective_option_weight = (
            non_effective_option_mask.to(dtype=dtype)
            * torch.maximum(pressure_weight, torch.full_like(pressure_weight, 0.25)).view(num_services, 1)
            * (0.25 + (1.0 - pair_quality).clamp(0.0, 1.0))
        ).detach()
        non_effective_option_den = non_effective_option_weight.sum().clamp_min(1.0)
        non_effective_option_loss = (
            pair_suppression_prob * non_effective_option_weight
        ).sum() / non_effective_option_den
        non_effective_option_probs = select_prob.masked_select(non_effective_option_mask)
        non_effective_selection_cost = (
            edge_prob * non_effective_option_weight
        ).sum() / non_effective_option_den
        service_quality_gap = policy_ctx.get("service_quality_gap_top_second")
        if isinstance(service_quality_gap, torch.Tensor):
            service_quality_gap = service_quality_gap.to(device=device, dtype=dtype).masked_select(has_effective_candidate)
        else:
            service_quality_gap = None
        pair_centered_logit = policy_ctx.get("pair_centered_logit")
        if isinstance(pair_centered_logit, torch.Tensor) and bool(has_edge.any().item()):
            centered_edge = pair_centered_logit.to(device=device, dtype=dtype).masked_select(edge_allowed)
            pair_centered_logit_std = centered_edge.std(unbiased=False) if centered_edge.numel() > 1 else zero
        else:
            pair_centered_logit_std = zero
        if bool(has_edge.any().item()):
            masked_prob = select_prob.masked_fill(~edge_allowed, 0.0)
            service_prob_mean = masked_prob.sum(dim=1) / edge_allowed.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
            centered_prob = torch.where(edge_allowed, select_prob - service_prob_mean.view(num_services, 1), torch.zeros_like(select_prob))
            per_service_prob_std = torch.sqrt(
                (centered_prob.pow(2).sum(dim=1) / edge_allowed.to(dtype=dtype).sum(dim=1).clamp_min(1.0)).clamp_min(0.0)
            ).masked_select(has_edge)
            per_service_prob_range = (
                select_prob.masked_fill(~edge_allowed, -1.0).max(dim=1).values
                - select_prob.masked_fill(~edge_allowed, 2.0).min(dim=1).values
            ).masked_select(has_edge)
        else:
            per_service_prob_std = None
            per_service_prob_range = None
        top_quality_logit = select_logits.masked_select(top_quality_mask)
        non_top_logit = select_logits.masked_select(non_top_mask)
        option_floor_values = option_masks["option_floor"].masked_select(has_effective_candidate)
        soft_target_values = soft_target.masked_select(edge_allowed)
        soft_target_weights = soft_target_weight.masked_select(edge_allowed)
        soft_target_gap = (soft_target - edge_prob).masked_select(edge_allowed)
        soft_target_positive_count = soft_target_positive_mask.to(dtype=dtype).sum(dim=1).masked_select(has_edge)
        soft_target_positive_prob = select_prob.masked_select(soft_target_positive_mask)
        soft_target_negative_prob = select_prob.masked_select(soft_target_negative_mask)
        return {
            "coverage_margin_loss": coverage_margin_loss,
            "quality_margin_loss": quality_margin_loss,
            "ranking_margin_loss": ranking_margin_loss,
            "contrast_margin_loss": contrast_margin_loss,
            "memory_margin_loss": memory_loss,
            "effective_option_mass_loss": effective_option_mass_loss,
            "non_effective_option_loss": non_effective_option_loss,
            "pair_centered_logit_std": pair_centered_logit_std,
            "edge_policy_prob_mean": edge_prob_mean,
            "edge_policy_prob_std": edge_prob_std,
            "edge_policy_logit_mean": edge_logit_mean,
            "edge_policy_logit_std": edge_logit_std,
            "prob_above_05_ratio": prob_above_05_ratio,
            "service_no_edge_prob_mean": feasible_no_edge.mean() if feasible_no_edge is not None else zero,
            "service_no_edge_prob_max": feasible_no_edge.max() if feasible_no_edge is not None else zero,
            "expected_edge_count_mean": feasible_edge_count.mean() if feasible_edge_count is not None else zero,
            "expected_edge_count_max": feasible_edge_count.max() if feasible_edge_count is not None else zero,
            "expected_quality_mass_mean": feasible_quality.mean() if feasible_quality is not None else zero,
            "expected_quality_mass_min": feasible_quality.min() if feasible_quality is not None else zero,
            "effective_option_mass_mean": (
                effective_option_mass.masked_select(has_effective_candidate).mean()
                if bool(has_effective_candidate.any().item()) else zero
            ),
            "desired_effective_option_mass_mean": (
                desired_effective_option_mass.masked_select(has_effective_candidate).mean()
                if bool(has_effective_candidate.any().item()) else zero
            ),
            "effective_option_shortage_mean": (
                effective_option_shortage.masked_select(has_effective_candidate).mean()
                if bool(has_effective_candidate.any().item()) else zero
            ),
            "non_effective_option_prob_mean": (
                non_effective_option_probs.mean() if non_effective_option_probs.numel() > 0 else zero
            ),
            "non_effective_selection_cost": non_effective_selection_cost,
            "contrast_margin_gap_mean": (
                contrast_gap.masked_select(contrast_services).mean()
                if bool(contrast_services.any().item()) else zero
            ),
            "expected_memory_overage_mean": memory_overage_mean,
            "expected_memory_overage_max": memory_overage_max,
            "top_quality_prob_mean": top_quality_probs.mean() if bool(top_quality_probs.numel()) else zero,
            "effective_option_prob_mean": (
                effective_option_probs.mean() if bool(effective_option_probs.numel()) else zero
            ),
            "non_top_prob_mean": non_top_probs.mean() if bool(non_top_probs.numel()) else zero,
            "top_quality_logit_mean": top_quality_logit.mean() if bool(top_quality_logit.numel()) else zero,
            "effective_option_logit_mean": (
                effective_option_logits.mean() if bool(effective_option_logits.numel()) else zero
            ),
            "non_top_logit_mean": non_top_logit.mean() if bool(non_top_logit.numel()) else zero,
            "top_quality_logit_gap_mean": (
                top_quality_logit.mean() - non_top_logit.mean()
                if bool(top_quality_logit.numel()) and bool(non_top_logit.numel()) else zero
            ),
            "effective_option_logit_gap_mean": (
                effective_option_logits.mean() - non_top_logit.mean()
                if bool(effective_option_logits.numel()) and bool(non_top_logit.numel()) else zero
            ),
            "top_quality_candidate_count_mean": (
                top_quality_count.mean() if bool(top_quality_count.numel()) else zero
            ),
            "effective_option_candidate_count_mean": (
                effective_option_count.mean() if bool(effective_option_count.numel()) else zero
            ),
            "non_top_candidate_count_mean": (
                non_top_count.mean() if bool(non_top_count.numel()) else zero
            ),
            "quality_gap_top_second_mean": (
                service_quality_gap.mean() if service_quality_gap is not None and bool(service_quality_gap.numel()) else zero
            ),
            "effective_option_floor_mean": (
                option_floor_values.mean() if bool(option_floor_values.numel()) else zero
            ),
            "soft_target_mean": (
                soft_target_values.mean() if bool(soft_target_values.numel()) else zero
            ),
            "soft_target_positive_count_mean": (
                soft_target_positive_count.mean() if bool(soft_target_positive_count.numel()) else zero
            ),
            "soft_target_weight_mean": (
                soft_target_weights.mean() if bool(soft_target_weights.numel()) else zero
            ),
            "soft_target_gap_mean": (
                soft_target_gap.mean() if bool(soft_target_gap.numel()) else zero
            ),
            "soft_target_positive_prob_mean": (
                soft_target_positive_prob.mean() if bool(soft_target_positive_prob.numel()) else zero
            ),
            "soft_target_negative_prob_mean": (
                soft_target_negative_prob.mean() if bool(soft_target_negative_prob.numel()) else zero
            ),
            "per_service_prob_std_mean": (
                per_service_prob_std.mean() if per_service_prob_std is not None else zero
            ),
            "per_service_prob_range_mean": (
                per_service_prob_range.mean() if per_service_prob_range is not None else zero
            ),
        }

    @torch.no_grad()
    def _project_deployment_mask(
            self,
            raw_deploy_mask: torch.Tensor,
            raw_probs: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            logic_edge_index: Optional[torch.Tensor] = None,
            prev_deploy_mask: Optional[torch.Tensor] = None,
            static_allowed: Optional[torch.Tensor] = None,
            retention_scores: Optional[torch.Tensor] = None,
            option_ctx: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, int, float, int, float, int, int, float, int,
        Dict[str, float], Dict[str, torch.Tensor], torch.Tensor,
    ]:
        """Apply only feasibility correction to the actor's 0/1 deployment matrix."""
        corrected = self._enforce_cloud_replica(raw_deploy_mask.bool())
        if static_allowed is None:
            static_allowed = self._static_allowed_mask(phys_feats, logic_feats)
        static_allowed = static_allowed.to(device=corrected.device).bool()
        corrected = corrected & static_allowed
        corrected = self._enforce_cloud_replica(corrected)
        if retention_scores is None:
            retention_scores = _safe_logit(raw_probs.clamp(1e-6, 1.0 - 1e-6))
        retention_scores = retention_scores.to(device=corrected.device, dtype=torch.float32)
        if option_ctx is None:
            qk_hint = torch.zeros_like(retention_scores)
            pair_ctx_hint = self._deployment_pair_context(
                logic_edge_index,
                logic_feats,
                static_allowed,
                deploy_mask=prev_deploy_mask,
            )
            option_ctx = self._deployment_runtime_context(
                logic_feats,
                phys_feats,
                qk_hint,
                static_allowed,
                pair_ctx_hint,
                prev_deploy_mask=prev_deploy_mask,
            )
        queue_pressure = option_ctx["queue_pressure"].to(device=corrected.device, dtype=torch.float32)
        runtime_risk_score = option_ctx["runtime_risk_score"].to(device=corrected.device, dtype=torch.float32)
        runtime_unknown_risk = option_ctx["runtime_unknown_risk"].to(device=corrected.device, dtype=torch.float32)
        runtime_stale_risk = option_ctx["runtime_stale_risk"].to(device=corrected.device, dtype=torch.float32)
        runtime_relative_weakness = option_ctx["runtime_relative_weakness"].to(device=corrected.device, dtype=torch.float32)
        pair_quality = option_ctx["pair_quality_score"].to(device=corrected.device, dtype=torch.float32)
        evidence_untrusted = option_ctx["evidence_untrusted"].to(device=corrected.device, dtype=torch.float32)
        low_quality_gap = option_ctx["low_quality_gap"].to(device=corrected.device, dtype=torch.float32)
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask)
        model_mem = logic_feats["model_mem"].float()
        cloud_idx = self._cloud_index(corrected.size(1))
        max_edge_replicas = self._max_edge_replicas_per_device()
        capacity_relax_cnt = 0
        capacity_relax_cost = 0.0
        capacity_removed_mask = torch.zeros_like(corrected, dtype=torch.bool)
        num_services = max(1, int(model_mem.numel()))

        if prev_deploy_mask is None:
            prev_edge_mask = torch.zeros_like(corrected)
        else:
            prev_edge_mask = prev_deploy_mask.bool()
        for device_idx in range(corrected.size(1)):
            if device_idx == cloud_idx:
                continue

            selected = torch.nonzero(corrected[:, device_idx], as_tuple=False).flatten().tolist()
            if not selected:
                continue

            total_selected_mem = sum(float(model_mem[service_idx].item()) for service_idx in selected)
            within_memory = total_selected_mem <= float(residual[device_idx].item()) + 1e-6
            within_count = max_edge_replicas is None or len(selected) <= max_edge_replicas
            if within_memory and within_count:
                continue

            keep = self._select_device_subset(
                selected=selected,
                capacity=float(residual[device_idx].item()),
                model_mem=model_mem,
                raw_probs=raw_probs[:, device_idx],
                rank_scores=retention_scores[:, device_idx],
                prev_selected=prev_edge_mask[:, device_idx],
                max_count=max_edge_replicas,
            )

            for service_idx in selected:
                if service_idx not in keep:
                    corrected[service_idx, device_idx] = False
                    capacity_removed_mask[service_idx, device_idx] = True
                    capacity_relax_cnt += 1
                    prob = float(raw_probs[service_idx, device_idx].item())
                    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
                    capacity_relax_cost += -math.log(max(1.0 - prob, 1e-6))

        capacity_relax_cost /= float(num_services)
        raw_risk_ctx = self._deployment_pair_context(
            logic_edge_index,
            logic_feats,
            static_allowed,
            deploy_mask=self._enforce_cloud_replica(raw_deploy_mask.bool()),
        )
        edge_cover_repair_cnt = 0
        edge_cover_repair_cost = 0.0
        edge_cover_unmet = 0
        hotspot_repair_cnt = 0
        hotspot_repair_cost = 0.0
        hotspot_unmet = 0
        exec_risk_ctx = self._deployment_pair_context(
            logic_edge_index,
            logic_feats,
            static_allowed,
            deploy_mask=corrected,
        )
        edge_slice = slice(0, max(0, cloud_idx))
        executed_edge = corrected[:, edge_slice].float() if cloud_idx > 0 \
            else torch.zeros((corrected.size(0), 0), device=corrected.device)
        selected_stale_option_cost = (
            executed_edge * runtime_stale_risk[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_runtime_weakness_cost = (
            executed_edge * runtime_relative_weakness[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_queue_pressure_cost = (
            executed_edge * queue_pressure[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_runtime_risk_cost = (
            executed_edge * runtime_risk_score[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_unknown_option_cost = (
            executed_edge * runtime_unknown_risk[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_low_quality_option_cost = (
            executed_edge * low_quality_gap[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_evidence_untrusted_cost = (
            executed_edge * evidence_untrusted[:, edge_slice]
        ).sum() / executed_edge.sum().clamp_min(1.0) if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_risky_pair_count = (
            executed_edge
            * (
                (runtime_risk_score[:, edge_slice] >= self._negative_runtime_risk_threshold())
                | (runtime_unknown_risk[:, edge_slice] >= self._negative_unknown_threshold())
                | (runtime_stale_risk[:, edge_slice] >= self._negative_stale_threshold())
                | (pair_quality[:, edge_slice] < self._positive_quality_threshold())
            ).float()
        ).sum() if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        selected_low_quality_pair_count = (
            executed_edge * (pair_quality[:, edge_slice] < self._positive_quality_threshold()).float()
        ).sum() if cloud_idx > 0 else torch.zeros((), device=corrected.device)
        risk_metrics = {
            "active_pair_hotspot_cost": _scalar_to_float(raw_risk_ctx["active_pair_hotspot_cost"]),
            "executed_active_pair_hotspot_cost": _scalar_to_float(exec_risk_ctx["active_pair_hotspot_cost"]),
            "service_pressure_mean": _scalar_to_float(raw_risk_ctx["service_pressure"].mean()),
            "service_pressure_max": _scalar_to_float(raw_risk_ctx["service_pressure"].max()),
            "selected_queue_pressure_cost": _scalar_to_float(selected_queue_pressure_cost),
            "selected_runtime_risk_cost": _scalar_to_float(selected_runtime_risk_cost),
            "selected_unknown_option_cost": _scalar_to_float(selected_unknown_option_cost),
            "selected_stale_option_cost": _scalar_to_float(selected_stale_option_cost),
            "selected_runtime_weakness_cost": _scalar_to_float(selected_runtime_weakness_cost),
            "selected_low_quality_option_cost": _scalar_to_float(selected_low_quality_option_cost),
            "selected_evidence_untrusted_cost": _scalar_to_float(selected_evidence_untrusted_cost),
            "selected_risky_pair_count": _scalar_to_float(selected_risky_pair_count),
            "selected_low_quality_pair_count": _scalar_to_float(selected_low_quality_pair_count),
        }
        return (
            corrected,
            capacity_relax_cnt,
            capacity_relax_cost,
            edge_cover_repair_cnt,
            edge_cover_repair_cost,
            edge_cover_unmet,
            hotspot_repair_cnt,
            hotspot_repair_cost,
            hotspot_unmet,
            risk_metrics,
            exec_risk_ctx,
            capacity_removed_mask,
        )

    def _select_device_subset(
            self,
            selected: List[int],
            capacity: float,
            model_mem: torch.Tensor,
            raw_probs: torch.Tensor,
            rank_scores: torch.Tensor,
            prev_selected: torch.Tensor,
            max_count: Optional[int] = None,
    ) -> set:
        """
        Select which sampled replicas to keep on one edge device.

        Start from the raw sampled set and prune the least-preferred replicas
        until the placement satisfies both memory and replica-count budgets.
        This is intentionally a minimum-intervention repair: the projector
        changes only what is needed for feasibility instead of searching for a
        different feasible subset with higher joint Bernoulli probability.
        """
        if capacity <= 1e-6 or not selected:
            return set()
        if max_count is not None and max_count <= 0:
            return set()

        candidates = []
        total_mem = 0.0
        for service_idx in selected:
            prob = float(raw_probs[service_idx].item())
            prob = min(max(prob, 1e-6), 1.0 - 1e-6)
            rank_score = float(rank_scores[service_idx].item())
            mem = float(model_mem[service_idx].item())
            was_selected = bool(prev_selected[service_idx].item())
            candidates.append((int(service_idx), prob, rank_score, mem, was_selected))
            total_mem += mem

        keep = {int(service_idx) for service_idx in selected}
        ranked_for_removal = sorted(
            candidates,
            key=lambda item: (
                item[2],  # Lower safety-aware preference is removed first.
                item[1],  # Then lower actor probability.
                1 if item[4] else 0,  # Prefer removing non-previous replicas on ties.
                -item[3],  # If preference is tied, remove the larger footprint first.
                -item[0],  # Keep lower service ids for deterministic ties.
            ),
        )

        for service_idx, _, _, mem, _ in ranked_for_removal:
            over_memory = total_mem > capacity + 1e-6
            over_count = max_count is not None and len(keep) > max_count
            if not over_memory and not over_count:
                break
            if service_idx not in keep:
                continue
            keep.remove(service_idx)
            total_mem -= mem

        if total_mem > capacity + 1e-6:
            return set()
        if max_count is not None and len(keep) > max_count:
            return set()
        return keep


class HedgerDeploymentPPO(_DeploymentBackbonePPO):
    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index,
               phys_feats, topo_order: Optional[list] = None,
               prev_deploy_mask: Optional[torch.Tensor] = None,
               deterministic: bool = False,
               logit_noise_std: float = 0.0):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)

        cloud_idx = self._cloud_index(Np)

        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]
        (
            q_embedding,
            k_embedding,
            qk_scores,
            qk_feature,
            pair_adjustment,
            candidate_features,
            select_logits,
            pair_ctx,
        ) = self._deployment_actor_terms(
            h_s,
            h_p,
            logic_edge_index,
            logic_feats,
            phys_feats,
            static_allowed,
            prev_deploy_mask=prev_deploy_mask,
        )
        if logit_noise_std > 0.0:
            noise = torch.randn_like(select_logits) * float(logit_noise_std)
            select_logits = select_logits + noise
            pair_ctx["select_logit"] = select_logits
            pair_ctx["select_prob"] = torch.sigmoid(select_logits)

        proposal_deploy_mask, option_debug = self._sample_matrix_deployment_mask(
            select_logits=select_logits,
            static_allowed=static_allowed,
            prev_deploy_mask=prev_deploy_mask,
            deterministic=deterministic,
        )
        raw_deploy_mask = option_debug["pre_quality_raw_mask"].to(device=h_s.device).bool()
        proposal_deploy_mask = proposal_deploy_mask.to(device=h_s.device).bool()
        if prev_deploy_mask is None:
            prev_edge = torch.zeros_like(static_allowed.bool())
        else:
            prev_edge = prev_deploy_mask.to(device=h_s.device).bool() & static_allowed.bool()
        if cloud_idx >= 0:
            prev_edge[:, cloud_idx] = False
        proposal_edge = proposal_deploy_mask.bool() & static_allowed.bool()
        if cloud_idx >= 0:
            proposal_edge[:, cloud_idx] = False
        option_debug["matrix_added_mask"] = proposal_edge & ~prev_edge
        option_debug["matrix_kept_mask"] = proposal_edge & prev_edge
        option_debug["matrix_removed_mask"] = prev_edge & ~proposal_edge
        raw_probs = torch.sigmoid(select_logits).float()
        if cloud_idx >= 0:
            device_ids = torch.arange(Np, device=h_s.device)
            cloud_mask = device_ids.view(1, -1).eq(cloud_idx).expand_as(raw_probs)
            raw_probs = torch.where(cloud_mask, torch.ones_like(raw_probs), raw_probs)
        (
            deploy_mask,
            capacity_relax_cnt,
            capacity_relax_cost,
            edge_cover_repair_cnt,
            edge_cover_repair_cost,
            edge_cover_unmet,
            hotspot_repair_cnt,
            hotspot_repair_cost,
            hotspot_unmet,
            risk_metrics,
            executed_pair_ctx,
            capacity_removed_mask,
        ) = self._project_deployment_mask(
            proposal_deploy_mask,
            raw_probs=raw_probs,
            logic_feats=logic_feats,
            phys_feats=phys_feats,
            logic_edge_index=logic_edge_index,
            prev_deploy_mask=prev_deploy_mask,
            static_allowed=static_allowed,
            retention_scores=select_logits,
            option_ctx=pair_ctx,
        )
        negative_action_mask = raw_deploy_mask & ~deploy_mask & static_allowed.bool()
        if cloud_idx >= 0:
            negative_action_mask[:, cloud_idx] = False
        effective_positive_mask = deploy_mask & static_allowed.bool()
        if cloud_idx >= 0:
            effective_positive_mask[:, cloud_idx] = False
        (
            logp_sum,
            ent_sum,
            _positive_logp,
            _negative_logp,
            positive_mask,
            negative_mask,
        ) = self._deployment_select_logp_entropy(
            select_logits,
            deploy_mask,
            static_allowed,
            topo_order=topo_order,
            positive_mask=effective_positive_mask,
            negative_mask=negative_action_mask,
        )
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        raw_zero = ((raw_deploy_mask[:, :cloud_idx].sum(dim=1) <= 0) & static_allowed[:, :cloud_idx].any(dim=1)).sum() \
            if cloud_idx > 0 else torch.tensor(0, device=h_s.device)
        matrix_added_cnt = int(option_debug["matrix_added_mask"].detach()[:, :cloud_idx].sum().cpu().item()) \
            if cloud_idx > 0 else 0
        matrix_kept_cnt = int(option_debug["matrix_kept_mask"].detach()[:, :cloud_idx].sum().cpu().item()) \
            if cloud_idx > 0 else 0
        matrix_removed_cnt = int(option_debug["matrix_removed_mask"].detach()[:, :cloud_idx].sum().cpu().item()) \
            if cloud_idx > 0 else 0
        plan_terms = self._deployment_plan_level_terms(
            select_logits,
            static_allowed,
            logic_feats,
            phys_feats,
            pair_ctx,
            prev_deploy_mask=prev_deploy_mask,
        )
        option_masks = self._deployment_soft_option_targets(
            static_allowed,
            pair_ctx,
            selected_mask=deploy_mask,
            raw_selected_mask=raw_deploy_mask,
        )
        plan_metrics = {key: _scalar_to_float(value) for key, value in plan_terms.items()}
        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt,
            "capacity_relax_cost": capacity_relax_cost,
            "edge_cover_repair_cnt": edge_cover_repair_cnt,
            "edge_cover_repair_cost": edge_cover_repair_cost,
            "edge_cover_unmet": edge_cover_unmet,
            "hotspot_repair_cnt": hotspot_repair_cnt,
            "hotspot_repair_cost": hotspot_repair_cost,
            "hotspot_unmet": hotspot_unmet,
            **risk_metrics,
            "raw_deploy_mask": raw_deploy_mask,
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
            "raw_zero_edge_services": int(raw_zero.detach().cpu().item()),
            "matrix_added_cnt": matrix_added_cnt,
            "matrix_kept_cnt": matrix_kept_cnt,
            "matrix_removed_cnt": matrix_removed_cnt,
            "matrix_raw_selected_cnt": _scalar_to_float(option_debug["matrix_raw_selected_cnt"]),
            "matrix_selected_prob_mean": _scalar_to_float(option_debug["matrix_selected_prob_mean"]),
            **plan_metrics,
            "capacity_removed_mask": capacity_removed_mask,
            "capacity_removed_cnt": int(capacity_removed_mask.detach()[:, :cloud_idx].sum().cpu().item())
            if cloud_idx > 0 else 0,
            "actor_debug": {
                "service_embedding": h_s.detach().cpu(),
                "device_embedding": h_p.detach().cpu(),
                "q_embedding": q_embedding.detach().cpu(),
                "k_embedding": k_embedding.detach().cpu(),
                "qk_score": qk_scores.detach().cpu(),
                "qk_feature": qk_feature.detach().cpu(),
                "matrix_logit_raw": pair_adjustment.detach().cpu(),
                "service_context_feature": pair_ctx["service_context_feature"].detach().cpu(),
                "service_context_feature_names": DEPLOYMENT_SERVICE_CONTEXT_FEATURE_NAMES,
                "pair_rank_logit_raw": pair_ctx["pair_rank_logit_raw"].detach().cpu(),
                "pair_centered_logit": pair_ctx["pair_centered_logit"].detach().cpu(),
                "base_score": pair_ctx["base_score"].detach().cpu(),
                "centered_score": pair_ctx["centered_score"].detach().cpu(),
                "select_logit": select_logits.detach().cpu(),
                "select_prob": torch.sigmoid(select_logits).detach().cpu(),
                "matrix_raw_selected": option_debug["matrix_raw_selected_edge"].detach().cpu(),
                "static_option_score": pair_ctx["static_option_score"].detach().cpu(),
                "static_quality_score": pair_ctx["static_quality_score"].detach().cpu(),
                "observed_quality_score": pair_ctx["observed_quality_score"].detach().cpu(),
                "runtime_risk_score": pair_ctx["runtime_risk_score"].detach().cpu(),
                "pair_quality_score": pair_ctx["pair_quality_score"].detach().cpu(),
                "service_best_pair_quality": pair_ctx["service_best_pair_quality"].detach().cpu(),
                "service_second_pair_quality": pair_ctx["service_second_pair_quality"].detach().cpu(),
                "service_quality_gap_top_second": pair_ctx["service_quality_gap_top_second"].detach().cpu(),
                "service_best_runtime_risk": pair_ctx["service_best_runtime_risk"].detach().cpu(),
                "service_max_queue_pressure": pair_ctx["service_max_queue_pressure"].detach().cpu(),
                "evidence_confidence": pair_ctx["evidence_confidence"].detach().cpu(),
                "evidence_untrusted": pair_ctx["evidence_untrusted"].detach().cpu(),
                "low_quality_gap": pair_ctx["low_quality_gap"].detach().cpu(),
                "runtime_unknown_risk": pair_ctx["runtime_unknown_risk"].detach().cpu(),
                "runtime_stale_risk": pair_ctx["runtime_stale_risk"].detach().cpu(),
                "runtime_relative_weakness": pair_ctx["runtime_relative_weakness"].detach().cpu(),
                "runtime_confidence": pair_ctx["runtime_confidence"].detach().cpu(),
                "runtime_recency": pair_ctx["runtime_recency"].detach().cpu(),
                "runtime_unknown": pair_ctx["runtime_unknown"].detach().cpu(),
                "runtime_stale": pair_ctx["runtime_stale"].detach().cpu(),
                "queue_pressure": pair_ctx["queue_pressure"].detach().cpu(),
                "candidate_feature": candidate_features.detach().cpu(),
                "candidate_feature_names": DEPLOYMENT_CANDIDATE_FEATURE_NAMES,
                "service_pressure": pair_ctx["service_pressure"].detach().cpu(),
                "edge_feasible_count": pair_ctx["edge_feasible_count"].detach().cpu(),
                "edge_replica_count": pair_ctx["edge_replica_count"].detach().cpu(),
                "device_replica_count": pair_ctx["device_replica_count"].detach().cpu(),
                "active_pair_hotspot": pair_ctx["active_pair_hotspot"].detach().cpu(),
                "executed_active_pair_hotspot": executed_pair_ctx["active_pair_hotspot"].detach().cpu(),
                "runtime_pair_feature_names": RUNTIME_PAIR_FEATURE_NAMES,
                "service_demand_feature_names": SERVICE_DEMAND_FEATURE_NAMES,
                "device_capability_feature": _debug_tensor(phys_feats, "device_capability_feat"),
                "device_capability_feature_names": DEVICE_CAPABILITY_FEATURE_NAMES,
                "final_score": select_logits.detach().cpu(),
                "policy_prob": option_debug["select_prob"].detach().cpu(),
                "static_mask": static_allowed.detach().cpu(),
                "raw_mode_mask": raw_deploy_mask.detach().cpu(),
                "matrix_added_mask": option_debug["matrix_added_mask"].detach().cpu(),
                "matrix_kept_mask": option_debug["matrix_kept_mask"].detach().cpu(),
                "matrix_removed_mask": option_debug["matrix_removed_mask"].detach().cpu(),
                "capacity_removed_mask": capacity_removed_mask.detach().cpu(),
                "effective_option_mask": option_masks["effective_option_mask"].detach().cpu(),
                "top_quality_option_mask": option_masks["top_quality_mask"].detach().cpu(),
                "clear_non_effective_option_mask": option_masks["clear_non_effective_mask"].detach().cpu(),
                "risky_option_mask": option_masks["risky_option_mask"].detach().cpu(),
                "effective_option_floor": option_masks["option_floor"].detach().cpu(),
                "soft_option_target": option_masks["soft_target"].detach().cpu(),
                "soft_option_target_weight": option_masks["soft_target_weight"].detach().cpu(),
                "soft_option_positive_mask": option_masks["soft_target_positive_mask"].detach().cpu(),
                "soft_option_negative_mask": option_masks["soft_target_negative_mask"].detach().cpu(),
                "soft_option_target_floor": option_masks["soft_target_floor"].detach().cpu(),
                "soft_option_target_margin": option_masks["soft_target_margin"].detach().cpu(),
                "positive_mask": positive_mask.detach().cpu(),
                "negative_mask": negative_mask.detach().cpu(),
            },
        }

    def evaluate(
            self,
            logic_edge_index,
            logic_feats,
            phys_edge_index,
            phys_feats,
            deploy_mask: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor] = None,
            topo_order: Optional[list] = None,
            return_policy: bool = False,
            positive_mask: Optional[torch.Tensor] = None,
            negative_mask: Optional[torch.Tensor] = None,
    ):
        """
        Evaluate `deploy_mask` under the current parameters.

        PPO is evaluated against the executed post-projection deployment mask.
        The policy still samples a raw Bernoulli mask and projects it before
        execution; training follows the action that actually touched the system.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)

        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]
        cloud_idx = self._cloud_index(Np)
        deploy_mask = self._enforce_cloud_replica(deploy_mask.bool())
        (
            _q_embedding,
            _k_embedding,
            _qk_scores,
            _qk_feature,
            _pair_adjustment,
            candidate_features,
            select_logits,
            pair_ctx,
        ) = self._deployment_actor_terms(
            h_s,
            h_p,
            logic_edge_index,
            logic_feats,
            phys_feats,
            static_allowed,
            prev_deploy_mask=prev_deploy_mask,
        )
        select_prob = torch.where(static_allowed, torch.sigmoid(select_logits), torch.zeros_like(select_logits)).float()
        if cloud_idx >= 0:
            device_ids = torch.arange(Np, device=h_s.device)
            cloud_mask = device_ids.view(1, -1).eq(cloud_idx).expand_as(static_allowed)
            select_prob = torch.where(cloud_mask, torch.ones_like(select_prob), select_prob)
        logp_sum, ent_sum, positive_logp, negative_logp, positive_mask_t, negative_mask_t = (
            self._deployment_select_logp_entropy(
                select_logits,
                deploy_mask,
                static_allowed,
                topo_order=topo_order,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
            )
        )
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        if return_policy:
            return logp_sum, ent_sum, value.squeeze(0), {
                "policy_prob": select_prob,
                "select_prob": select_prob,
                "select_logit": select_logits,
                "static_allowed": static_allowed,
                "positive_logp": positive_logp,
                "negative_logp": negative_logp,
                "positive_mask": positive_mask_t,
                "negative_mask": negative_mask_t,
                "pair_rank_logit_raw": pair_ctx["pair_rank_logit_raw"],
                "pair_centered_logit": pair_ctx["pair_centered_logit"],
                "service_pressure": pair_ctx["service_pressure"],
                "queue_pressure": pair_ctx["queue_pressure"],
                "active_pair_hotspot": pair_ctx["active_pair_hotspot"],
                "static_option_score": pair_ctx["static_option_score"],
                "static_quality_score": pair_ctx["static_quality_score"],
                "observed_quality_score": pair_ctx["observed_quality_score"],
                "runtime_risk_score": pair_ctx["runtime_risk_score"],
                "pair_quality_score": pair_ctx["pair_quality_score"],
                "service_quality_gap_top_second": pair_ctx["service_quality_gap_top_second"],
                "evidence_confidence": pair_ctx["evidence_confidence"],
                "evidence_untrusted": pair_ctx["evidence_untrusted"],
                "low_quality_gap": pair_ctx["low_quality_gap"],
                "runtime_unknown_risk": pair_ctx["runtime_unknown_risk"],
                "runtime_stale_risk": pair_ctx["runtime_stale_risk"],
                "runtime_unknown": pair_ctx["runtime_unknown"],
                "runtime_stale": pair_ctx["runtime_stale"],
                "runtime_recency": pair_ctx["runtime_recency"],
                "runtime_relative_weakness": pair_ctx["runtime_relative_weakness"],
                "pair_memory_cost": pair_ctx["pair_memory_cost"],
                "edge_replica_count": pair_ctx["edge_replica_count"],
                "device_replica_count": pair_ctx["device_replica_count"],
            }
        return logp_sum, ent_sum, value.squeeze(0)

    def ppo_update(self, transitions: List[dict], epochs=4, batch_size=16, clip_eps=None, entropy_coef=0.01,
                   value_coef=0.5):
        clip_eps = self.clip_eps if clip_eps is None else clip_eps
        device = next(self.parameters()).device
        old_logp = torch.stack([t['logp'].detach().to(device) for t in transitions])
        old_val = [_scalar_to_float(t['value']) for t in transitions]
        rewards = [float(t['reward']) for t in transitions]
        dones = [t['done'] for t in transitions]
        last_value = 0.0 if dones[-1] else _scalar_to_float(transitions[-1].get("next_value", 0.0))
        adv, rets = compute_returns_advantages(rewards, old_val, dones, self.gamma, self.lamda, last_value=last_value)
        adv_raw = torch.tensor(adv, device=device)
        rets = torch.tensor(rets, device=device)
        adv_raw_mean = adv_raw.mean()
        adv_raw_std = adv_raw.std(unbiased=False)
        adv = (adv_raw - adv_raw_mean) / (adv_raw_std + 1e-6)
        device_transitions = [
            {
                "logic_edge_index": tr["logic_edge_index"].to(device),
                "logic_feats": _move_tensor_dict_to_device(tr["logic_feats"], device),
                "phys_edge_index": tr["phys_edge_index"].to(device),
                "phys_feats": _move_tensor_dict_to_device(tr["phys_feats"], device),
                "deploy_mask": tr["deploy_mask"].to(device),
                "positive_mask": tr["positive_mask"].to(device) if tr.get("positive_mask") is not None else None,
                "negative_mask": tr["negative_mask"].to(device) if tr.get("negative_mask") is not None else None,
                "prev_deploy_mask": (
                    tr["prev_deploy_mask"].to(device) if tr.get("prev_deploy_mask") is not None else None
                ),
                "topo_order": tr.get("topo_order"),
            }
            for tr in transitions
        ]
        T = len(transitions)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []
        ratio_means: List[float] = []
        ratio_stds: List[float] = []
        actor_grad_norms: List[float] = []
        critic_grad_norms: List[float] = []
        new_value_means: List[float] = []
        for _ in range(epochs):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, batch_size):
                idx = perm[start:start + batch_size]
                new_logp_list = []
                new_val_list = []
                ent_list = []
                for j in idx:
                    tr = device_transitions[int(j)]
                    lp, ent, val = self.evaluate(
                        tr['logic_edge_index'],
                        tr['logic_feats'],
                        tr['phys_edge_index'],
                        tr['phys_feats'],
                        tr['deploy_mask'],
                        prev_deploy_mask=tr['prev_deploy_mask'],
                        topo_order=tr['topo_order'],
                        positive_mask=tr.get("positive_mask"),
                        negative_mask=tr.get("negative_mask"),
                    )
                    new_logp_list.append(lp)
                    new_val_list.append(val)
                    ent_list.append(ent)
                new_logp = torch.stack(new_logp_list)
                new_val = torch.stack(new_val_list).squeeze(-1)
                ent = torch.stack(ent_list).mean()
                ratio = torch.exp(new_logp - old_logp[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[idx]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(new_val, rets[idx])
                loss = policy_loss + value_coef * value_loss - entropy_coef * ent

                with torch.no_grad():
                    approx_kl = (old_logp[idx] - new_logp).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
                    policy_losses.append(_scalar_to_float(policy_loss))
                    value_losses.append(_scalar_to_float(value_loss))
                    entropies.append(_scalar_to_float(ent))
                    approx_kls.append(_scalar_to_float(approx_kl))
                    clip_fractions.append(_scalar_to_float(clip_fraction))
                    ratio_means.append(_scalar_to_float(ratio.mean()))
                    ratio_stds.append(_tensor_std_float(ratio))
                    new_value_means.append(_scalar_to_float(new_val.mean()))

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(self._actor_train_params, 1.0)
                actor_grad_norms.append(_scalar_to_float(actor_grad_norm))
                critic_grad_norms.append(_parameters_grad_norm(self.critic.parameters()))
                self.actor_opt.step()
                self.critic_opt.step()

        return {
            "samples": T,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "minibatches": len(policy_losses),
            "reward_mean": _mean_or_zero(rewards),
            "reward_std": _std_or_zero(rewards),
            "reward_min": float(min(rewards)) if rewards else 0.0,
            "reward_max": float(max(rewards)) if rewards else 0.0,
            "value_old_mean": _mean_or_zero(old_val),
            "value_old_std": _std_or_zero(old_val),
            "value_new_mean": _mean_or_zero(new_value_means),
            "return_mean": _scalar_to_float(rets.mean()),
            "return_std": _tensor_std_float(rets),
            "adv_mean": _scalar_to_float(adv_raw_mean),
            "adv_std": _scalar_to_float(adv_raw_std),
            "last_value": float(last_value),
            "done_fraction": _mean_or_zero([1.0 if done else 0.0 for done in dones]),
            "policy_loss": _mean_or_zero(policy_losses),
            "value_loss": _mean_or_zero(value_losses),
            "entropy": _mean_or_zero(entropies),
            "entropy_coef": float(entropy_coef),
            "value_coef": float(value_coef),
            "approx_kl": _mean_or_zero(approx_kls),
            "clip_fraction": _mean_or_zero(clip_fractions),
            "ratio_mean": _mean_or_zero(ratio_means),
            "ratio_std": _mean_or_zero(ratio_stds),
            "actor_grad_norm": _mean_or_zero(actor_grad_norms),
            "critic_grad_norm": _mean_or_zero(critic_grad_norms),
        }

    def _deployment_offline_actor_masks(
            self,
            deploy_mask: torch.Tensor,
            raw_deploy_mask: Optional[torch.Tensor],
            policy_aux: Dict[str, torch.Tensor],
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor],
            quality_bucket: str,
            top_quality_tolerance: float = 0.05,
            option_quality_ratio: Optional[float] = None,
            coverage_pressure_floor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        static_allowed = policy_aux["static_allowed"].bool()
        num_devices = static_allowed.size(1)
        cloud_idx = self._cloud_index(num_devices)
        edge_allowed = static_allowed.clone()
        if cloud_idx >= 0:
            edge_allowed[:, cloud_idx] = False
        selected = deploy_mask.to(device=edge_allowed.device).bool() & edge_allowed
        if raw_deploy_mask is None:
            raw_selected = selected
        else:
            raw_selected = raw_deploy_mask.to(device=edge_allowed.device).bool() & edge_allowed
        queue_pressure = policy_aux.get("queue_pressure")
        hotspot = policy_aux.get("active_pair_hotspot")
        runtime_risk = policy_aux.get("runtime_risk_score")
        runtime_unknown = policy_aux.get("runtime_unknown_risk")
        runtime_stale = policy_aux.get("runtime_stale_risk")
        pair_quality = policy_aux.get("pair_quality_score")
        if not isinstance(queue_pressure, torch.Tensor):
            queue_pressure = torch.zeros_like(static_allowed, dtype=torch.float32)
        if not isinstance(hotspot, torch.Tensor):
            hotspot = torch.zeros_like(static_allowed, dtype=torch.float32)
        if not isinstance(runtime_risk, torch.Tensor):
            runtime_risk = torch.zeros_like(static_allowed, dtype=torch.float32)
        if not isinstance(runtime_unknown, torch.Tensor):
            runtime_unknown = torch.zeros_like(static_allowed, dtype=torch.float32)
        if not isinstance(runtime_stale, torch.Tensor):
            runtime_stale = torch.zeros_like(static_allowed, dtype=torch.float32)
        if not isinstance(pair_quality, torch.Tensor):
            pair_quality = torch.ones_like(static_allowed, dtype=torch.float32)

        queue_pressure = queue_pressure.to(device=edge_allowed.device, dtype=torch.float32)
        hotspot = hotspot.to(device=edge_allowed.device, dtype=torch.float32)
        runtime_risk = runtime_risk.to(device=edge_allowed.device, dtype=torch.float32)
        runtime_unknown = runtime_unknown.to(device=edge_allowed.device, dtype=torch.float32)
        runtime_stale = runtime_stale.to(device=edge_allowed.device, dtype=torch.float32)
        pair_quality = pair_quality.to(device=edge_allowed.device, dtype=torch.float32)

        queue_risky = queue_pressure >= self._negative_queue_threshold()
        hotspot_risky = hotspot >= self._negative_hotspot_threshold()
        runtime_risky = runtime_risk >= self._negative_runtime_risk_threshold()
        unknown_risky = runtime_unknown >= self._negative_unknown_threshold()
        stale_risky = runtime_stale >= self._negative_stale_threshold()
        low_quality = pair_quality < self._positive_quality_threshold()
        severe_risky = (queue_risky | hotspot_risky | runtime_risky) & edge_allowed
        evidence_risky = (unknown_risky | stale_risky) & edge_allowed
        diagnostic_risky = (severe_risky | evidence_risky | low_quality) & edge_allowed
        option_masks = self._deployment_effective_option_masks(
            static_allowed,
            policy_aux,
            top_quality_tolerance=top_quality_tolerance,
            option_quality_ratio=option_quality_ratio,
            coverage_pressure_floor=coverage_pressure_floor,
        )
        quality_eligible = option_masks["quality_eligible"]
        top_quality_mask = option_masks["top_quality_mask"]
        effective_option_mask = option_masks["effective_option_mask"]
        clear_non_effective_mask = option_masks["clear_non_effective_mask"]
        raw_removed = raw_selected & ~selected
        selected_effective = selected & effective_option_mask
        selected_safe_effective = selected_effective & quality_eligible & ~evidence_risky
        teacher_positive = top_quality_mask & edge_allowed
        clear_unselected_negative = (
            clear_non_effective_mask
            & ~selected
            & ~raw_removed
            & edge_allowed
        )
        debug_counts = {
            "actor_selected_risky_samples": float((selected & diagnostic_risky).float().sum().detach().cpu().item()),
            "actor_selected_low_quality_samples": float((selected & low_quality & edge_allowed).float().sum().detach().cpu().item()),
            "actor_selected_runtime_risky_samples": float((selected & runtime_risky & edge_allowed).float().sum().detach().cpu().item()),
            "actor_selected_unknown_samples": float((selected & unknown_risky & edge_allowed).float().sum().detach().cpu().item()),
            "actor_selected_stale_samples": float((selected & stale_risky & edge_allowed).float().sum().detach().cpu().item()),
            "actor_effective_target_samples": float(selected_effective.float().sum().detach().cpu().item()),
            "actor_teacher_positive_samples": float(teacher_positive.float().sum().detach().cpu().item()),
            "actor_selected_effective_samples": float(selected_effective.float().sum().detach().cpu().item()),
            "actor_clear_non_effective_samples": float(clear_non_effective_mask.float().sum().detach().cpu().item()),
        }
        quality = str(quality_bucket or "unknown").strip().lower()
        # Positive labels are now service-local effective options, not the
        # whole executed deployment.  This keeps the Bernoulli matrix literal:
        # a service's best effective edge candidates are trained above the
        # p=0.5 boundary, while bad transitions still avoid cloning the
        # executed edge set.
        positive_mask = (
            (teacher_positive | selected_safe_effective)
            if quality != "bad" else torch.zeros_like(selected)
        )
        aux_positive_mask = torch.zeros_like(selected)
        if quality != "bad":
            aux_positive_mask = selected_effective & ~positive_mask
        negative_mask = (selected & (severe_risky | evidence_risky | low_quality) & ~positive_mask) & edge_allowed
        if quality == "bad":
            negative_mask = negative_mask | (selected & edge_allowed)
        unselected_negative_mask = (
            edge_allowed
            & ~positive_mask
            & ~raw_removed
            & ~selected
            & (severe_risky | evidence_risky | clear_unselected_negative)
        )
        return positive_mask, negative_mask, raw_removed, unselected_negative_mask, aux_positive_mask, debug_counts

    def offline_update(
            self,
            transitions: List[dict],
            batch_size: Optional[int] = None,
            action_target: str = "executed",
            advantage_temperature: float = 1.0,
            min_advantage_weight: float = 0.0,
            max_advantage_weight: float = 20.0,
            actor_bc_coef: float = 1.0,
            executed_aux_positive_coef: float = 0.20,
            negative_bc_coef: float = 0.2,
            raw_removed_negative_coef: float = 0.0,
            unselected_negative_coef: float = 0.0,
            value_coef: float = 0.5,
            entropy_coef: float = 0.0,
            bootstrap_current_value: bool = True,
            coverage_margin_coef: float = 0.85,
            quality_margin_coef: float = 0.55,
            ranking_margin_coef: float = 0.45,
            contrast_margin_coef: float = 0.75,
            memory_margin_coef: float = 0.18,
            effective_option_mass_coef: float = 0.25,
            non_effective_option_coef: float = 0.25,
            soft_target_bc_coef: float = 0.85,
            positive_logit_margin: float = 0.30,
            negative_logit_margin: float = 0.25,
            coverage_logit_margin: float = 0.20,
            ranking_logit_margin: float = 0.15,
            contrast_logit_margin: float = 0.30,
            top_quality_tolerance: float = 0.05,
            coverage_pressure_floor: float = 0.35,
            option_quality_ratio: Optional[float] = None,
    ):
        """
        Offline/replay update for deployment macro-transitions.

        This is an AWAC-style actor-critic step: fit the critic to one-step
        bootstrapped returns and calibrate the service-device Bernoulli matrix
        directly.  Positive labels are service-local top effective edge
        options plus safe executed effective replicas; bad transitions
        suppress risky executed bits instead of cloning them.
        """
        if not transitions:
            return None

        target_name = str(action_target or "executed").strip().lower()
        if target_name not in {"executed", "raw"}:
            raise ValueError("deployment offline action_target must be one of: executed, raw.")

        device = next(self.parameters()).device
        batch = transitions if batch_size is None else transitions[:max(1, int(batch_size))]
        values = []
        positive_logps = []
        negative_logps = []
        aux_positive_logps = []
        raw_removed_logps = []
        unselected_negative_logps = []
        entropies = []
        targets = []
        rewards = []
        quality_values = []
        select_logits_for_bce = []
        pair_negative_logits_for_bce = []
        soft_target_logits_for_bce = []
        soft_targets_for_bce = []
        soft_target_weights_for_bce = []
        positive_masks_for_bce = []
        negative_masks_for_bce = []
        aux_positive_masks_for_bce = []
        raw_removed_masks_for_bce = []
        unselected_negative_masks_for_bce = []
        positive_counts = []
        negative_counts = []
        aux_positive_counts = []
        raw_removed_counts = []
        unselected_negative_counts = []
        selected_risky_counts = []
        selected_low_quality_counts = []
        selected_runtime_risky_counts = []
        selected_unknown_counts = []
        selected_stale_counts = []
        teacher_positive_counts = []
        selected_effective_counts = []
        clear_non_effective_counts = []
        positive_prob_means = []
        negative_prob_means = []
        aux_positive_prob_means = []
        raw_removed_prob_means = []
        unselected_negative_prob_means = []
        soft_target_means = []
        soft_target_positive_counts = []
        soft_target_weight_means = []
        soft_target_gap_means = []
        soft_target_positive_prob_means = []
        soft_target_negative_prob_means = []
        positive_logit_means = []
        negative_logit_means = []
        aux_positive_logit_means = []
        raw_removed_logit_means = []
        unselected_negative_logit_means = []
        edge_prob_means = []
        edge_prob_stds = []
        edge_logit_means = []
        edge_logit_stds = []
        raw_mode_edge_densities = []
        coverage_margin_losses = []
        quality_margin_losses = []
        ranking_margin_losses = []
        contrast_margin_losses = []
        memory_margin_losses = []
        effective_option_mass_losses = []
        non_effective_option_losses = []
        pair_centered_logit_stds = []
        service_no_edge_prob_means = []
        service_no_edge_prob_maxes = []
        expected_edge_count_means = []
        expected_edge_count_maxes = []
        expected_quality_mass_means = []
        expected_quality_mass_mins = []
        effective_option_mass_means = []
        desired_effective_option_mass_means = []
        effective_option_shortage_means = []
        non_effective_option_prob_means = []
        non_effective_selection_costs = []
        expected_memory_overage_means = []
        expected_memory_overage_maxes = []
        top_quality_prob_means = []
        effective_option_prob_means = []
        non_top_prob_means = []
        top_quality_logit_means = []
        effective_option_logit_means = []
        non_top_logit_means = []
        top_quality_logit_gap_means = []
        effective_option_logit_gap_means = []
        top_quality_candidate_count_means = []
        effective_option_candidate_count_means = []
        non_top_candidate_count_means = []
        quality_gap_top_second_means = []
        effective_option_floor_means = []
        contrast_margin_gap_means = []
        per_service_prob_std_means = []
        per_service_prob_range_means = []
        prob_above_05_ratios = []
        for tr in batch:
            logic_edge_index = tr["logic_edge_index"].to(device)
            logic_feats = _move_tensor_dict_to_device(tr["logic_feats"], device)
            phys_edge_index = tr["phys_edge_index"].to(device)
            phys_feats = _move_tensor_dict_to_device(tr["phys_feats"], device)
            prev_deploy_mask = (
                tr["prev_deploy_mask"].to(device) if tr.get("prev_deploy_mask") is not None else None
            )
            action_key = "raw_deploy_mask" if target_name == "raw" else "deploy_mask"
            deploy_mask = tr.get(action_key, tr["deploy_mask"]).to(device).bool()
            raw_deploy_mask = tr.get("raw_deploy_mask")
            raw_deploy_mask = raw_deploy_mask.to(device).bool() if raw_deploy_mask is not None else None
            quality_bucket = transition_quality_bucket(tr)
            with torch.no_grad():
                _, _, _, probe_aux = self.evaluate(
                    logic_edge_index,
                    logic_feats,
                    phys_edge_index,
                    phys_feats,
                    deploy_mask,
                    prev_deploy_mask=prev_deploy_mask,
                    topo_order=tr.get("topo_order"),
                    return_policy=True,
                )
                (
                    positive_mask,
                    negative_mask,
                    raw_removed_mask,
                    unselected_negative_mask,
                    aux_positive_mask,
                    mask_debug,
                ) = self._deployment_offline_actor_masks(
                    deploy_mask,
                    raw_deploy_mask,
                    probe_aux,
                    logic_feats,
                    phys_feats,
                    prev_deploy_mask,
                    quality_bucket,
                    top_quality_tolerance=float(top_quality_tolerance),
                    option_quality_ratio=option_quality_ratio,
                    coverage_pressure_floor=float(coverage_pressure_floor),
                )
                soft_target_info = self._deployment_soft_option_targets(
                    probe_aux["static_allowed"].to(device).bool(),
                    probe_aux,
                    selected_mask=deploy_mask,
                    raw_selected_mask=raw_deploy_mask,
                    quality_bucket=quality_bucket,
                    top_quality_tolerance=float(top_quality_tolerance),
                    option_quality_ratio=option_quality_ratio,
                    coverage_pressure_floor=float(coverage_pressure_floor),
                )
            _logp, ent, _actor_value, policy_aux = self.evaluate(
                logic_edge_index,
                logic_feats,
                phys_edge_index,
                phys_feats,
                deploy_mask,
                prev_deploy_mask=prev_deploy_mask,
                topo_order=tr.get("topo_order"),
                return_policy=True,
                positive_mask=positive_mask,
                negative_mask=negative_mask,
            )
            _, _, critic_value = self.evaluate(
                logic_edge_index,
                logic_feats,
                phys_edge_index,
                phys_feats,
                deploy_mask,
                prev_deploy_mask=prev_deploy_mask,
                topo_order=tr.get("topo_order"),
                return_policy=False,
            )
            plan_terms = self._deployment_plan_level_terms(
                policy_aux["select_logit"],
                policy_aux["static_allowed"],
                logic_feats,
                phys_feats,
                policy_aux,
                prev_deploy_mask=prev_deploy_mask,
                positive_logit_margin=float(positive_logit_margin),
                coverage_logit_margin=float(coverage_logit_margin),
                ranking_logit_margin=float(ranking_logit_margin),
                contrast_logit_margin=float(contrast_logit_margin),
                top_quality_tolerance=float(top_quality_tolerance),
                coverage_pressure_floor=float(coverage_pressure_floor),
                option_quality_ratio=option_quality_ratio,
            )
            raw_removed_mask = raw_removed_mask.to(device=policy_aux["policy_prob"].device).bool()
            unselected_negative_mask = unselected_negative_mask.to(
                device=policy_aux["policy_prob"].device
            ).bool()
            raw_removed_probs = policy_aux["policy_prob"].clamp(1e-6, 1.0 - 1e-6)
            raw_removed_logits = policy_aux["select_logit"].to(device=raw_removed_probs.device, dtype=torch.float32)
            pair_negative_logits = raw_removed_logits
            positive_mask_t = policy_aux["positive_mask"].to(device=raw_removed_probs.device).bool()
            negative_mask_t = policy_aux["negative_mask"].to(device=raw_removed_probs.device).bool()
            aux_positive_mask_t = aux_positive_mask.to(device=raw_removed_probs.device).bool()
            edge_allowed_t = policy_aux["static_allowed"].to(device=raw_removed_probs.device).bool().clone()
            cloud_idx = self._cloud_index(edge_allowed_t.size(1))
            if cloud_idx >= 0:
                edge_allowed_t[:, cloud_idx] = False
            select_logits_for_bce.append(raw_removed_logits)
            pair_negative_logits_for_bce.append(pair_negative_logits)
            soft_target_logits_for_bce.append(raw_removed_logits)
            soft_targets_for_bce.append(soft_target_info["soft_target"].to(
                device=raw_removed_logits.device,
                dtype=torch.float32,
            ).detach())
            soft_target_weights_for_bce.append(soft_target_info["soft_target_weight"].to(
                device=raw_removed_logits.device,
                dtype=torch.float32,
            ).detach())
            positive_masks_for_bce.append(positive_mask_t.detach())
            negative_masks_for_bce.append(negative_mask_t.detach())
            aux_positive_masks_for_bce.append((aux_positive_mask_t & edge_allowed_t & ~positive_mask_t).detach())
            raw_removed_masks_for_bce.append((raw_removed_mask & edge_allowed_t).detach())
            unselected_negative_masks_for_bce.append((unselected_negative_mask & edge_allowed_t).detach())

            if bool(raw_removed_mask.any().item()):
                raw_removed_logp = (
                    torch.log1p(-raw_removed_probs).masked_select(raw_removed_mask).sum()
                    / raw_removed_mask.float().sum().clamp_min(1.0)
                )
            else:
                raw_removed_logp = torch.tensor(0.0, device=device)
            if bool(unselected_negative_mask.any().item()):
                unselected_negative_logp = (
                    torch.log1p(-raw_removed_probs).masked_select(unselected_negative_mask).sum()
                    / unselected_negative_mask.float().sum().clamp_min(1.0)
                )
            else:
                unselected_negative_logp = torch.tensor(0.0, device=device)

            reward = float(tr["reward"])
            rewards.append(reward)
            done = bool(tr.get("done", False))
            next_value = float(tr.get("next_value", 0.0))
            if bootstrap_current_value and not done \
                    and tr.get("next_logic_feats") is not None and tr.get("next_phys_feats") is not None:
                with torch.no_grad():
                    next_value = _scalar_to_float(
                        self.estimate_value(
                            logic_edge_index=logic_edge_index,
                            logic_feats=_move_tensor_dict_to_device(tr["next_logic_feats"], device),
                            phys_edge_index=phys_edge_index,
                            phys_feats=_move_tensor_dict_to_device(tr["next_phys_feats"], device),
                            prev_deploy_mask=deploy_mask,
                        )
                    )
            targets.append(reward + self.gamma * next_value * (0.0 if done else 1.0))
            values.append(critic_value.squeeze())
            positive_logps.append(policy_aux["positive_logp"].detach())
            negative_logps.append(policy_aux["negative_logp"].detach())
            if bool(aux_positive_mask_t.any().item()):
                aux_positive_logp = (
                    torch.log(raw_removed_probs).masked_select(aux_positive_mask_t).sum()
                    / aux_positive_mask_t.float().sum().clamp_min(1.0)
                )
            else:
                aux_positive_logp = torch.tensor(0.0, device=device)
            aux_positive_logps.append(aux_positive_logp.detach())
            raw_removed_logps.append(raw_removed_logp.detach())
            unselected_negative_logps.append(unselected_negative_logp.detach())
            entropies.append(ent)
            quality_values.append(quality_bucket)
            positive_counts.append(float(policy_aux["positive_mask"].float().sum().detach().cpu().item()))
            negative_counts.append(float(policy_aux["negative_mask"].float().sum().detach().cpu().item()))
            aux_positive_counts.append(float(aux_positive_mask_t.float().sum().detach().cpu().item()))
            raw_removed_counts.append(float(raw_removed_mask.float().sum().detach().cpu().item()))
            unselected_negative_counts.append(
                float(unselected_negative_mask.float().sum().detach().cpu().item())
            )
            selected_risky_counts.append(float(mask_debug.get("actor_selected_risky_samples", 0.0)))
            selected_low_quality_counts.append(float(mask_debug.get("actor_selected_low_quality_samples", 0.0)))
            selected_runtime_risky_counts.append(float(mask_debug.get("actor_selected_runtime_risky_samples", 0.0)))
            selected_unknown_counts.append(float(mask_debug.get("actor_selected_unknown_samples", 0.0)))
            selected_stale_counts.append(float(mask_debug.get("actor_selected_stale_samples", 0.0)))
            teacher_positive_counts.append(float(mask_debug.get("actor_teacher_positive_samples", 0.0)))
            selected_effective_counts.append(float(mask_debug.get("actor_selected_effective_samples", 0.0)))
            clear_non_effective_counts.append(float(mask_debug.get("actor_clear_non_effective_samples", 0.0)))
            if bool(positive_mask_t.any().item()):
                positive_prob_means.append(_scalar_to_float(raw_removed_probs.masked_select(positive_mask_t).mean()))
                positive_logit_means.append(_scalar_to_float(raw_removed_logits.masked_select(positive_mask_t).mean()))
            if bool(negative_mask_t.any().item()):
                negative_prob_means.append(_scalar_to_float(raw_removed_probs.masked_select(negative_mask_t).mean()))
                negative_logit_means.append(_scalar_to_float(raw_removed_logits.masked_select(negative_mask_t).mean()))
            if bool(aux_positive_mask_t.any().item()):
                aux_positive_prob_means.append(
                    _scalar_to_float(raw_removed_probs.masked_select(aux_positive_mask_t).mean())
                )
                aux_positive_logit_means.append(
                    _scalar_to_float(raw_removed_logits.masked_select(aux_positive_mask_t).mean())
                )
            if bool(raw_removed_mask.any().item()):
                raw_removed_prob_means.append(_scalar_to_float(raw_removed_probs.masked_select(raw_removed_mask).mean()))
                raw_removed_logit_means.append(_scalar_to_float(raw_removed_logits.masked_select(raw_removed_mask).mean()))
            if bool(unselected_negative_mask.any().item()):
                unselected_negative_prob_means.append(
                    _scalar_to_float(raw_removed_probs.masked_select(unselected_negative_mask).mean())
                )
                unselected_negative_logit_means.append(
                    _scalar_to_float(raw_removed_logits.masked_select(unselected_negative_mask).mean())
                )
            if bool(edge_allowed_t.any().item()):
                edge_probs = raw_removed_probs.masked_select(edge_allowed_t)
                edge_logits = raw_removed_logits.masked_select(edge_allowed_t)
                edge_prob_means.append(_scalar_to_float(edge_probs.mean()))
                edge_prob_stds.append(_tensor_std_float(edge_probs))
                edge_logit_means.append(_scalar_to_float(edge_logits.mean()))
                edge_logit_stds.append(_tensor_std_float(edge_logits))
                raw_mode_edge_densities.append(
                    _scalar_to_float(((raw_removed_logits > 0.0) & edge_allowed_t).float().sum()
                                     / edge_allowed_t.float().sum().clamp_min(1.0))
                )
            coverage_margin_losses.append(plan_terms["coverage_margin_loss"])
            quality_margin_losses.append(plan_terms["quality_margin_loss"])
            ranking_margin_losses.append(plan_terms["ranking_margin_loss"])
            contrast_margin_losses.append(plan_terms["contrast_margin_loss"])
            memory_margin_losses.append(plan_terms["memory_margin_loss"])
            effective_option_mass_losses.append(plan_terms["effective_option_mass_loss"])
            non_effective_option_losses.append(plan_terms["non_effective_option_loss"])
            pair_centered_logit_stds.append(_scalar_to_float(plan_terms["pair_centered_logit_std"]))
            service_no_edge_prob_means.append(_scalar_to_float(plan_terms["service_no_edge_prob_mean"]))
            service_no_edge_prob_maxes.append(_scalar_to_float(plan_terms["service_no_edge_prob_max"]))
            expected_edge_count_means.append(_scalar_to_float(plan_terms["expected_edge_count_mean"]))
            expected_edge_count_maxes.append(_scalar_to_float(plan_terms["expected_edge_count_max"]))
            expected_quality_mass_means.append(_scalar_to_float(plan_terms["expected_quality_mass_mean"]))
            expected_quality_mass_mins.append(_scalar_to_float(plan_terms["expected_quality_mass_min"]))
            effective_option_mass_means.append(_scalar_to_float(plan_terms["effective_option_mass_mean"]))
            desired_effective_option_mass_means.append(_scalar_to_float(
                plan_terms["desired_effective_option_mass_mean"]
            ))
            effective_option_shortage_means.append(_scalar_to_float(plan_terms["effective_option_shortage_mean"]))
            non_effective_option_prob_means.append(_scalar_to_float(plan_terms["non_effective_option_prob_mean"]))
            non_effective_selection_costs.append(_scalar_to_float(plan_terms["non_effective_selection_cost"]))
            expected_memory_overage_means.append(_scalar_to_float(plan_terms["expected_memory_overage_mean"]))
            expected_memory_overage_maxes.append(_scalar_to_float(plan_terms["expected_memory_overage_max"]))
            top_quality_prob_means.append(_scalar_to_float(plan_terms["top_quality_prob_mean"]))
            effective_option_prob_means.append(_scalar_to_float(plan_terms["effective_option_prob_mean"]))
            non_top_prob_means.append(_scalar_to_float(plan_terms["non_top_prob_mean"]))
            top_quality_logit_means.append(_scalar_to_float(plan_terms["top_quality_logit_mean"]))
            effective_option_logit_means.append(_scalar_to_float(plan_terms["effective_option_logit_mean"]))
            non_top_logit_means.append(_scalar_to_float(plan_terms["non_top_logit_mean"]))
            top_quality_logit_gap_means.append(_scalar_to_float(plan_terms["top_quality_logit_gap_mean"]))
            effective_option_logit_gap_means.append(_scalar_to_float(
                plan_terms["effective_option_logit_gap_mean"]
            ))
            top_quality_candidate_count_means.append(_scalar_to_float(plan_terms["top_quality_candidate_count_mean"]))
            effective_option_candidate_count_means.append(_scalar_to_float(
                plan_terms["effective_option_candidate_count_mean"]
            ))
            non_top_candidate_count_means.append(_scalar_to_float(plan_terms["non_top_candidate_count_mean"]))
            quality_gap_top_second_means.append(_scalar_to_float(plan_terms["quality_gap_top_second_mean"]))
            effective_option_floor_means.append(_scalar_to_float(plan_terms["effective_option_floor_mean"]))
            soft_target_means.append(_scalar_to_float(plan_terms["soft_target_mean"]))
            soft_target_positive_counts.append(_scalar_to_float(plan_terms["soft_target_positive_count_mean"]))
            soft_target_weight_means.append(_scalar_to_float(plan_terms["soft_target_weight_mean"]))
            soft_target_gap_means.append(_scalar_to_float(plan_terms["soft_target_gap_mean"]))
            soft_target_positive_prob_means.append(_scalar_to_float(plan_terms["soft_target_positive_prob_mean"]))
            soft_target_negative_prob_means.append(_scalar_to_float(plan_terms["soft_target_negative_prob_mean"]))
            contrast_margin_gap_means.append(_scalar_to_float(plan_terms["contrast_margin_gap_mean"]))
            per_service_prob_std_means.append(_scalar_to_float(plan_terms["per_service_prob_std_mean"]))
            per_service_prob_range_means.append(_scalar_to_float(plan_terms["per_service_prob_range_mean"]))
            prob_above_05_ratios.append(_scalar_to_float(plan_terms["prob_above_05_ratio"]))

        value_t = torch.stack(values).float()
        positive_logp_t = torch.stack(positive_logps).float()
        negative_logp_t = torch.stack(negative_logps).float()
        aux_positive_logp_t = torch.stack(aux_positive_logps).float()
        raw_removed_logp_t = torch.stack(raw_removed_logps).float()
        unselected_negative_logp_t = torch.stack(unselected_negative_logps).float()
        ent_t = torch.stack(entropies).float()
        coverage_margin_loss_t = torch.stack(coverage_margin_losses).float()
        quality_margin_loss_t = torch.stack(quality_margin_losses).float()
        ranking_margin_loss_t = torch.stack(ranking_margin_losses).float()
        contrast_margin_loss_t = torch.stack(contrast_margin_losses).float()
        memory_margin_loss_t = torch.stack(memory_margin_losses).float()
        effective_option_mass_loss_t = torch.stack(effective_option_mass_losses).float()
        non_effective_option_loss_t = torch.stack(non_effective_option_losses).float()
        target_t = torch.tensor(targets, device=device, dtype=torch.float32)
        adv_t = target_t.detach() - value_t.detach()
        temp = max(1e-6, float(advantage_temperature))
        min_weight = max(0.0, float(min_advantage_weight))
        max_weight = max(min_weight, float(max_advantage_weight))
        awac_weight = torch.exp(adv_t / temp).clamp(min=min_weight, max=max_weight)

        bad_mask = torch.tensor(
            [1.0 if quality == "bad" else 0.0 for quality in quality_values],
            device=device,
            dtype=torch.float32,
        )
        positive_weight = awac_weight * (1.0 - bad_mask)
        negative_weight = torch.where(
            bad_mask > 0.5,
            torch.ones_like(bad_mask),
            torch.full_like(bad_mask, 0.25),
        )
        raw_removed_weight = torch.where(
            bad_mask > 0.5,
            torch.ones_like(bad_mask),
            torch.full_like(bad_mask, 0.75),
        )
        unselected_negative_weight = torch.where(
            bad_mask > 0.5,
            torch.zeros_like(bad_mask),
            torch.ones_like(bad_mask),
        )
        soft_target_weight = torch.where(
            bad_mask > 0.5,
            torch.ones_like(bad_mask),
            awac_weight.clamp_min(0.25),
        )

        def pair_soft_target_loss() -> Tuple[torch.Tensor, torch.Tensor]:
            if soft_target_logits_for_bce:
                numerator = soft_target_logits_for_bce[0].sum() * 0.0
            else:
                numerator = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
            denominator = torch.zeros((), device=device, dtype=torch.float32)
            for logits, target, pair_weight, transition_weight in zip(
                    soft_target_logits_for_bce,
                    soft_targets_for_bce,
                    soft_target_weights_for_bce,
                    soft_target_weight,
            ):
                pair_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    target.to(device=logits.device, dtype=torch.float32),
                    reduction="none",
                )
                weight = pair_weight.to(device=logits.device, dtype=torch.float32) \
                    * transition_weight.to(device=logits.device, dtype=torch.float32)
                numerator = numerator + (pair_loss * weight).sum()
                denominator = denominator + weight.sum()
            loss = numerator / denominator.clamp_min(1.0)
            return loss * float(soft_target_bc_coef), denominator.detach()

        def pair_bce_loss(
                masks: List[torch.Tensor],
                weights: torch.Tensor,
                positive_target: bool,
                coef: float,
                logits_list: Optional[List[torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            source_logits = logits_list if logits_list is not None else select_logits_for_bce
            if source_logits:
                numerator = source_logits[0].sum() * 0.0
            else:
                numerator = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)
            denominator = torch.zeros((), device=device, dtype=torch.float32)
            for logits, mask, transition_weight in zip(source_logits, masks, weights):
                if not bool(mask.any().item()):
                    continue
                mask_f = mask.to(device=logits.device, dtype=torch.float32)
                if positive_target:
                    pair_loss = F.softplus(float(positive_logit_margin) - logits)
                else:
                    pair_loss = F.softplus(logits + float(negative_logit_margin))
                pair_weight = mask_f * transition_weight.to(device=logits.device, dtype=torch.float32)
                numerator = numerator + (pair_loss * pair_weight).sum()
                denominator = denominator + pair_weight.sum()
            loss = numerator / denominator.clamp_min(1.0)
            return loss * float(coef), denominator.detach()

        soft_target_loss, soft_target_pair_weight_sum = pair_soft_target_loss()
        actor_loss, positive_pair_weight_sum = pair_bce_loss(
            positive_masks_for_bce,
            positive_weight,
            positive_target=True,
            coef=float(actor_bc_coef),
        )
        negative_loss, negative_pair_weight_sum = pair_bce_loss(
            negative_masks_for_bce,
            negative_weight,
            positive_target=False,
            coef=float(negative_bc_coef),
            logits_list=pair_negative_logits_for_bce,
        )
        aux_positive_loss, aux_positive_pair_weight_sum = pair_bce_loss(
            aux_positive_masks_for_bce,
            positive_weight,
            positive_target=True,
            coef=float(executed_aux_positive_coef),
        )
        raw_removed_negative_loss, raw_removed_pair_weight_sum = pair_bce_loss(
            raw_removed_masks_for_bce,
            raw_removed_weight,
            positive_target=False,
            coef=float(raw_removed_negative_coef),
            logits_list=pair_negative_logits_for_bce,
        )
        unselected_negative_loss, unselected_negative_pair_weight_sum = pair_bce_loss(
            unselected_negative_masks_for_bce,
            unselected_negative_weight,
            positive_target=False,
            coef=float(unselected_negative_coef),
            logits_list=pair_negative_logits_for_bce,
        )
        value_loss = F.mse_loss(value_t, target_t)
        entropy = ent_t.mean()
        critic_loss = float(value_coef) * value_loss
        actor_pair_bce_loss = (
            soft_target_loss
            + actor_loss
            + aux_positive_loss
            + negative_loss
            + raw_removed_negative_loss
            + unselected_negative_loss
        )
        coverage_margin_loss = coverage_margin_loss_t.mean() * float(coverage_margin_coef)
        quality_margin_loss = quality_margin_loss_t.mean() * float(quality_margin_coef)
        ranking_margin_loss = ranking_margin_loss_t.mean() * float(ranking_margin_coef)
        contrast_margin_loss = contrast_margin_loss_t.mean() * float(contrast_margin_coef)
        memory_margin_loss = memory_margin_loss_t.mean() * float(memory_margin_coef)
        effective_option_mass_loss = effective_option_mass_loss_t.mean() * float(effective_option_mass_coef)
        non_effective_option_loss = non_effective_option_loss_t.mean() * float(non_effective_option_coef)
        margin_loss = (
            coverage_margin_loss
            + quality_margin_loss
            + ranking_margin_loss
            + contrast_margin_loss
            + memory_margin_loss
            + effective_option_mass_loss
            + non_effective_option_loss
        )
        actor_objective_loss = (
            actor_pair_bce_loss
            + margin_loss
            - float(entropy_coef) * entropy
        )

        # Keep the deployment actor update behavior-cloning/AWAC driven.
        # Actor and critic use separate forward graphs so optimizer steps cannot
        # invalidate masks or tensors saved for the other backward pass.
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        critic_grad_norm = _parameters_grad_norm(self.critic.parameters())
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_objective_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self._actor_train_params, 1.0)
        self.actor_opt.step()

        return {
            "samples": len(batch),
            "epochs": 1,
            "batch_size": len(batch),
            "minibatches": 1,
            "reward_mean": _mean_or_zero(rewards),
            "reward_std": _std_or_zero(rewards),
            "reward_min": float(min(rewards)) if rewards else 0.0,
            "reward_max": float(max(rewards)) if rewards else 0.0,
            "value_old_mean": _scalar_to_float(value_t.detach().mean()),
            "value_old_std": _tensor_std_float(value_t.detach()),
            "value_new_mean": _scalar_to_float(value_t.detach().mean()),
            "return_mean": _scalar_to_float(target_t.detach().mean()),
            "return_std": _tensor_std_float(target_t.detach()),
            "adv_mean": _scalar_to_float(adv_t.detach().mean()),
            "adv_std": _tensor_std_float(adv_t.detach()),
            "last_value": float(targets[-1] if targets else 0.0),
            "done_fraction": _mean_or_zero([1.0 if bool(tr.get("done", False)) else 0.0 for tr in batch]),
            "policy_loss": _scalar_to_float(actor_objective_loss.detach()),
            "actor_pair_bce_loss": _scalar_to_float(actor_pair_bce_loss.detach()),
            "soft_target_loss": _scalar_to_float(soft_target_loss.detach()),
            "aux_positive_loss": _scalar_to_float(aux_positive_loss.detach()),
            "negative_loss": _scalar_to_float(negative_loss.detach()),
            "raw_removed_negative_loss": _scalar_to_float(raw_removed_negative_loss.detach()),
            "unselected_negative_loss": _scalar_to_float(unselected_negative_loss.detach()),
            "coverage_margin_loss": _scalar_to_float(coverage_margin_loss.detach()),
            "quality_margin_loss": _scalar_to_float(quality_margin_loss.detach()),
            "ranking_margin_loss": _scalar_to_float(ranking_margin_loss.detach()),
            "contrast_margin_loss": _scalar_to_float(contrast_margin_loss.detach()),
            "memory_margin_loss": _scalar_to_float(memory_margin_loss.detach()),
            "effective_option_mass_loss": _scalar_to_float(effective_option_mass_loss.detach()),
            "non_effective_option_loss": _scalar_to_float(non_effective_option_loss.detach()),
            "margin_loss": _scalar_to_float(margin_loss.detach()),
            "value_loss": _scalar_to_float(value_loss.detach()),
            "entropy": _scalar_to_float(entropy.detach()),
            "entropy_coef": float(entropy_coef),
            "soft_target_bc_coef": float(soft_target_bc_coef),
            "executed_aux_positive_coef": float(executed_aux_positive_coef),
            "value_coef": float(value_coef),
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "ratio_mean": _scalar_to_float(awac_weight.detach().mean()),
            "ratio_std": _tensor_std_float(awac_weight.detach()),
            "actor_grad_norm": _scalar_to_float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_positive_weight_mean": _scalar_to_float(positive_weight.detach().mean()),
            "actor_negative_weight_mean": _scalar_to_float(negative_weight.detach().mean()),
            "actor_raw_removed_weight_mean": _scalar_to_float(raw_removed_weight.detach().mean()),
            "actor_unselected_negative_weight_mean": _scalar_to_float(
                unselected_negative_weight.detach().mean()
            ),
            "positive_pair_weight_sum": _scalar_to_float(positive_pair_weight_sum),
            "soft_target_pair_weight_sum": _scalar_to_float(soft_target_pair_weight_sum),
            "aux_positive_pair_weight_sum": _scalar_to_float(aux_positive_pair_weight_sum),
            "negative_pair_weight_sum": _scalar_to_float(negative_pair_weight_sum),
            "raw_removed_pair_weight_sum": _scalar_to_float(raw_removed_pair_weight_sum),
            "unselected_negative_pair_weight_sum": _scalar_to_float(unselected_negative_pair_weight_sum),
            "actor_positive_samples": _mean_or_zero(positive_counts),
            "actor_aux_positive_samples": _mean_or_zero(aux_positive_counts),
            "actor_negative_samples": _mean_or_zero(negative_counts),
            "actor_raw_removed_samples": _mean_or_zero(raw_removed_counts),
            "actor_unselected_negative_samples": _mean_or_zero(unselected_negative_counts),
            "actor_selected_risky_samples": _mean_or_zero(selected_risky_counts),
            "actor_selected_low_quality_samples": _mean_or_zero(selected_low_quality_counts),
            "actor_selected_runtime_risky_samples": _mean_or_zero(selected_runtime_risky_counts),
            "actor_selected_unknown_samples": _mean_or_zero(selected_unknown_counts),
            "actor_selected_stale_samples": _mean_or_zero(selected_stale_counts),
            "actor_teacher_positive_samples": _mean_or_zero(teacher_positive_counts),
            "actor_selected_effective_samples": _mean_or_zero(selected_effective_counts),
            "actor_clear_non_effective_samples": _mean_or_zero(clear_non_effective_counts),
            "bad_actor_masked": _scalar_to_float(bad_mask.detach().mean()),
            "positive_logp_mean": _scalar_to_float(positive_logp_t.detach().mean()),
            "aux_positive_logp_mean": _scalar_to_float(aux_positive_logp_t.detach().mean()),
            "negative_logp_mean": _scalar_to_float(negative_logp_t.detach().mean()),
            "raw_removed_logp_mean": _scalar_to_float(raw_removed_logp_t.detach().mean()),
            "unselected_negative_logp_mean": _scalar_to_float(
                unselected_negative_logp_t.detach().mean()
            ),
            "positive_prob_mean": _mean_or_zero(positive_prob_means),
            "aux_positive_prob_mean": _mean_or_zero(aux_positive_prob_means),
            "negative_prob_mean": _mean_or_zero(negative_prob_means),
            "raw_removed_prob_mean": _mean_or_zero(raw_removed_prob_means),
            "unselected_negative_prob_mean": _mean_or_zero(unselected_negative_prob_means),
            "soft_target_mean": _mean_or_zero(soft_target_means),
            "soft_target_positive_count_mean": _mean_or_zero(soft_target_positive_counts),
            "soft_target_weight_mean": _mean_or_zero(soft_target_weight_means),
            "soft_target_gap_mean": _mean_or_zero(soft_target_gap_means),
            "soft_target_positive_prob_mean": _mean_or_zero(soft_target_positive_prob_means),
            "soft_target_negative_prob_mean": _mean_or_zero(soft_target_negative_prob_means),
            "positive_logit_mean": _mean_or_zero(positive_logit_means),
            "aux_positive_logit_mean": _mean_or_zero(aux_positive_logit_means),
            "negative_logit_mean": _mean_or_zero(negative_logit_means),
            "raw_removed_logit_mean": _mean_or_zero(raw_removed_logit_means),
            "unselected_negative_logit_mean": _mean_or_zero(unselected_negative_logit_means),
            "edge_prob_mean": _mean_or_zero(edge_prob_means),
            "edge_prob_std": _mean_or_zero(edge_prob_stds),
            "edge_logit_mean": _mean_or_zero(edge_logit_means),
            "edge_logit_std": _mean_or_zero(edge_logit_stds),
            "pair_centered_logit_std": _mean_or_zero(pair_centered_logit_stds),
            "prob_above_05_ratio": _mean_or_zero(prob_above_05_ratios),
            "raw_mode_edge_density": _mean_or_zero(raw_mode_edge_densities),
            "logit_margin_mean": (
                _mean_or_zero(positive_logit_means) - _mean_or_zero(negative_logit_means)
            ),
            "service_no_edge_prob_mean": _mean_or_zero(service_no_edge_prob_means),
            "service_no_edge_prob_max": _mean_or_zero(service_no_edge_prob_maxes),
            "expected_edge_count_mean": _mean_or_zero(expected_edge_count_means),
            "expected_edge_count_max": _mean_or_zero(expected_edge_count_maxes),
            "expected_quality_mass_mean": _mean_or_zero(expected_quality_mass_means),
            "expected_quality_mass_min": _mean_or_zero(expected_quality_mass_mins),
            "effective_option_mass_mean": _mean_or_zero(effective_option_mass_means),
            "desired_effective_option_mass_mean": _mean_or_zero(desired_effective_option_mass_means),
            "effective_option_shortage_mean": _mean_or_zero(effective_option_shortage_means),
            "non_effective_option_prob_mean": _mean_or_zero(non_effective_option_prob_means),
            "non_effective_selection_cost": _mean_or_zero(non_effective_selection_costs),
            "expected_memory_overage_mean": _mean_or_zero(expected_memory_overage_means),
            "expected_memory_overage_max": _mean_or_zero(expected_memory_overage_maxes),
            "top_quality_prob_mean": _mean_or_zero(top_quality_prob_means),
            "effective_option_prob_mean": _mean_or_zero(effective_option_prob_means),
            "non_top_prob_mean": _mean_or_zero(non_top_prob_means),
            "top_quality_logit_mean": _mean_or_zero(top_quality_logit_means),
            "effective_option_logit_mean": _mean_or_zero(effective_option_logit_means),
            "non_top_logit_mean": _mean_or_zero(non_top_logit_means),
            "top_quality_logit_gap_mean": _mean_or_zero(top_quality_logit_gap_means),
            "effective_option_logit_gap_mean": _mean_or_zero(effective_option_logit_gap_means),
            "top_quality_candidate_count_mean": _mean_or_zero(top_quality_candidate_count_means),
            "effective_option_candidate_count_mean": _mean_or_zero(effective_option_candidate_count_means),
            "non_top_candidate_count_mean": _mean_or_zero(non_top_candidate_count_means),
            "quality_gap_top_second_mean": _mean_or_zero(quality_gap_top_second_means),
            "effective_option_floor_mean": _mean_or_zero(effective_option_floor_means),
            "contrast_margin_gap_mean": _mean_or_zero(contrast_margin_gap_means),
            "per_service_prob_std_mean": _mean_or_zero(per_service_prob_std_means),
            "per_service_prob_range_mean": _mean_or_zero(per_service_prob_range_means),
            "coverage_margin_coef": float(coverage_margin_coef),
            "quality_margin_coef": float(quality_margin_coef),
            "ranking_margin_coef": float(ranking_margin_coef),
            "contrast_margin_coef": float(contrast_margin_coef),
            "memory_margin_coef": float(memory_margin_coef),
            "effective_option_mass_coef": float(effective_option_mass_coef),
            "non_effective_option_coef": float(non_effective_option_coef),
            "positive_logit_margin": float(positive_logit_margin),
            "negative_logit_margin": float(negative_logit_margin),
            "coverage_logit_margin": float(coverage_logit_margin),
            "ranking_logit_margin": float(ranking_logit_margin),
            "contrast_logit_margin": float(contrast_logit_margin),
            "top_quality_tolerance": float(top_quality_tolerance),
            "coverage_pressure_floor": float(coverage_pressure_floor),
            "option_quality_ratio": (
                self._option_quality_ratio() if option_quality_ratio is None else float(option_quality_ratio)
            ),
        }


class HedgerOffloadingPPO(nn.Module):
    def __init__(self, encoder: TopologyEncoders, d_model=64,
                 actor_lr=3e-4, critic_lr=1e-3, update_encoder: bool = True,
                 gamma=0.99, lamda=0.95, clip_eps=0.2,
                 cloud_node_idx: int = -1,
                 unknown_exploration_prob: float = 0.0,
                 risk_exploration_prob: float = 0.0,
                 risk_exploration_temperature: float = 0.25,
                 risk_exploration_min_gap: float = 0.05,
                 static_prior_scale: float = 0.45,
                 runtime_weight: float = 0.45,
                 runtime_clip: float = 3.0,
                 absolute_queue_weight: float = 0.35,
                 relative_queue_weight: float = 1.8,
                 overload_weight: float = 3.0,
                 planned_load_weight: float = 0.8,
                 relative_planned_load_weight: float = 0.8,
                 offered_load_weight: float = 0.45,
                 offered_load_clip: float = 1.0,
                 weak_replica_weight: float = 1.2,
                 weak_gap_clip: float = 1.0,
                 runtime_weak_gap_clip: float = 0.35,
                 runtime_weakness_min_confidence: float = 0.2,
                 runtime_recency_floor: float = 0.7,
                 weak_service_time_weight: float = 0.5,
                 weak_capacity_weight: float = 0.35,
                 weak_queue_amplifier: float = 1.0,
                 cloud_fallback_penalty: float = 1.2,
                 cross_tier_weight: float = 0.2,
                 planned_load_clip: float = 3.0,
                 load_clip: float = 3.0,
                 risk_clip: float = 3.0):
        super().__init__()
        self.encoder = encoder
        self.actor = OffloadActor(d_model)
        hidden_dim = max(32, d_model)
        self.offloading_static_prior_head = CandidateCostHead(
            input_dim=len(OFFLOADING_STATIC_PRIOR_FEATURE_NAMES),
            hidden_dim=hidden_dim,
        )
        self.critic = CandidateValueHead(input_dim=len(OFFLOADING_CANDIDATE_FEATURE_NAMES), d_model=d_model)

        encoder_params = list(self.encoder.parameters()) if update_encoder else []
        params_actor = (
            list(self.actor.parameters())
            + list(self.offloading_static_prior_head.parameters())
            + encoder_params
        )
        self.actor_opt = torch.optim.Adam(params_actor, lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self._actor_train_params = params_actor

        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.cloud_idx = cloud_node_idx
        self.unknown_exploration_prob = float(max(0.0, min(1.0, unknown_exploration_prob)))
        self.risk_exploration_prob = float(max(0.0, min(1.0, risk_exploration_prob)))
        self.risk_exploration_temperature = float(max(1e-6, risk_exploration_temperature))
        self.risk_exploration_min_gap = float(max(0.0, risk_exploration_min_gap))
        self.static_prior_scale = float(max(0.0, static_prior_scale))
        self.runtime_weight = float(max(0.0, runtime_weight))
        self.runtime_clip = float(max(1e-6, runtime_clip))
        self.absolute_queue_weight = float(max(0.0, absolute_queue_weight))
        self.relative_queue_weight = float(max(0.0, relative_queue_weight))
        self.overload_weight = float(max(0.0, overload_weight))
        self.planned_load_weight = float(max(0.0, planned_load_weight))
        self.relative_planned_load_weight = float(max(0.0, relative_planned_load_weight))
        self.offered_load_weight = float(max(0.0, offered_load_weight))
        self.offered_load_clip = float(max(1e-6, offered_load_clip))
        self.weak_replica_weight = float(max(0.0, weak_replica_weight))
        self.weak_gap_clip = float(max(1e-6, weak_gap_clip))
        self.runtime_weak_gap_clip = float(max(1e-6, runtime_weak_gap_clip))
        self.runtime_weakness_min_confidence = float(max(0.0, min(1.0, runtime_weakness_min_confidence)))
        self.runtime_recency_floor = float(max(0.0, min(1.0, runtime_recency_floor)))
        self.weak_service_time_weight = float(max(0.0, weak_service_time_weight))
        self.weak_capacity_weight = float(max(0.0, weak_capacity_weight))
        self.weak_queue_amplifier = float(max(0.0, weak_queue_amplifier))
        self.cloud_fallback_penalty = float(max(0.0, cloud_fallback_penalty))
        self.cross_tier_weight = float(max(0.0, cross_tier_weight))
        self.planned_load_clip = float(max(1e-6, planned_load_clip))
        self.load_clip = float(max(1e-6, load_clip))
        self.risk_clip = float(max(1e-6, risk_clip))

    def _cloud_index(self, width: int) -> int:
        if width <= 0:
            return -1
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (width - 1)
        if cloud_idx < 0 or cloud_idx >= width:
            return width - 1
        return cloud_idx

    @staticmethod
    def topo_order(edge_index: torch.Tensor, num_nodes: int):
        if num_nodes <= 0:
            return []
        if edge_index.numel() == 0:
            return list(range(num_nodes))

        row, col = edge_index.detach().cpu()
        adj = [[] for _ in range(num_nodes)]
        indeg = [0 for _ in range(num_nodes)]
        for u_raw, v_raw in zip(row.tolist(), col.tolist()):
            u, v = int(u_raw), int(v_raw)
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adj[u].append(v)
                indeg[v] += 1

        q = [i for i, deg in enumerate(indeg) if deg == 0]
        order = []
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(order) < num_nodes:
            seen = set(order)
            order += [i for i in range(num_nodes) if i not in seen]
        return order

    def _adapt_embeddings(self, h_s: torch.Tensor, h_p: torch.Tensor):
        return h_s, h_p

    def _effective_offloading_mask(self, static_mask: torch.Tensor) -> torch.Tensor:
        effective_mask = static_mask.detach().clone().bool()
        row_has_any = effective_mask.any(dim=1)
        if (~row_has_any).any():
            effective_mask[~row_has_any, self._cloud_index(effective_mask.size(1))] = True
        return effective_mask

    def _qk_components(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_embedding = self.actor.q(h_s)
        k_embedding = self.actor.k(h_p)
        qk_scores = (q_embedding @ k_embedding.t()) / math.sqrt(q_embedding.size(-1))
        qk_scores = qk_scores / max(self.actor.temperature, 1e-6)
        qk_feature = _masked_standardize(qk_scores, mask)
        return q_embedding, k_embedding, qk_scores, qk_feature

    @staticmethod
    def _offloading_action_at(actions: Union[Sequence[int], torch.Tensor], idx: int) -> int:
        value = actions[idx]
        return int(value.item()) if isinstance(value, torch.Tensor) else int(value)

    def _offloading_dependency_context(
            self,
            parents: List[List[int]],
            service_idx: int,
            actions: Union[Sequence[int], torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            num_devices: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        cloud_idx = self._cloud_index(num_devices)
        role_id, is_cloud = _role_tensors(phys_feats, num_devices, device, dtype, cloud_idx)
        bandwidth = _feature_vector(phys_feats, "bandwidth_latest", num_devices, device, dtype).clamp_min(1e-6)
        edge_mask = is_cloud < 0.5
        edge_bw_ref = bandwidth[edge_mask].mean() if edge_mask.any() else bandwidth.mean()

        context = torch.zeros((num_devices, 2), device=device, dtype=dtype)
        parent_indices = parents[service_idx]
        if not parent_indices:
            return context

        valid_parent_actions = []
        for parent_idx in parent_indices:
            action = self._offloading_action_at(actions, parent_idx)
            if 0 <= action < num_devices:
                valid_parent_actions.append(action)
        if not valid_parent_actions:
            return context
        denom = float(max(1, len(valid_parent_actions)))
        for device_idx in range(num_devices):
            role = int(role_id[device_idx].item())
            same_device = sum(1 for action in valid_parent_actions if action == device_idx) / denom
            cross_tier = sum(1 for action in valid_parent_actions if int(role_id[action].item()) != role) / denom
            context[device_idx, 0] = same_device
            context[device_idx, 1] = cross_tier * torch.log1p(edge_bw_ref / bandwidth[device_idx])
        return context

    def _offloading_candidate_row(
            self,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            qk_feature: torch.Tensor,
            runtime_feat: torch.Tensor,
            demand_feat: torch.Tensor,
            parents: List[List[int]],
            service_idx: int,
            actions: Union[Sequence[int], torch.Tensor],
            allowed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_devices = qk_feature.size(1)
        device = qk_feature.device
        dtype = qk_feature.dtype
        if allowed is None:
            allowed = torch.ones((num_devices,), device=device, dtype=torch.bool)
        else:
            allowed = allowed.to(device=device).bool()
        dependency_context = self._offloading_dependency_context(
            parents,
            service_idx,
            actions,
            phys_feats,
            num_devices,
            device,
            dtype,
        )
        runtime_row = runtime_feat[service_idx]
        cloud_idx = self._cloud_index(num_devices)
        device_feat = _device_capability_features(phys_feats, num_devices, device, dtype, cloud_idx)
        _, is_cloud = _role_tensors(phys_feats, num_devices, device, dtype, cloud_idx)
        compute_gap = demand_feat[service_idx, 0] - device_feat[:, 0]
        arrival_rate = demand_feat[service_idx, 2].expand(num_devices)
        queue_short = runtime_row[..., 0]
        queue_busy = runtime_row[..., 1]
        confidence = runtime_row[..., 3]
        runtime_recency = runtime_row[..., 4]
        queue_freshness = runtime_row[..., 5]
        runtime_ratio = runtime_row[..., 2].clamp_min(0.0).clamp(max=self.runtime_clip) / self.runtime_clip
        speed_evidence = (
            confidence.clamp(0.0, 1.0)
            * (self.runtime_recency_floor + (1.0 - self.runtime_recency_floor) * runtime_recency.clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)
        capability_prior = torch.sigmoid(compute_gap + 4.0)
        capacity_pressure = capability_prior
        service_time_factor = (
            speed_evidence * runtime_ratio
            + (1.0 - speed_evidence) * capability_prior
        ).clamp(0.0, 1.0)
        pair_load = (
            queue_freshness.clamp(0.0, 1.0)
            * (queue_short.clamp_min(0.0) + 0.5 * queue_busy.clamp_min(0.0))
        ).clamp(max=self.load_clip)
        device_pair_load = (
            runtime_feat[..., 5].clamp(0.0, 1.0)
            * (runtime_feat[..., 0].clamp_min(0.0) + 0.5 * runtime_feat[..., 1].clamp_min(0.0))
        )
        device_load = device_pair_load.sum(dim=0).clamp(max=self.load_clip)
        return torch.stack(
            [
                qk_feature[service_idx],
                compute_gap,
                arrival_rate,
                runtime_ratio,
                confidence,
                runtime_recency,
                queue_freshness,
                speed_evidence,
                capacity_pressure,
                pair_load,
                device_load,
                service_time_factor,
                dependency_context[..., 0],
                dependency_context[..., 1],
                is_cloud,
            ],
            dim=-1,
        )

    @staticmethod
    def _offloading_feature(candidate_row: torch.Tensor, name: str) -> torch.Tensor:
        return candidate_row[..., OFFLOADING_CANDIDATE_FEATURE_NAMES.index(name)]

    def _offloading_static_prior_row(self, candidate_row: torch.Tensor) -> torch.Tensor:
        indices = [
            OFFLOADING_CANDIDATE_FEATURE_NAMES.index(name)
            for name in OFFLOADING_STATIC_PRIOR_FEATURE_NAMES
        ]
        return candidate_row[..., indices]

    def _offloading_static_prior(self, candidate_row: torch.Tensor) -> torch.Tensor:
        raw_score = self.offloading_static_prior_head(self._offloading_static_prior_row(candidate_row).float())
        return self.static_prior_scale * torch.tanh(raw_score)

    def _offloading_runtime_risk(self, candidate_row: torch.Tensor) -> torch.Tensor:
        service_time = self._offloading_feature(candidate_row, "service_time_factor").float().clamp(0.0, 1.0)
        return self.runtime_weight * service_time

    def _offloading_load_pressure(self, candidate_row: torch.Tensor) -> torch.Tensor:
        pair_load = self._offloading_feature(candidate_row, "pair_load").float().clamp_min(0.0)
        device_load = self._offloading_feature(candidate_row, "device_load").float().clamp_min(0.0)
        return (pair_load + device_load).clamp(max=self.load_clip)

    def _offloading_raw_queue_risk(
            self,
            candidate_row: torch.Tensor,
    ) -> torch.Tensor:
        service_time = self._offloading_feature(candidate_row, "service_time_factor").float().clamp(0.0, 1.0)
        load_pressure = self._offloading_load_pressure(candidate_row)
        risk = service_time * torch.log1p(load_pressure)
        return risk.clamp(max=self.risk_clip)

    def _offloading_planned_load_risk(
            self,
            candidate_row: torch.Tensor,
            planned_device_load: torch.Tensor,
    ) -> torch.Tensor:
        load = planned_device_load.to(device=candidate_row.device, dtype=candidate_row.dtype)
        load = load.clamp_min(0.0).clamp(max=self.planned_load_clip)
        service_time = self._offloading_feature(candidate_row, "service_time_factor").float().clamp(0.0, 1.0)
        risk = service_time * torch.log1p(load.float())
        return self.planned_load_weight * risk.clamp(max=self.risk_clip)

    def _offloading_score_baseline_mask(
            self,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
    ) -> torch.Tensor:
        allowed = allowed.to(device=candidate_row.device).bool()
        is_cloud = self._offloading_feature(candidate_row, "is_cloud").float() > 0.5
        edge_allowed = allowed & ~is_cloud
        if bool(edge_allowed.any().detach().item()):
            return edge_allowed
        if bool(allowed.any().detach().item()):
            return allowed
        return torch.ones_like(allowed, dtype=torch.bool, device=candidate_row.device)

    @staticmethod
    def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask.to(device=values.device).bool(), float("inf"))
        min_value = masked.min()
        if bool(torch.isfinite(min_value).detach().item()):
            return min_value
        return torch.zeros((), dtype=values.dtype, device=values.device)

    def _offloading_relative_queue_risk(
            self,
            base_queue_risk: torch.Tensor,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
    ) -> torch.Tensor:
        baseline_mask = self._offloading_score_baseline_mask(candidate_row, allowed)
        baseline = self._masked_min(base_queue_risk.detach(), baseline_mask)
        relative = torch.relu(base_queue_risk - baseline)
        return self.relative_queue_weight * relative.clamp(max=self.risk_clip)

    def _offloading_overload_risk(
            self,
            candidate_row: torch.Tensor,
            load_pressure: torch.Tensor,
            allowed: torch.Tensor,
    ) -> torch.Tensor:
        baseline_mask = self._offloading_score_baseline_mask(candidate_row, allowed)
        baseline_load = self._masked_min(load_pressure.detach(), baseline_mask)
        service_time = self._offloading_feature(candidate_row, "service_time_factor").float().clamp(0.0, 1.0)
        overload = torch.relu(load_pressure - baseline_load)
        risk = service_time * overload.pow(2)
        return self.overload_weight * risk.clamp(max=self.risk_clip)

    def _offloading_planned_pressure(
            self,
            planned_device_load: torch.Tensor,
            candidate_row: torch.Tensor,
    ) -> torch.Tensor:
        load = planned_device_load.to(device=candidate_row.device, dtype=candidate_row.dtype)
        return torch.log1p(load.clamp_min(0.0).clamp(max=self.planned_load_clip))

    def _offloading_relative_planned_load_risk(
            self,
            planned_pressure: torch.Tensor,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
    ) -> torch.Tensor:
        baseline_mask = self._offloading_score_baseline_mask(candidate_row, allowed)
        baseline = self._masked_min(planned_pressure.detach(), baseline_mask)
        relative = torch.relu(planned_pressure - baseline)
        return self.relative_planned_load_weight * relative.clamp(max=self.risk_clip)

    def _offloading_offered_load_pressure(self, candidate_row: torch.Tensor) -> torch.Tensor:
        arrival_rate = self._offloading_feature(candidate_row, "arrival_rate_short").float().clamp_min(0.0)
        arrival_pressure = torch.log1p(arrival_rate).clamp(max=2.0) / 2.0
        service_time = self._offloading_feature(candidate_row, "service_time_factor").float().clamp(0.0, 1.0)
        capacity_pressure = self._offloading_feature(candidate_row, "capacity_pressure").float().clamp(0.0, 1.0)
        pressure = arrival_pressure * service_time * (0.5 + 0.5 * capacity_pressure)
        return pressure.clamp(max=self.offered_load_clip)

    def _offloading_offered_load_risk(self, offered_load_pressure: torch.Tensor) -> torch.Tensor:
        return self.offered_load_weight * offered_load_pressure.clamp(max=self.risk_clip)

    def _offloading_relative_weakness(
            self,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        allowed = allowed.to(device=candidate_row.device).bool()
        is_cloud = self._offloading_feature(candidate_row, "is_cloud").float() > 0.5
        edge_allowed = allowed & ~is_cloud
        zeros = torch.zeros_like(self._offloading_feature(candidate_row, "compute_gap").float())
        if not bool(edge_allowed.any().detach().item()):
            return {
                "compute_relative_weakness": zeros,
                "runtime_relative_weakness": zeros,
                "relative_weakness": zeros,
            }

        compute_gap = self._offloading_feature(candidate_row, "compute_gap").float()
        best_gap = self._masked_min(compute_gap.detach(), edge_allowed)
        compute_relative = torch.relu(compute_gap - best_gap) / self.weak_gap_clip
        compute_relative = torch.where(edge_allowed, compute_relative.clamp(max=1.0), zeros)

        runtime_confidence = self._offloading_feature(candidate_row, "runtime_confidence").float().clamp(0.0, 1.0)
        speed_evidence = self._offloading_feature(candidate_row, "speed_evidence").float().clamp(0.0, 1.0)
        runtime_known = edge_allowed & (runtime_confidence >= self.runtime_weakness_min_confidence)
        runtime_relative = zeros
        if bool(runtime_known.any().detach().item()):
            runtime_ratio = self._offloading_feature(candidate_row, "runtime_ratio").float().clamp(0.0, 1.0)
            best_runtime = self._masked_min(runtime_ratio.detach(), runtime_known)
            runtime_gap = torch.relu(runtime_ratio - best_runtime) / self.runtime_weak_gap_clip
            runtime_relative = torch.where(
                runtime_known,
                (runtime_gap.clamp(max=1.0) * speed_evidence).clamp(max=1.0),
                zeros,
            )

        relative = torch.maximum(compute_relative, runtime_relative)
        return {
            "compute_relative_weakness": compute_relative,
            "runtime_relative_weakness": runtime_relative,
            "relative_weakness": relative,
        }

    def _offloading_weak_pressure(
            self,
            offered_load_pressure: torch.Tensor,
            service_time_factor: torch.Tensor,
            capacity_pressure: torch.Tensor,
    ) -> torch.Tensor:
        service_time_pressure = self.weak_service_time_weight * service_time_factor.float().clamp(0.0, 1.0)
        capacity_pressure = self.weak_capacity_weight * capacity_pressure.float().clamp(0.0, 1.0)
        return torch.maximum(torch.maximum(offered_load_pressure, service_time_pressure), capacity_pressure).clamp(
            max=self.risk_clip
        )

    def _offloading_weak_replica_risk(
            self,
            weak_pressure: torch.Tensor,
            relative_weakness: torch.Tensor,
            load_pressure: torch.Tensor,
            planned_pressure: torch.Tensor,
    ) -> torch.Tensor:
        load_context = (load_pressure + planned_pressure).clamp_min(0.0).clamp(max=self.load_clip) / self.load_clip
        amplified_pressure = weak_pressure * (1.0 + self.weak_queue_amplifier * load_context)
        risk = relative_weakness * amplified_pressure
        return self.weak_replica_weight * risk.clamp(max=self.risk_clip)

    def _offloading_cloud_penalty(self, candidate_row: torch.Tensor) -> torch.Tensor:
        is_cloud = self._offloading_feature(candidate_row, "is_cloud").float().clamp(0.0, 1.0)
        return self.cloud_fallback_penalty * is_cloud

    def _offloading_cross_tier_penalty_term(self, candidate_row: torch.Tensor) -> torch.Tensor:
        cross_tier = self._offloading_feature(candidate_row, "cross_tier_penalty").float().clamp_min(0.0)
        return self.cross_tier_weight * cross_tier

    def _offloading_planned_load_increment(self, selected_candidate: torch.Tensor) -> torch.Tensor:
        arrival_rate = self._offloading_feature(selected_candidate, "arrival_rate_short").float().clamp_min(0.0)
        service_time = self._offloading_feature(selected_candidate, "service_time_factor").float().clamp(0.0, 1.0)
        arrival_pressure = torch.log1p(arrival_rate).clamp(max=2.0)
        increment = (0.1 + arrival_pressure) * service_time
        return increment.clamp(min=0.05, max=self.planned_load_clip)

    def _score_offloading_candidates(
            self,
            candidate_row: torch.Tensor,
            planned_device_load: torch.Tensor,
            allowed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        static_prior = self._offloading_static_prior(candidate_row)
        runtime_risk = self._offloading_runtime_risk(candidate_row)
        load_pressure = self._offloading_load_pressure(candidate_row.float())
        raw_queue_risk = self._offloading_raw_queue_risk(candidate_row.float())
        base_queue_risk = self.absolute_queue_weight * raw_queue_risk
        relative_queue_risk = self._offloading_relative_queue_risk(
            raw_queue_risk,
            candidate_row.float(),
            allowed,
        )
        overload_risk = self._offloading_overload_risk(
            candidate_row.float(),
            load_pressure,
            allowed,
        )
        queue_risk_total = base_queue_risk + relative_queue_risk + overload_risk
        planned_pressure = self._offloading_planned_pressure(planned_device_load, candidate_row.float())
        planned_load_risk = self._offloading_planned_load_risk(
            candidate_row.float(),
            planned_device_load,
        )
        relative_planned_load_risk = self._offloading_relative_planned_load_risk(
            planned_pressure,
            candidate_row.float(),
            allowed,
        )
        offered_load_pressure = self._offloading_offered_load_pressure(candidate_row.float())
        offered_load_risk = self._offloading_offered_load_risk(offered_load_pressure)
        service_time = self._offloading_feature(candidate_row, "service_time_factor").float().clamp(0.0, 1.0)
        capacity_pressure = self._offloading_feature(candidate_row, "capacity_pressure").float().clamp(0.0, 1.0)
        weakness_terms = self._offloading_relative_weakness(candidate_row.float(), allowed)
        compute_relative_weakness = weakness_terms["compute_relative_weakness"]
        runtime_relative_weakness = weakness_terms["runtime_relative_weakness"]
        relative_weakness = weakness_terms["relative_weakness"]
        weak_pressure = self._offloading_weak_pressure(offered_load_pressure, service_time, capacity_pressure)
        weak_replica_risk = self._offloading_weak_replica_risk(
            weak_pressure,
            relative_weakness,
            load_pressure,
            planned_pressure,
        )
        cloud_penalty = self._offloading_cloud_penalty(candidate_row.float())
        cross_tier_penalty_term = self._offloading_cross_tier_penalty_term(candidate_row.float())
        dynamic_risk = (
            runtime_risk
            + queue_risk_total
            + planned_load_risk
            + relative_planned_load_risk
            + offered_load_risk
            + weak_replica_risk
            + cloud_penalty
            + cross_tier_penalty_term
        )
        final_score = static_prior - dynamic_risk
        return {
            "static_prior": static_prior,
            "runtime_risk": runtime_risk,
            "service_time_factor": service_time,
            "load_pressure": load_pressure,
            "base_queue_risk": base_queue_risk,
            "relative_queue_risk": relative_queue_risk,
            "overload_risk": overload_risk,
            "queue_risk_total": queue_risk_total,
            "planned_pressure": planned_pressure,
            "planned_load_risk": planned_load_risk,
            "relative_planned_load_risk": relative_planned_load_risk,
            "offered_load_pressure": offered_load_pressure,
            "offered_load_risk": offered_load_risk,
            "compute_relative_weakness": compute_relative_weakness,
            "runtime_relative_weakness": runtime_relative_weakness,
            "relative_weakness": relative_weakness,
            "weak_pressure": weak_pressure,
            "weak_replica_risk": weak_replica_risk,
            "cloud_penalty": cloud_penalty,
            "cross_tier_penalty_term": cross_tier_penalty_term,
            "dynamic_risk": dynamic_risk,
            "final_score": final_score,
        }

    @staticmethod
    def _empty_selected_queue_metrics() -> Dict[str, float]:
        return {
            "selected_runtime_ratio": 0.0,
            "selected_runtime_recency": 0.0,
            "selected_queue_freshness": 0.0,
            "selected_speed_evidence": 0.0,
            "selected_capacity_pressure": 0.0,
            "selected_pair_load": 0.0,
            "selected_device_load": 0.0,
            "selected_load_pressure": 0.0,
            "selected_service_time_factor": 0.0,
            "selected_base_queue_risk": 0.0,
            "selected_relative_queue_risk": 0.0,
            "selected_overload_risk": 0.0,
            "selected_queue_risk_total": 0.0,
            "selected_planned_load_risk": 0.0,
            "selected_relative_planned_load_risk": 0.0,
            "selected_offered_load_pressure": 0.0,
            "selected_offered_load_risk": 0.0,
            "selected_compute_relative_weakness": 0.0,
            "selected_runtime_relative_weakness": 0.0,
            "selected_relative_weakness": 0.0,
            "selected_weak_pressure": 0.0,
            "selected_weak_replica_risk": 0.0,
            "selected_dynamic_risk": 0.0,
            "selected_runtime_confidence": 0.0,
            "selected_queue_risk_cost": 0.0,
        }

    @staticmethod
    def _selected_score_mean(
            matrix: Optional[torch.Tensor],
            service_idx: torch.Tensor,
            actions: torch.Tensor,
    ) -> float:
        if matrix is None or matrix.numel() == 0 or service_idx.numel() == 0:
            return 0.0
        selected = matrix[service_idx, actions].float()
        return _scalar_to_float(selected.mean())

    def _selected_offloading_queue_metrics(
            self,
            candidate_features: torch.Tensor,
            actions: torch.Tensor,
            score_matrices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        if candidate_features.numel() == 0 or actions.numel() == 0:
            return self._empty_selected_queue_metrics()
        actions = actions.detach().long().to(candidate_features.device)
        valid = (actions >= 0) & (actions < candidate_features.size(1))
        if not bool(valid.any().item()):
            return self._empty_selected_queue_metrics()
        service_idx = torch.arange(candidate_features.size(0), device=candidate_features.device)[valid]
        selected = candidate_features[service_idx, actions[valid]].float()
        runtime_ratio = self._offloading_feature(selected, "runtime_ratio")
        runtime_recency = self._offloading_feature(selected, "runtime_recency")
        queue_freshness = self._offloading_feature(selected, "queue_freshness")
        speed_evidence = self._offloading_feature(selected, "speed_evidence")
        capacity_pressure = self._offloading_feature(selected, "capacity_pressure")
        pair_load = self._offloading_feature(selected, "pair_load")
        device_load = self._offloading_feature(selected, "device_load")
        service_time = self._offloading_feature(selected, "service_time_factor")
        load_pressure = self._offloading_load_pressure(selected)
        pair_confidence = self._offloading_feature(selected, "runtime_confidence").clamp(0.0, 1.0)
        score_matrices = score_matrices or {}
        base_queue_risk = self._selected_score_mean(score_matrices.get("base_queue_risk"), service_idx, actions[valid])
        relative_queue_risk = self._selected_score_mean(
            score_matrices.get("relative_queue_risk"), service_idx, actions[valid]
        )
        overload_risk = self._selected_score_mean(score_matrices.get("overload_risk"), service_idx, actions[valid])
        queue_risk_total = self._selected_score_mean(
            score_matrices.get("queue_risk_total"), service_idx, actions[valid]
        )
        planned_load_risk = self._selected_score_mean(
            score_matrices.get("planned_load_risk"), service_idx, actions[valid]
        )
        relative_planned_load_risk = self._selected_score_mean(
            score_matrices.get("relative_planned_load_risk"), service_idx, actions[valid]
        )
        offered_load_pressure = self._selected_score_mean(
            score_matrices.get("offered_load_pressure"), service_idx, actions[valid]
        )
        offered_load_risk = self._selected_score_mean(
            score_matrices.get("offered_load_risk"), service_idx, actions[valid]
        )
        compute_relative_weakness = self._selected_score_mean(
            score_matrices.get("compute_relative_weakness"), service_idx, actions[valid]
        )
        runtime_relative_weakness = self._selected_score_mean(
            score_matrices.get("runtime_relative_weakness"), service_idx, actions[valid]
        )
        relative_weakness = self._selected_score_mean(
            score_matrices.get("relative_weakness"), service_idx, actions[valid]
        )
        weak_pressure = self._selected_score_mean(
            score_matrices.get("weak_pressure"), service_idx, actions[valid]
        )
        weak_replica_risk = self._selected_score_mean(
            score_matrices.get("weak_replica_risk"), service_idx, actions[valid]
        )
        dynamic_risk = self._selected_score_mean(score_matrices.get("dynamic_risk"), service_idx, actions[valid])

        return {
            "selected_runtime_ratio": _scalar_to_float(runtime_ratio.mean()),
            "selected_runtime_recency": _scalar_to_float(runtime_recency.mean()),
            "selected_queue_freshness": _scalar_to_float(queue_freshness.mean()),
            "selected_speed_evidence": _scalar_to_float(speed_evidence.mean()),
            "selected_capacity_pressure": _scalar_to_float(capacity_pressure.mean()),
            "selected_pair_load": _scalar_to_float(pair_load.mean()),
            "selected_device_load": _scalar_to_float(device_load.mean()),
            "selected_load_pressure": _scalar_to_float(load_pressure.mean()),
            "selected_service_time_factor": _scalar_to_float(service_time.mean()),
            "selected_base_queue_risk": base_queue_risk,
            "selected_relative_queue_risk": relative_queue_risk,
            "selected_overload_risk": overload_risk,
            "selected_queue_risk_total": queue_risk_total,
            "selected_planned_load_risk": planned_load_risk,
            "selected_relative_planned_load_risk": relative_planned_load_risk,
            "selected_offered_load_pressure": offered_load_pressure,
            "selected_offered_load_risk": offered_load_risk,
            "selected_compute_relative_weakness": compute_relative_weakness,
            "selected_runtime_relative_weakness": runtime_relative_weakness,
            "selected_relative_weakness": relative_weakness,
            "selected_weak_pressure": weak_pressure,
            "selected_weak_replica_risk": weak_replica_risk,
            "selected_dynamic_risk": dynamic_risk,
            "selected_runtime_confidence": _scalar_to_float(pair_confidence.mean()),
            "selected_queue_risk_cost": base_queue_risk + relative_queue_risk + overload_risk,
        }

    def _offloading_value_candidate_features(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            qk_feature: torch.Tensor,
            runtime_feat: torch.Tensor,
    ) -> torch.Tensor:
        num_services, num_devices = qk_feature.shape
        parents = self._build_parents(logic_edge_index, num_services)
        demand_feat = _service_demand_features(logic_feats, num_services, qk_feature.device, qk_feature.dtype)
        unknown_actions = torch.full(
            (num_services,),
            fill_value=-1,
            dtype=torch.long,
            device=qk_feature.device,
        )
        rows = [
            self._offloading_candidate_row(
                logic_feats,
                phys_feats,
                qk_feature,
                runtime_feat,
                demand_feat,
                parents,
                service_idx,
                unknown_actions,
            )
            for service_idx in range(num_services)
        ]
        return torch.stack(rows, dim=0)

    def _offloading_static_terms(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            effective_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_services, num_devices = h_s.size(0), h_p.size(0)
        q_embedding, k_embedding, qk_scores, qk_feature = self._qk_components(h_s, h_p, effective_mask)
        runtime_feat = _runtime_pair_features(logic_feats, num_services, num_devices, h_s.device, h_s.dtype)
        demand_feat = _service_demand_features(logic_feats, num_services, h_s.device, h_s.dtype)
        return q_embedding, k_embedding, qk_scores, qk_feature, runtime_feat, demand_feat

    @torch.no_grad()
    def estimate_value(
            self,
            logic_edge_index,
            logic_feats,
            phys_edge_index,
            phys_feats,
            static_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the state value for rollout bootstrap.

        The offloading loop is a continuing task. Rollout boundaries are
        truncation boundaries, not terminal states, so the final step should be
        bootstrapped with the successor-state value.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        effective_mask = self._effective_offloading_mask(static_mask)
        _, _, _, qk_feature, runtime_feat, _ = self._offloading_static_terms(
            h_s, h_p, logic_feats, effective_mask
        )
        candidate_features = self._offloading_value_candidate_features(
            logic_edge_index, logic_feats, phys_feats, qk_feature, runtime_feat
        )
        value = self.critic(h_s, h_p, candidate_features, effective_mask)
        return value.squeeze(0)

    @staticmethod
    def _build_parents(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        parents = [[] for _ in range(num_nodes)]
        if num_nodes <= 0 or edge_index.numel() == 0:
            return parents
        row, col = edge_index.detach().cpu()
        for u_raw, v_raw in zip(row.tolist(), col.tolist()):
            u, v = int(u_raw), int(v_raw)
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                parents[v].append(u)
        return parents

    def _dynamic_allowed_row(
            self,
            base_mask: torch.Tensor,
            parent_indices: List[int],
            actions: Union[Sequence[int], torch.Tensor],
    ) -> torch.Tensor:
        allowed = base_mask.clone().bool()
        cloud_idx = self._cloud_index(allowed.numel())
        if parent_indices and any(self._offloading_action_at(actions, parent_idx) == cloud_idx
                                  for parent_idx in parent_indices):
            allowed = torch.zeros_like(allowed)
            allowed[cloud_idx] = True
        if not bool(allowed.any().item()):
            allowed[cloud_idx] = True
        return allowed

    def _best_allowed_offloading_target(
            self,
            probs: torch.Tensor,
            allowed: torch.Tensor,
    ) -> int:
        cloud_idx = self._cloud_index(allowed.numel())
        allowed_indices = torch.nonzero(allowed.bool(), as_tuple=False).flatten()
        if allowed_indices.numel() == 0:
            return cloud_idx
        masked_probs = probs.detach().float().clone()
        masked_probs = masked_probs.masked_fill(~allowed.bool(), float("-inf"))
        if torch.isfinite(masked_probs).any():
            return int(torch.argmax(masked_probs).item())
        if 0 <= cloud_idx < allowed.numel() and bool(allowed[cloud_idx].item()):
            return cloud_idx
        return int(allowed_indices[0].item())

    def _unknown_exploration_probs(
            self,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        unknown_probs = torch.zeros_like(allowed, dtype=torch.float32)
        unknown_weight = torch.zeros_like(unknown_probs)
        if candidate_row.numel() == 0 or not bool(allowed.any().item()):
            return unknown_probs, unknown_weight
        confidence_idx = OFFLOADING_CANDIDATE_FEATURE_NAMES.index("runtime_confidence")
        is_cloud_idx = OFFLOADING_CANDIDATE_FEATURE_NAMES.index("is_cloud")
        confidence = candidate_row[:, confidence_idx].detach().float().clamp(0.0, 1.0)
        is_cloud = candidate_row[:, is_cloud_idx].detach().float() > 0.5
        unknown_weight = (1.0 - confidence).masked_fill(~allowed.bool(), 0.0)
        unknown_weight = unknown_weight.masked_fill(is_cloud, 0.0)
        weight_sum = unknown_weight.sum()
        if float(weight_sum.detach().item()) > 1e-8:
            unknown_probs = unknown_weight / weight_sum.clamp_min(1e-8)
        return unknown_probs, unknown_weight

    def _risk_exploration_probs(
            self,
            dynamic_risk: torch.Tensor,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        risk_probs = torch.zeros_like(allowed, dtype=torch.float32)
        risk_weight = torch.zeros_like(risk_probs)
        if dynamic_risk.numel() == 0 or candidate_row.numel() == 0 or not bool(allowed.any().item()):
            return risk_probs, risk_weight
        is_cloud = self._offloading_feature(candidate_row, "is_cloud").detach().float() > 0.5
        risk_mask = allowed.to(device=dynamic_risk.device).bool() & ~is_cloud.to(device=dynamic_risk.device)
        if int(risk_mask.sum().detach().item()) < 2:
            return risk_probs, risk_weight
        risk = dynamic_risk.detach().float()
        masked_risk = risk.masked_fill(~risk_mask, float("inf"))
        min_risk = masked_risk.min()
        max_risk = risk.masked_fill(~risk_mask, float("-inf")).max()
        if not bool((torch.isfinite(min_risk) & torch.isfinite(max_risk)).detach().item()):
            return risk_probs, risk_weight
        if float((max_risk - min_risk).detach().item()) < self.risk_exploration_min_gap:
            return risk_probs, risk_weight
        risk_weight = torch.relu(max_risk - risk).masked_fill(~risk_mask, 0.0)
        logits = (-risk / self.risk_exploration_temperature).masked_fill(~risk_mask, float("-inf"))
        risk_probs = F.softmax(logits, dim=-1)
        risk_probs = risk_probs.masked_fill(~risk_mask, 0.0)
        if float(risk_probs.sum().detach().item()) <= 1e-8:
            return torch.zeros_like(risk_probs), risk_weight
        return risk_probs / risk_probs.sum().clamp_min(1e-8), risk_weight

    def _mix_offloading_exploration_probs(
            self,
            base_probs: torch.Tensor,
            candidate_row: torch.Tensor,
            score_terms: Dict[str, torch.Tensor],
            allowed: torch.Tensor,
            *,
            enable_exploration: bool,
            deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        unknown_probs, unknown_weight = self._unknown_exploration_probs(candidate_row, allowed)
        risk_probs, risk_weight = self._risk_exploration_probs(
            score_terms["dynamic_risk"],
            candidate_row,
            allowed,
        )
        eps_unknown = self.unknown_exploration_prob if enable_exploration and not deterministic else 0.0
        eps_risk = self.risk_exploration_prob if enable_exploration and not deterministic else 0.0
        if float(unknown_probs.sum().detach().item()) <= 1e-8:
            eps_unknown = 0.0
        if float(risk_probs.sum().detach().item()) <= 1e-8:
            eps_risk = 0.0
        eps_total = eps_unknown + eps_risk
        if eps_total <= 0.0:
            return base_probs, unknown_probs, unknown_weight, risk_probs, risk_weight
        if eps_total >= 0.95:
            scale = 0.95 / eps_total
            eps_unknown *= scale
            eps_risk *= scale
            eps_total = eps_unknown + eps_risk
        mixed = (
            (1.0 - eps_total) * base_probs
            + eps_unknown * unknown_probs.to(device=base_probs.device, dtype=base_probs.dtype)
            + eps_risk * risk_probs.to(device=base_probs.device, dtype=base_probs.dtype)
        )
        mixed = mixed.masked_fill(~allowed.bool(), 0.0)
        mixed_sum = mixed.sum()
        if float(mixed_sum.detach().item()) <= 1e-8:
            return base_probs, unknown_probs, unknown_weight, risk_probs, risk_weight
        return mixed / mixed_sum.clamp_min(1e-8), unknown_probs, unknown_weight, risk_probs, risk_weight

    def _project_offloading_actions(
            self,
            proposal_actions: torch.Tensor,
            effective_mask: torch.Tensor,
            policy_probs: torch.Tensor,
            parents: List[List[int]],
            topo_order: List[int],
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """
        Final offloading safety projection.

        The actor already applies dynamic dependency masks before sampling. This
        projection is intentionally a defensive execution step: it catches stale
        masks, invalid targets, and dependency cascades caused by an earlier
        projected parent without adding an extra reward term.
        """
        projected = proposal_actions.clone().long()
        cloud_idx = self._cloud_index(effective_mask.size(1))
        reasons = ["none" for _ in range(int(projected.numel()))]
        projection_cnt = 0
        dependency_projection_cnt = 0
        infeasible_projection_cnt = 0
        projection_cost = 0.0

        for service_idx in topo_order:
            proposal = int(proposal_actions[service_idx].item())
            target = proposal
            reason_parts: List[str] = []

            allowed = effective_mask[service_idx].clone().bool()
            if not bool(allowed.any().item()):
                allowed[cloud_idx] = True

            proposal_feasible = 0 <= proposal < allowed.numel() and bool(allowed[proposal].item())
            if not proposal_feasible:
                target = self._best_allowed_offloading_target(policy_probs[service_idx], allowed)
                infeasible_projection_cnt += 1
                reason_parts.append("static_infeasible")

            parent_cloud = any(
                0 <= int(projected[parent_idx].item()) == cloud_idx
                for parent_idx in parents[service_idx]
            )
            if parent_cloud and target != cloud_idx:
                target = cloud_idx
                dependency_projection_cnt += 1
                reason_parts.append("dependency_cloud")

            if target < 0 or target >= allowed.numel():
                target = cloud_idx
                reason_parts.append("fallback_cloud")

            projected[service_idx] = target
            if target != proposal:
                projection_cnt += 1
                if 0 <= target < policy_probs.size(1):
                    prob = float(policy_probs[service_idx, target].detach().item())
                else:
                    prob = 0.0
                prob = min(max(prob, 1e-6), 1.0)
                projection_cost += -math.log(prob)
                reasons[service_idx] = "+".join(reason_parts) if reason_parts else "projected"

        num_services = max(1, int(proposal_actions.numel()))
        return projected, {
            "projection_cnt": int(projection_cnt),
            "dependency_projection_cnt": int(dependency_projection_cnt),
            "infeasible_projection_cnt": int(infeasible_projection_cnt),
            "projection_cost": float(projection_cost / float(num_services)),
            "projection_reasons": reasons,
        }

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
               static_mask: torch.Tensor, topo_order: Optional[list] = None,
               deterministic: bool = False,
               enable_exploration: bool = False):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        effective_mask = self._effective_offloading_mask(static_mask)

        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)
        parents = self._build_parents(logic_edge_index, Ms)
        (
            q_embedding,
            k_embedding,
            qk_scores,
            qk_feature,
            runtime_feat,
            demand_feat,
        ) = self._offloading_static_terms(
            h_s,
            h_p,
            logic_feats,
            effective_mask,
        )
        policy_prob_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        base_policy_prob_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        unknown_policy_prob_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        unknown_exploration_weight_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        risk_policy_prob_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        risk_exploration_weight_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        final_scores = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        static_priors = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        runtime_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        service_time_factors = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        runtime_recencies = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        queue_freshnesses = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        speed_evidences = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        capacity_pressures = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        load_pressures = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        base_queue_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        relative_queue_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        overload_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        queue_risk_totals = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        planned_pressures = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        planned_load_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        relative_planned_load_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        offered_load_pressures = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        offered_load_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        compute_relative_weaknesses = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        runtime_relative_weaknesses = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        relative_weaknesses = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        weak_pressures = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        weak_replica_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        cloud_penalties = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        cross_tier_penalty_terms = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        dynamic_risks = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        planned_device_loads = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        candidate_features_matrix = torch.zeros(
            (Ms, Np, len(OFFLOADING_CANDIDATE_FEATURE_NAMES)),
            dtype=h_s.dtype,
            device=h_p.device,
        )

        proposal_action_list = [-1 for _ in range(Ms)]
        logp_sum = torch.tensor(0.0, device=h_p.device)
        ent_sum = torch.tensor(0.0, device=h_p.device)
        planned_device_load = torch.zeros((Np,), dtype=torch.float32, device=h_p.device)

        for i in topo_order:
            allowed = self._dynamic_allowed_row(effective_mask[i], parents[i], proposal_action_list)
            candidate_row = self._offloading_candidate_row(
                logic_feats,
                phys_feats,
                qk_feature,
                runtime_feat,
                demand_feat,
                parents,
                i,
                proposal_action_list,
                allowed=allowed,
            )
            score_terms = self._score_offloading_candidates(candidate_row, planned_device_load, allowed)
            candidate_scores = score_terms["final_score"]
            candidate_features_matrix[i] = candidate_row
            static_priors[i] = score_terms["static_prior"]
            runtime_risks[i] = score_terms["runtime_risk"]
            service_time_factors[i] = score_terms["service_time_factor"]
            runtime_recencies[i] = self._offloading_feature(candidate_row, "runtime_recency").float()
            queue_freshnesses[i] = self._offloading_feature(candidate_row, "queue_freshness").float()
            speed_evidences[i] = self._offloading_feature(candidate_row, "speed_evidence").float()
            capacity_pressures[i] = self._offloading_feature(candidate_row, "capacity_pressure").float()
            load_pressures[i] = score_terms["load_pressure"]
            base_queue_risks[i] = score_terms["base_queue_risk"]
            relative_queue_risks[i] = score_terms["relative_queue_risk"]
            overload_risks[i] = score_terms["overload_risk"]
            queue_risk_totals[i] = score_terms["queue_risk_total"]
            planned_pressures[i] = score_terms["planned_pressure"]
            planned_load_risks[i] = score_terms["planned_load_risk"]
            relative_planned_load_risks[i] = score_terms["relative_planned_load_risk"]
            offered_load_pressures[i] = score_terms["offered_load_pressure"]
            offered_load_risks[i] = score_terms["offered_load_risk"]
            compute_relative_weaknesses[i] = score_terms["compute_relative_weakness"]
            runtime_relative_weaknesses[i] = score_terms["runtime_relative_weakness"]
            relative_weaknesses[i] = score_terms["relative_weakness"]
            weak_pressures[i] = score_terms["weak_pressure"]
            weak_replica_risks[i] = score_terms["weak_replica_risk"]
            cloud_penalties[i] = score_terms["cloud_penalty"]
            cross_tier_penalty_terms[i] = score_terms["cross_tier_penalty_term"]
            dynamic_risks[i] = score_terms["dynamic_risk"]
            planned_device_loads[i] = planned_device_load
            final_scores[i] = candidate_scores
            effective_mask[i] = allowed
            masked_scores = candidate_scores.masked_fill(~allowed, float('-inf'))
            base_probs_i = F.softmax(masked_scores, dim=-1)
            probs_i, unknown_probs_i, unknown_weight_i, risk_probs_i, risk_weight_i = self._mix_offloading_exploration_probs(
                base_probs_i,
                candidate_row,
                score_terms,
                allowed,
                enable_exploration=enable_exploration,
                deterministic=deterministic,
            )
            base_policy_prob_matrix[i] = base_probs_i
            unknown_policy_prob_matrix[i] = unknown_probs_i
            unknown_exploration_weight_matrix[i] = unknown_weight_i
            risk_policy_prob_matrix[i] = risk_probs_i
            risk_exploration_weight_matrix[i] = risk_weight_i
            policy_prob_matrix[i] = probs_i
            dist = torch.distributions.Categorical(probs=probs_i)
            sampled_action = torch.argmax(probs_i) if deterministic else dist.sample()
            sampled_idx = int(sampled_action.item())
            proposal_action_list[i] = sampled_idx
            if 0 <= sampled_idx < Np:
                load_increment = self._offloading_planned_load_increment(
                    candidate_row[sampled_idx].float()
                ).detach()
                next_planned_load = planned_device_load.clone()
                next_planned_load[sampled_idx] = (
                    next_planned_load[sampled_idx] + load_increment
                ).clamp(max=self.planned_load_clip)
                planned_device_load = next_planned_load
            logp_sum += dist.log_prob(sampled_action)
            ent_sum += dist.entropy()

        proposal_actions = torch.tensor(proposal_action_list, dtype=torch.long, device=h_p.device)

        projected_actions, projection_info = self._project_offloading_actions(
            proposal_actions,
            effective_mask,
            policy_prob_matrix,
            parents,
            topo_order,
        )
        selected_queue_metrics = self._selected_offloading_queue_metrics(
            candidate_features_matrix,
            projected_actions,
            score_matrices={
                "base_queue_risk": base_queue_risks,
                "relative_queue_risk": relative_queue_risks,
                "overload_risk": overload_risks,
                "queue_risk_total": queue_risk_totals,
                "planned_load_risk": planned_load_risks,
                "relative_planned_load_risk": relative_planned_load_risks,
                "offered_load_pressure": offered_load_pressures,
                "offered_load_risk": offered_load_risks,
                "compute_relative_weakness": compute_relative_weaknesses,
                "runtime_relative_weakness": runtime_relative_weaknesses,
                "relative_weakness": relative_weaknesses,
                "weak_pressure": weak_pressures,
                "weak_replica_risk": weak_replica_risks,
                "dynamic_risk": dynamic_risks,
            },
        )

        value_candidates = self._offloading_value_candidate_features(
            logic_edge_index, logic_feats, phys_feats, qk_feature, runtime_feat
        )
        value = self.critic(h_s, h_p, value_candidates, effective_mask)
        return projected_actions, logp_sum, ent_sum, value.squeeze(0), {
            "proposal_actions": proposal_actions.detach(),
            "projected_actions": projected_actions.detach(),
            **projection_info,
            **selected_queue_metrics,
            "actor_debug": {
                "service_embedding": h_s.detach().cpu(),
                "device_embedding": h_p.detach().cpu(),
                "q_embedding": q_embedding.detach().cpu(),
                "k_embedding": k_embedding.detach().cpu(),
                "qk_score": qk_scores.detach().cpu(),
                "qk_feature": qk_feature.detach().cpu(),
                "static_prior": static_priors.detach().cpu(),
                "runtime_risk": runtime_risks.detach().cpu(),
                "service_time_factor": service_time_factors.detach().cpu(),
                "runtime_recency": runtime_recencies.detach().cpu(),
                "queue_freshness": queue_freshnesses.detach().cpu(),
                "speed_evidence": speed_evidences.detach().cpu(),
                "capacity_pressure": capacity_pressures.detach().cpu(),
                "load_pressure": load_pressures.detach().cpu(),
                "base_queue_risk": base_queue_risks.detach().cpu(),
                "relative_queue_risk": relative_queue_risks.detach().cpu(),
                "overload_risk": overload_risks.detach().cpu(),
                "queue_risk_total": queue_risk_totals.detach().cpu(),
                "planned_pressure": planned_pressures.detach().cpu(),
                "planned_load_risk": planned_load_risks.detach().cpu(),
                "relative_planned_load_risk": relative_planned_load_risks.detach().cpu(),
                "offered_load_pressure": offered_load_pressures.detach().cpu(),
                "offered_load_risk": offered_load_risks.detach().cpu(),
                "compute_relative_weakness": compute_relative_weaknesses.detach().cpu(),
                "runtime_relative_weakness": runtime_relative_weaknesses.detach().cpu(),
                "relative_weakness": relative_weaknesses.detach().cpu(),
                "weak_pressure": weak_pressures.detach().cpu(),
                "weak_replica_risk": weak_replica_risks.detach().cpu(),
                "cloud_penalty": cloud_penalties.detach().cpu(),
                "cross_tier_penalty_term": cross_tier_penalty_terms.detach().cpu(),
                "dynamic_risk": dynamic_risks.detach().cpu(),
                "planned_device_load": planned_device_loads.detach().cpu(),
                "candidate_feature": candidate_features_matrix.detach().cpu(),
                "candidate_feature_names": OFFLOADING_CANDIDATE_FEATURE_NAMES,
                "runtime_pair_feature_names": RUNTIME_PAIR_FEATURE_NAMES,
                "service_demand_feature": demand_feat.detach().cpu(),
                "service_demand_feature_names": SERVICE_DEMAND_FEATURE_NAMES,
                "device_capability_feature": _debug_tensor(phys_feats, "device_capability_feat"),
                "device_capability_feature_names": DEVICE_CAPABILITY_FEATURE_NAMES,
                "final_score": final_scores.detach().cpu(),
                "base_policy_prob": base_policy_prob_matrix.detach().cpu(),
                "unknown_policy_prob": unknown_policy_prob_matrix.detach().cpu(),
                "unknown_exploration_weight": unknown_exploration_weight_matrix.detach().cpu(),
                "unknown_exploration_eps": torch.full(
                    (Ms, Np),
                    fill_value=float(self.unknown_exploration_prob if enable_exploration and not deterministic else 0.0),
                    dtype=torch.float32,
                ),
                "risk_policy_prob": risk_policy_prob_matrix.detach().cpu(),
                "risk_exploration_weight": risk_exploration_weight_matrix.detach().cpu(),
                "risk_exploration_eps": torch.full(
                    (Ms, Np),
                    fill_value=float(self.risk_exploration_prob if enable_exploration and not deterministic else 0.0),
                    dtype=torch.float32,
                ),
                "policy_prob": policy_prob_matrix.detach().cpu(),
                "raw_static_mask": static_mask.detach().cpu(),
                "effective_mask": effective_mask.detach().cpu(),
                "proposal_action": proposal_actions.detach().cpu(),
                "projected_action": projected_actions.detach().cpu(),
            },
        }

    def evaluate(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
                 proposal_actions: torch.Tensor, static_mask: torch.Tensor, topo_order: Optional[list] = None,
                 enable_exploration: bool = False):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        effective_mask = self._effective_offloading_mask(static_mask)
        proposal_action_list = [int(action) for action in proposal_actions.detach().cpu().long().tolist()]

        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)
        planned_device_load = torch.zeros((Np,), dtype=torch.float32, device=h_p.device)

        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)
        parents = self._build_parents(logic_edge_index, Ms)
        (
            _q_embedding,
            _k_embedding,
            _qk_scores,
            qk_feature,
            runtime_feat,
            demand_feat,
        ) = self._offloading_static_terms(
            h_s,
            h_p,
            logic_feats,
            effective_mask,
        )

        for i in topo_order:
            allowed = self._dynamic_allowed_row(effective_mask[i], parents[i], proposal_action_list)
            candidate_row = self._offloading_candidate_row(
                logic_feats,
                phys_feats,
                qk_feature,
                runtime_feat,
                demand_feat,
                parents,
                i,
                proposal_action_list,
                allowed=allowed,
            )
            score_terms = self._score_offloading_candidates(
                candidate_row,
                planned_device_load,
                allowed,
            )
            scores_i = score_terms["final_score"]
            base_probs_i = F.softmax(scores_i.masked_fill(~allowed, float('-inf')), dim=-1)
            probs_i, _, _, _, _ = self._mix_offloading_exploration_probs(
                base_probs_i,
                candidate_row,
                score_terms,
                allowed,
                enable_exploration=enable_exploration,
                deterministic=False,
            )
            dist = torch.distributions.Categorical(probs=probs_i)
            a = torch.tensor(proposal_action_list[i], dtype=torch.long, device=h_p.device)
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()
            action_idx = int(a.item())
            if 0 <= action_idx < Np:
                load_increment = self._offloading_planned_load_increment(
                    candidate_row[action_idx].float()
                ).detach()
                next_planned_load = planned_device_load.clone()
                next_planned_load[action_idx] = (
                    next_planned_load[action_idx] + load_increment
                ).clamp(max=self.planned_load_clip)
                planned_device_load = next_planned_load

        value_candidates = self._offloading_value_candidate_features(
            logic_edge_index, logic_feats, phys_feats, qk_feature, runtime_feat
        )
        value = self.critic(h_s, h_p, value_candidates, effective_mask)
        return logp_sum, ent_sum, value.squeeze(0), {}

    def ppo_update(self, transitions: List[dict], epochs=4, batch_size=32, clip_eps=None, entropy_coef=0.01,
                   value_coef=0.5):
        clip_eps = self.clip_eps if clip_eps is None else clip_eps
        device = next(self.parameters()).device
        old_logp = torch.stack([t['logp'].detach().to(device) for t in transitions])  # [T]
        old_val = [_scalar_to_float(t['value']) for t in transitions]
        rewards = [float(t['reward']) for t in transitions]
        dones = [t['done'] for t in transitions]
        last_value = 0.0 if dones[-1] else _scalar_to_float(transitions[-1].get("next_value", 0.0))
        adv, rets = compute_returns_advantages(rewards, old_val, dones, self.gamma, self.lamda, last_value=last_value)
        adv_raw = torch.tensor(adv, device=device)
        rets = torch.tensor(rets, device=device)
        adv_raw_mean = adv_raw.mean()
        adv_raw_std = adv_raw.std(unbiased=False)
        adv = (adv_raw - adv_raw_mean) / (adv_raw_std + 1e-6)
        device_transitions = [
            {
                "logic_edge_index": tr["logic_edge_index"].to(device),
                "logic_feats": _move_tensor_dict_to_device(tr["logic_feats"], device),
                "phys_edge_index": tr["phys_edge_index"].to(device),
                "phys_feats": _move_tensor_dict_to_device(tr["phys_feats"], device),
                "proposal_actions": tr["proposal_actions"].to(device),
                "static_mask": tr["static_mask"].to(device),
                "topo_order": tr["topo_order"],
                "exploration_enabled": bool(tr.get("exploration_enabled", False)),
            }
            for tr in transitions
        ]
        T = len(transitions)
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        approx_kls: List[float] = []
        clip_fractions: List[float] = []
        ratio_means: List[float] = []
        ratio_stds: List[float] = []
        actor_grad_norms: List[float] = []
        critic_grad_norms: List[float] = []
        new_value_means: List[float] = []
        for _ in range(epochs):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, batch_size):
                idx = perm[start:start + batch_size]
                new_logp_list = []
                new_val_list = []
                ent_list = []
                for j in idx:
                    tr = device_transitions[int(j)]
                    lp, ent, val, _ = self.evaluate(tr['logic_edge_index'], tr['logic_feats'],
                                                    tr['phys_edge_index'], tr['phys_feats'],
                                                    tr['proposal_actions'], tr['static_mask'], tr['topo_order'],
                                                    enable_exploration=tr["exploration_enabled"])
                    new_logp_list.append(lp)
                    new_val_list.append(val)
                    ent_list.append(ent)
                new_logp = torch.stack(new_logp_list)
                new_val = torch.stack(new_val_list).squeeze(-1)
                ent = torch.stack(ent_list).mean()
                ratio = torch.exp(new_logp - old_logp[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[idx]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(new_val, rets[idx])
                loss = policy_loss + value_coef * value_loss - entropy_coef * ent

                with torch.no_grad():
                    approx_kl = (old_logp[idx] - new_logp).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()
                    policy_losses.append(_scalar_to_float(policy_loss))
                    value_losses.append(_scalar_to_float(value_loss))
                    entropies.append(_scalar_to_float(ent))
                    approx_kls.append(_scalar_to_float(approx_kl))
                    clip_fractions.append(_scalar_to_float(clip_fraction))
                    ratio_means.append(_scalar_to_float(ratio.mean()))
                    ratio_stds.append(_tensor_std_float(ratio))
                    new_value_means.append(_scalar_to_float(new_val.mean()))

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(self._actor_train_params, 1.0)
                actor_grad_norms.append(_scalar_to_float(actor_grad_norm))
                critic_grad_norms.append(_parameters_grad_norm(self.critic.parameters()))
                self.actor_opt.step()
                self.critic_opt.step()

        return {
            "samples": T,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "minibatches": len(policy_losses),
            "reward_mean": _mean_or_zero(rewards),
            "reward_std": _std_or_zero(rewards),
            "reward_min": float(min(rewards)) if rewards else 0.0,
            "reward_max": float(max(rewards)) if rewards else 0.0,
            "value_old_mean": _mean_or_zero(old_val),
            "value_old_std": _std_or_zero(old_val),
            "value_new_mean": _mean_or_zero(new_value_means),
            "return_mean": _scalar_to_float(rets.mean()),
            "return_std": _tensor_std_float(rets),
            "adv_mean": _scalar_to_float(adv_raw_mean),
            "adv_std": _scalar_to_float(adv_raw_std),
            "last_value": float(last_value),
            "done_fraction": _mean_or_zero([1.0 if done else 0.0 for done in dones]),
            "policy_loss": _mean_or_zero(policy_losses),
            "value_loss": _mean_or_zero(value_losses),
            "entropy": _mean_or_zero(entropies),
            "entropy_coef": float(entropy_coef),
            "value_coef": float(value_coef),
            "approx_kl": _mean_or_zero(approx_kls),
            "clip_fraction": _mean_or_zero(clip_fractions),
            "ratio_mean": _mean_or_zero(ratio_means),
            "ratio_std": _mean_or_zero(ratio_stds),
            "actor_grad_norm": _mean_or_zero(actor_grad_norms),
            "critic_grad_norm": _mean_or_zero(critic_grad_norms),
        }
