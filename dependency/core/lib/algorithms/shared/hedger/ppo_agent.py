from typing import Optional, List, Dict, Tuple, Sequence, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_encoder import TopologyEncoders
from .ppo_network import DeploymentActor, OffloadActor
from .hedger_config import DeploymentConstraintCfg
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
    "runtime_freshness",
]

OFFLOADING_CANDIDATE_FEATURE_NAMES = [
    "qk_feature",
    "compute_gap",
    "queue_short",
    "queue_busy",
    "real_time_per_complexity",
    "load_score",
    "runtime_confidence",
    "runtime_freshness",
    "parent_same_device",
    "cross_tier_penalty",
    "is_cloud",
]

DEPLOYMENT_CANDIDATE_FEATURE_NAMES = [
    "qk_feature",
    "compute_gap",
    "mem_gap",
    "residual_after_place",
    "current_replica",
    "service_replica_count",
    "device_replica_count",
    "pair_queue_short",
    "pair_real_time_per_complexity",
    "parent_child_coverage",
    "is_cloud",
    "static_allowed",
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
     runtime_confidence, runtime_freshness].
    """
    existing = logic_feats.get("runtime_pair_feat")
    if isinstance(existing, torch.Tensor) and existing.dim() == 3 \
            and existing.size(0) == num_services and existing.size(1) == num_devices and existing.size(2) >= 5:
        return existing[..., :5].to(device=device, dtype=dtype)
    raise ValueError("Hedger state is missing `runtime_pair_feat`; collect state with the current runtime builder.")


def _masked_standardize(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    mask_f = mask.float()
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (scores * mask_f).sum(dim=1, keepdim=True) / counts
    var = (((scores - mean) * mask_f) ** 2).sum(dim=1, keepdim=True) / counts
    standardized = torch.clamp((scores - mean) / torch.sqrt(var + 1e-6), min=-5.0, max=5.0)
    return standardized * mask_f


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
    """Shared candidate scorer. The policy uses `-cost` as the action logit."""
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
        self.deployment_cost_head = CandidateCostHead(
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

    def _rebuild_actor_optimizer(self, extra_actor_modules: Optional[List[nn.Module]] = None):
        extra_actor_modules = extra_actor_modules or []
        params_actor = list(self.actor.parameters())
        params_actor.extend(list(self.deployment_cost_head.parameters()))
        for module in extra_actor_modules:
            params_actor.extend(list(module.parameters()))
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

    def _min_edge_replicas_per_service(self) -> int:
        value = getattr(self.cfg, "min_edge_replicas_per_service", 0)
        if value is None:
            return 0
        value = int(value)
        return max(0, value)

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

    def _deployment_topology_coverage(
            self,
            logic_edge_index: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor],
            num_services: int,
            num_devices: int,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        parents, children, _ = _topology_context(logic_edge_index, num_services)
        if prev_deploy_mask is None:
            prev_mask = torch.zeros((num_services, num_devices), device=device, dtype=torch.bool)
        else:
            prev_mask = prev_deploy_mask.to(device=device).bool()

        coverage = torch.zeros((num_services, num_devices, 2), device=device, dtype=dtype)
        for service_idx in range(num_services):
            for device_idx in range(num_devices):
                parent_coverage = 0.0
                if parents[service_idx]:
                    matched = sum(1 for parent_idx in parents[service_idx] if bool(prev_mask[parent_idx, device_idx]))
                    parent_coverage = matched / float(len(parents[service_idx]))

                child_coverage = 0.0
                if children[service_idx]:
                    matched = sum(1 for child_idx in children[service_idx] if bool(prev_mask[child_idx, device_idx]))
                    child_coverage = matched / float(len(children[service_idx]))

                coverage[service_idx, device_idx, 0] = parent_coverage
                coverage[service_idx, device_idx, 1] = child_coverage
        return coverage

    def _deployment_candidate_features(
            self,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            qk_feature: torch.Tensor,
            static_allowed: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_services, num_devices = qk_feature.shape
        device = qk_feature.device
        dtype = qk_feature.dtype
        cloud_idx = self._cloud_index(num_devices)
        demand_feat = _service_demand_features(logic_feats, num_services, device, dtype)
        runtime_feat = _runtime_pair_features(logic_feats, num_services, num_devices, device, dtype)
        model_mem = _feature_vector(logic_feats, "model_mem", num_services, device, dtype)
        device_feat = _device_capability_features(phys_feats, num_devices, device, dtype, cloud_idx)
        _, is_cloud = _role_tensors(phys_feats, num_devices, device, dtype, cloud_idx)

        residual_mem = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask).to(device=device, dtype=dtype)
        if prev_deploy_mask is None:
            prev_mask = torch.zeros((num_services, num_devices), device=device, dtype=dtype)
        else:
            prev_mask = prev_deploy_mask.to(device=device, dtype=dtype)
        current_replica_count = prev_mask.sum(dim=1).clamp_min(0.0)
        device_service_count = prev_mask.sum(dim=0).clamp_min(0.0)

        compute_gap = demand_feat[:, 0].view(num_services, 1) - device_feat[:, 0].view(1, num_devices)
        mem_gap = demand_feat[:, 3].view(num_services, 1) - torch.log1p(
            residual_mem.clamp_min(0.0)
        ).view(1, num_devices)
        residual_after_place = torch.log1p(
            (residual_mem.view(1, num_devices) - model_mem.view(num_services, 1)).clamp_min(0.0)
        )
        pair_queue_short = runtime_feat[..., 0]
        pair_real_time_per_complexity = runtime_feat[..., 2]
        topology_coverage = self._deployment_topology_coverage(
            logic_edge_index, prev_deploy_mask, num_services, num_devices, device, dtype
        )
        parent_child_coverage = 0.5 * (topology_coverage[..., 0] + topology_coverage[..., 1])
        cloud_role = is_cloud.view(1, num_devices).expand(num_services, -1)
        static_allowed_float = static_allowed.to(device=device, dtype=dtype)
        return torch.stack(
            [
                qk_feature,
                compute_gap,
                mem_gap,
                residual_after_place,
                prev_mask,
                torch.log1p(current_replica_count).view(num_services, 1).expand(-1, num_devices),
                torch.log1p(device_service_count).view(1, num_devices).expand(num_services, -1),
                pair_queue_short,
                pair_real_time_per_complexity,
                parent_child_coverage,
                cloud_role,
                static_allowed_float,
            ],
            dim=-1,
        )

    def _deployment_actor_terms(
            self,
            h_s: torch.Tensor,
            h_p: torch.Tensor,
            logic_edge_index: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            static_allowed: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_embedding, k_embedding, qk_scores, qk_feature = self._qk_components(h_s, h_p, static_allowed)
        candidate_features = self._deployment_candidate_features(
            logic_edge_index,
            logic_feats,
            phys_feats,
            qk_feature,
            static_allowed,
            prev_deploy_mask=prev_deploy_mask,
        )
        candidate_cost = self.deployment_cost_head(candidate_features.float())
        final_scores = -candidate_cost
        return q_embedding, k_embedding, qk_scores, qk_feature, candidate_cost, candidate_features, final_scores

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
        _, _, _, _, _, candidate_features, _ = self._deployment_actor_terms(
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

    def _project_deployment_mask(
            self,
            raw_deploy_mask: torch.Tensor,
            raw_probs: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor] = None,
            static_allowed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, float, int, float, int]:
        """
        Project a sampled deployment mask back into memory-feasible space.

        The projection is deterministic given the sampled deployment and the
        actor probabilities for sampled replicas:

        - For each edge device, consider only the services whose raw Bernoulli
          sample selected that device.
        - Remove the lowest-preference sampled replicas until the remaining
          subset fits within the device residual-memory and replica-count budget.

        This keeps the executed placement close to the raw sample while still
        reflecting the actor's per-service preference ordering. Previous
        deployment state and memory footprint are only used as tie-breakers.

        Besides the removed replica count, the projector also reports a
        correction cost. Each removed sampled replica contributes
        `-log(1 - p_ij)`, i.e. the negative log-probability of the corrected
        zero action under the raw Bernoulli policy. The final cost is normalized
        by the number of logical services so it can be used directly in reward
        shaping across DAGs of different sizes.

        If configured, a second repair step then tries to give each service a
        minimum number of edge replicas while respecting the same memory and
        per-device replica limits.
        """
        corrected = self._enforce_cloud_replica(raw_deploy_mask.bool())
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask)
        model_mem = logic_feats["model_mem"].float()
        cloud_idx = self._cloud_index(corrected.size(1))
        max_edge_replicas = self._max_edge_replicas_per_device()
        capacity_relax_cnt = 0
        capacity_relax_cost = 0.0
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
                prev_selected=prev_edge_mask[:, device_idx],
                max_count=max_edge_replicas,
            )

            for service_idx in selected:
                if service_idx not in keep:
                    corrected[service_idx, device_idx] = False
                    capacity_relax_cnt += 1
                    prob = float(raw_probs[service_idx, device_idx].item())
                    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
                    capacity_relax_cost += -math.log(max(1.0 - prob, 1e-6))

        capacity_relax_cost /= float(num_services)
        edge_cover_repair_cnt, edge_cover_repair_cost, edge_cover_unmet = \
            self._repair_min_edge_replicas_per_service(
                corrected=corrected,
                raw_deploy_mask=raw_deploy_mask.bool(),
                raw_probs=raw_probs,
                logic_feats=logic_feats,
                phys_feats=phys_feats,
                prev_deploy_mask=prev_deploy_mask,
                static_allowed=static_allowed,
            )
        return (
            corrected,
            capacity_relax_cnt,
            capacity_relax_cost,
            edge_cover_repair_cnt,
            edge_cover_repair_cost,
            edge_cover_unmet,
        )

    def _repair_min_edge_replicas_per_service(
            self,
            corrected: torch.Tensor,
            raw_deploy_mask: torch.Tensor,
            raw_probs: torch.Tensor,
            logic_feats: Dict[str, torch.Tensor],
            phys_feats: Dict[str, torch.Tensor],
            prev_deploy_mask: Optional[torch.Tensor] = None,
            static_allowed: Optional[torch.Tensor] = None,
    ) -> Tuple[int, float, int]:
        min_edge_replicas = self._min_edge_replicas_per_service()
        if min_edge_replicas <= 0:
            return 0, 0.0, 0

        num_services = max(1, int(corrected.size(0)))
        cloud_idx = self._cloud_index(corrected.size(1))
        if cloud_idx <= 0:
            return 0, 0.0, num_services

        if static_allowed is None:
            static_allowed = self._static_allowed_mask(phys_feats, logic_feats)
        static_allowed = static_allowed.bool()
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask)
        model_mem = logic_feats["model_mem"].float()
        max_edge_replicas = self._max_edge_replicas_per_device()

        used_count = corrected[:, :cloud_idx].sum(dim=0).to(torch.long)
        used_mem = torch.matmul(corrected[:, :cloud_idx].float().t(), model_mem)
        if prev_deploy_mask is None:
            prev_edge_mask = torch.zeros_like(corrected)
        else:
            prev_edge_mask = prev_deploy_mask.bool()

        repair_cnt = 0
        repair_cost = 0.0
        unmet_services = 0

        service_order = []
        for service_idx in range(corrected.size(0)):
            current_edges = int(corrected[service_idx, :cloud_idx].sum().item())
            if current_edges >= min_edge_replicas:
                continue
            service_mem = float(model_mem[service_idx].item())
            feasible_count = 0
            best_prob = 0.0
            for device_idx in range(cloud_idx):
                if bool(corrected[service_idx, device_idx].item()):
                    continue
                if not bool(static_allowed[service_idx, device_idx].item()):
                    continue
                if max_edge_replicas is not None and int(used_count[device_idx].item()) >= max_edge_replicas:
                    continue
                remaining_after = float(residual[device_idx].item()) - float(used_mem[device_idx].item()) - service_mem
                if remaining_after < -1e-6:
                    continue
                feasible_count += 1
                best_prob = max(best_prob, float(raw_probs[service_idx, device_idx].item()))
            service_order.append((
                feasible_count if feasible_count > 0 else cloud_idx + 1,
                -service_mem,
                -best_prob,
                int(service_idx),
            ))

        for _, _, _, service_idx in sorted(service_order):
            current_edges = int(corrected[service_idx, :cloud_idx].sum().item())
            service_mem = float(model_mem[service_idx].item())
            while current_edges < min_edge_replicas:
                candidates = []
                for device_idx in range(cloud_idx):
                    if bool(corrected[service_idx, device_idx].item()):
                        continue
                    if not bool(static_allowed[service_idx, device_idx].item()):
                        continue
                    if max_edge_replicas is not None and int(used_count[device_idx].item()) >= max_edge_replicas:
                        continue
                    remaining_after = float(residual[device_idx].item()) - float(used_mem[device_idx].item()) - service_mem
                    if remaining_after < -1e-6:
                        continue

                    raw_prob = float(raw_probs[service_idx, device_idx].item())
                    raw_prob = min(max(raw_prob, 1e-6), 1.0 - 1e-6)
                    was_selected = bool(prev_edge_mask[service_idx, device_idx].item())
                    device_is_empty = int(used_count[device_idx].item()) == 0
                    candidates.append((
                        raw_prob,
                        1 if was_selected else 0,
                        1 if device_is_empty else 0,
                        remaining_after,
                        -int(device_idx),
                        int(device_idx),
                    ))

                if not candidates:
                    unmet_services += 1
                    break

                _, _, _, _, _, best_device = max(candidates)
                corrected[service_idx, best_device] = True
                used_count[best_device] += 1
                used_mem[best_device] += service_mem
                current_edges += 1
                repair_cnt += 1

                if not bool(raw_deploy_mask[service_idx, best_device].item()):
                    prob = float(raw_probs[service_idx, best_device].item())
                    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
                    repair_cost += -math.log(prob)

        repair_cost /= float(num_services)
        return repair_cnt, repair_cost, unmet_services

    def _select_device_subset(
            self,
            selected: List[int],
            capacity: float,
            model_mem: torch.Tensor,
            raw_probs: torch.Tensor,
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
            mem = float(model_mem[service_idx].item())
            was_selected = bool(prev_selected[service_idx].item())
            candidates.append((int(service_idx), prob, mem, was_selected))
            total_mem += mem

        keep = {int(service_idx) for service_idx in selected}
        ranked_for_removal = sorted(
            candidates,
            key=lambda item: (
                item[1],  # Lower actor preference is removed first.
                1 if item[3] else 0,  # Prefer removing non-previous replicas on ties.
                -item[2],  # If preference is tied, remove the larger footprint first.
                -item[0],  # Keep lower service ids for deterministic ties.
            ),
        )

        for service_idx, _, mem, _ in ranked_for_removal:
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

        # Static feasibility mask.
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]
        (
            q_embedding,
            k_embedding,
            qk_scores,
            qk_feature,
            candidate_cost,
            candidate_features,
            final_scores,
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
            final_scores = final_scores + torch.randn_like(final_scores) * float(logit_noise_std)
        policy_prob_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_s.device)

        # Sample first from the static deployment distribution, then apply a
        # deterministic capacity projection. The executed post-projection mask
        # is the default training target for offline, online, and PPO updates.
        raw_deploy_mask = torch.zeros((Ms, Np), dtype=torch.bool, device=h_s.device)
        raw_probs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_s.device)
        logp_sum = torch.tensor(0.0, device=h_s.device)
        ent_terms = []
        eps = 1e-6
        for service_idx in topo_order:
            stochastic_allowed = static_allowed[service_idx].clone()
            stochastic_allowed[cloud_idx] = False
            sampled_row = torch.zeros(Np, dtype=torch.bool, device=h_s.device)
            if stochastic_allowed.any():
                row_scores = final_scores[service_idx]
                probs_raw = torch.where(
                    stochastic_allowed,
                    torch.sigmoid(row_scores),
                    torch.zeros_like(row_scores),
                )
                probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
                probs_log = torch.where(stochastic_allowed, probs_log, torch.full_like(probs_log, eps))
                probs_sample = torch.where(stochastic_allowed, probs_raw, torch.zeros_like(probs_raw))
                dist = torch.distributions.Bernoulli(probs=probs_log)
                if deterministic:
                    sampled_bits = probs_sample >= 0.5
                else:
                    sampled_bits = torch.rand_like(probs_sample) < probs_sample
                sampled_row = torch.where(
                    stochastic_allowed,
                    sampled_bits,
                    torch.zeros_like(sampled_bits, dtype=torch.bool),
                )
                logp_sum += dist.log_prob(sampled_row.float()).sum()
                ent_terms.append(dist.entropy()[stochastic_allowed].mean())
                raw_probs[service_idx] = probs_raw
                policy_prob_matrix[service_idx] = probs_raw
            sampled_row[cloud_idx] = True
            raw_deploy_mask[service_idx] = sampled_row
            raw_probs[service_idx, cloud_idx] = 1.0
            policy_prob_matrix[service_idx, cloud_idx] = 1.0

        ent_sum = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=h_s.device)
        (
            deploy_mask,
            capacity_relax_cnt,
            capacity_relax_cost,
            edge_cover_repair_cnt,
            edge_cover_repair_cost,
            edge_cover_unmet,
        ) = self._project_deployment_mask(
            raw_deploy_mask,
            raw_probs=raw_probs,
            logic_feats=logic_feats,
            phys_feats=phys_feats,
            prev_deploy_mask=prev_deploy_mask,
            static_allowed=static_allowed,
        )
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt,
            "capacity_relax_cost": capacity_relax_cost,
            "edge_cover_repair_cnt": edge_cover_repair_cnt,
            "edge_cover_repair_cost": edge_cover_repair_cost,
            "edge_cover_unmet": edge_cover_unmet,
            "raw_deploy_mask": raw_deploy_mask,
            "actor_debug": {
                "service_embedding": h_s.detach().cpu(),
                "device_embedding": h_p.detach().cpu(),
                "q_embedding": q_embedding.detach().cpu(),
                "k_embedding": k_embedding.detach().cpu(),
                "qk_score": qk_scores.detach().cpu(),
                "qk_feature": qk_feature.detach().cpu(),
                "candidate_cost": candidate_cost.detach().cpu(),
                "candidate_feature": candidate_features.detach().cpu(),
                "candidate_feature_names": DEPLOYMENT_CANDIDATE_FEATURE_NAMES,
                "runtime_pair_feature_names": RUNTIME_PAIR_FEATURE_NAMES,
                "service_demand_feature_names": SERVICE_DEMAND_FEATURE_NAMES,
                "device_capability_feature": _debug_tensor(phys_feats, "device_capability_feat"),
                "device_capability_feature_names": DEVICE_CAPABILITY_FEATURE_NAMES,
                "final_score": final_scores.detach().cpu(),
                "policy_prob": policy_prob_matrix.detach().cpu(),
                "static_mask": static_allowed.detach().cpu(),
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
            _candidate_cost,
            candidate_features,
            final_scores,
        ) = self._deployment_actor_terms(
            h_s,
            h_p,
            logic_edge_index,
            logic_feats,
            phys_feats,
            static_allowed,
            prev_deploy_mask=prev_deploy_mask,
        )
        eps = 1e-6

        logp_sum = torch.tensor(0.0, device=h_s.device)
        ent_terms = []
        policy_prob_matrix = torch.zeros((Ms, Np), dtype=torch.float32, device=h_s.device)
        for service_idx in topo_order:
            stochastic_allowed = static_allowed[service_idx].clone()
            stochastic_allowed[cloud_idx] = False
            if stochastic_allowed.any():
                probs_raw = torch.where(
                    stochastic_allowed,
                    torch.sigmoid(final_scores[service_idx]),
                    torch.zeros((Np,), dtype=h_p.dtype, device=h_p.device),
                )
                probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
                probs_log = torch.where(stochastic_allowed, probs_log, torch.full_like(probs_log, eps))
                dist = torch.distributions.Bernoulli(probs=probs_log)
                acts_row = deploy_mask[service_idx].float()
                logp_sum += dist.log_prob(acts_row).masked_select(stochastic_allowed).sum()
                ent_terms.append(dist.entropy().masked_select(stochastic_allowed).mean())
                policy_prob_matrix[service_idx] = probs_raw
            policy_prob_matrix[service_idx, cloud_idx] = 1.0

        ent_sum = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=h_s.device)
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        if return_policy:
            return logp_sum, ent_sum, value.squeeze(0), {
                "policy_prob": policy_prob_matrix,
                "static_allowed": static_allowed,
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

    def offline_update(
            self,
            transitions: List[dict],
            batch_size: Optional[int] = None,
            action_target: str = "executed",
            advantage_temperature: float = 1.0,
            min_advantage_weight: float = 0.05,
            max_advantage_weight: float = 20.0,
            actor_bc_coef: float = 1.0,
            value_coef: float = 0.5,
            entropy_coef: float = 0.0,
            conservative_coef: float = 0.0,
            bootstrap_current_value: bool = True,
    ):
        """
        Offline/replay update for deployment macro-transitions.

        This is an AWAC-style actor-critic step: fit the critic to one-step
        bootstrapped returns, then improve the Bernoulli deployment actor by
        advantage-weighted log-likelihood of the actually executed placement.
        """
        if not transitions:
            return None

        target_name = str(action_target or "executed").strip().lower()
        if target_name not in {"executed", "raw"}:
            raise ValueError("deployment offline action_target must be one of: executed, raw.")

        device = next(self.parameters()).device
        batch = transitions if batch_size is None else transitions[:max(1, int(batch_size))]
        values = []
        logps = []
        entropies = []
        targets = []
        rewards = []
        conservative_terms = []
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
            logp, ent, value, policy_aux = self.evaluate(
                logic_edge_index,
                logic_feats,
                phys_edge_index,
                phys_feats,
                deploy_mask,
                prev_deploy_mask=prev_deploy_mask,
                topo_order=tr.get("topo_order"),
                return_policy=True,
            )

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
            values.append(value.squeeze())
            logps.append(logp)
            entropies.append(ent)

            if conservative_coef > 0.0:
                probs = policy_aux["policy_prob"]
                allowed = policy_aux["static_allowed"].float()
                cloud_idx = self._cloud_index(probs.size(1))
                allowed[:, cloud_idx] = 0.0
                negative_mask = (1.0 - deploy_mask.float()) * allowed
                conservative_terms.append((probs * negative_mask).sum() / allowed.sum().clamp_min(1.0))

        value_t = torch.stack(values).float()
        logp_t = torch.stack(logps).float()
        ent_t = torch.stack(entropies).float()
        target_t = torch.tensor(targets, device=device, dtype=torch.float32)
        adv_t = target_t.detach() - value_t.detach()
        temp = max(1e-6, float(advantage_temperature))
        min_weight = max(0.0, float(min_advantage_weight))
        max_weight = max(min_weight, float(max_advantage_weight))
        awac_weight = torch.exp(adv_t / temp).clamp(min=min_weight, max=max_weight)

        actor_loss = -(awac_weight * logp_t).mean() * float(actor_bc_coef)
        value_loss = F.mse_loss(value_t, target_t)
        entropy = ent_t.mean()
        conservative_loss = (
            torch.stack(conservative_terms).mean()
            if conservative_terms
            else torch.tensor(0.0, device=device)
        )
        critic_loss = float(value_coef) * value_loss
        actor_objective_loss = actor_loss \
            + float(conservative_coef) * conservative_loss \
            - float(entropy_coef) * entropy

        # Keep the deployment actor update behavior-cloning/AWAC driven.
        # The critic is fit in a separate step so value gradients do not move
        # the Bernoulli deployment policy through shared candidate features.
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
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
            "policy_loss": _scalar_to_float(actor_loss.detach()),
            "value_loss": _scalar_to_float(value_loss.detach()),
            "entropy": _scalar_to_float(entropy.detach()),
            "entropy_coef": float(entropy_coef),
            "value_coef": float(value_coef),
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "ratio_mean": _scalar_to_float(awac_weight.detach().mean()),
            "ratio_std": _tensor_std_float(awac_weight.detach()),
            "actor_grad_norm": _scalar_to_float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "conservative_loss": _scalar_to_float(conservative_loss.detach()),
        }


class HedgerOffloadingPPO(nn.Module):
    def __init__(self, encoder: TopologyEncoders, d_model=64,
                 actor_lr=3e-4, critic_lr=1e-3, update_encoder: bool = True,
                 gamma=0.99, lamda=0.95, clip_eps=0.2,
                 cloud_node_idx: int = -1,
                 unknown_exploration_prob: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.actor = OffloadActor(d_model)
        hidden_dim = max(32, d_model)
        self.offloading_cost_head = CandidateCostHead(
            input_dim=len(OFFLOADING_CANDIDATE_FEATURE_NAMES),
            hidden_dim=hidden_dim,
        )
        self.critic = CandidateValueHead(input_dim=len(OFFLOADING_CANDIDATE_FEATURE_NAMES), d_model=d_model)

        encoder_params = list(self.encoder.parameters()) if update_encoder else []
        params_actor = (
            list(self.actor.parameters())
            + list(self.offloading_cost_head.parameters())
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
        queue_short = runtime_row[..., 0]
        queue_busy = runtime_row[..., 1]
        real_time_per_complexity = runtime_row[..., 2]
        confidence = runtime_row[..., 3]
        freshness = runtime_row[..., 4]
        load_score = queue_short + demand_feat[service_idx, 2] * real_time_per_complexity
        return torch.stack(
            [
                qk_feature[service_idx],
                compute_gap,
                queue_short,
                queue_busy,
                real_time_per_complexity,
                load_score,
                confidence,
                freshness,
                dependency_context[..., 0],
                dependency_context[..., 1],
                is_cloud,
            ],
            dim=-1,
        )

    def _offloading_base_candidate_features(
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
        candidate_features = self._offloading_base_candidate_features(
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

    def _mix_unknown_exploration_probs(
            self,
            base_probs: torch.Tensor,
            candidate_row: torch.Tensor,
            allowed: torch.Tensor,
            *,
            enable_unknown_exploration: bool,
            deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unknown_probs, unknown_weight = self._unknown_exploration_probs(candidate_row, allowed)
        eps = self.unknown_exploration_prob if enable_unknown_exploration and not deterministic else 0.0
        if eps <= 0.0 or float(unknown_probs.sum().detach().item()) <= 1e-8:
            return base_probs, unknown_probs, unknown_weight
        mixed = (1.0 - eps) * base_probs + eps * unknown_probs.to(device=base_probs.device, dtype=base_probs.dtype)
        mixed = mixed.masked_fill(~allowed.bool(), 0.0)
        mixed_sum = mixed.sum()
        if float(mixed_sum.detach().item()) <= 1e-8:
            return base_probs, unknown_probs, unknown_weight
        return mixed / mixed_sum.clamp_min(1e-8), unknown_probs, unknown_weight

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
               enable_unknown_exploration: bool = False):
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
        final_scores = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        candidate_costs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        candidate_features_matrix = torch.zeros(
            (Ms, Np, len(OFFLOADING_CANDIDATE_FEATURE_NAMES)),
            dtype=h_s.dtype,
            device=h_p.device,
        )

        proposal_action_list = [-1 for _ in range(Ms)]
        logp_sum = torch.tensor(0.0, device=h_p.device)
        ent_sum = torch.tensor(0.0, device=h_p.device)

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
            candidate_cost = self.offloading_cost_head(candidate_row.float())
            candidate_scores = -candidate_cost
            candidate_features_matrix[i] = candidate_row
            candidate_costs[i] = candidate_cost
            final_scores[i] = candidate_scores
            effective_mask[i] = allowed
            masked_scores = candidate_scores.masked_fill(~allowed, float('-inf'))
            base_probs_i = F.softmax(masked_scores, dim=-1)
            probs_i, unknown_probs_i, unknown_weight_i = self._mix_unknown_exploration_probs(
                base_probs_i,
                candidate_row,
                allowed,
                enable_unknown_exploration=enable_unknown_exploration,
                deterministic=deterministic,
            )
            base_policy_prob_matrix[i] = base_probs_i
            unknown_policy_prob_matrix[i] = unknown_probs_i
            unknown_exploration_weight_matrix[i] = unknown_weight_i
            policy_prob_matrix[i] = probs_i
            dist = torch.distributions.Categorical(probs=probs_i)
            sampled_action = torch.argmax(probs_i) if deterministic else dist.sample()
            proposal_action_list[i] = int(sampled_action.item())
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

        value_candidates = self._offloading_base_candidate_features(
            logic_edge_index, logic_feats, phys_feats, qk_feature, runtime_feat
        )
        value = self.critic(h_s, h_p, value_candidates, effective_mask)
        return projected_actions, logp_sum, ent_sum, value.squeeze(0), {
            "proposal_actions": proposal_actions.detach(),
            "projected_actions": projected_actions.detach(),
            **projection_info,
            "actor_debug": {
                "service_embedding": h_s.detach().cpu(),
                "device_embedding": h_p.detach().cpu(),
                "q_embedding": q_embedding.detach().cpu(),
                "k_embedding": k_embedding.detach().cpu(),
                "qk_score": qk_scores.detach().cpu(),
                "qk_feature": qk_feature.detach().cpu(),
                "candidate_cost": candidate_costs.detach().cpu(),
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
                    fill_value=float(self.unknown_exploration_prob if enable_unknown_exploration and not deterministic else 0.0),
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
                 enable_unknown_exploration: bool = False):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        effective_mask = self._effective_offloading_mask(static_mask)
        proposal_action_list = [int(action) for action in proposal_actions.detach().cpu().long().tolist()]

        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)

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
            scores_i = -self.offloading_cost_head(candidate_row.float())
            base_probs_i = F.softmax(scores_i.masked_fill(~allowed, float('-inf')), dim=-1)
            probs_i, _, _ = self._mix_unknown_exploration_probs(
                base_probs_i,
                candidate_row,
                allowed,
                enable_unknown_exploration=enable_unknown_exploration,
                deterministic=False,
            )
            dist = torch.distributions.Categorical(probs=probs_i)
            a = torch.tensor(proposal_action_list[i], dtype=torch.long, device=h_p.device)
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()

        value_candidates = self._offloading_base_candidate_features(
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
                "unknown_exploration_enabled": bool(tr.get("unknown_exploration_enabled", False)),
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
                                                    enable_unknown_exploration=tr["unknown_exploration_enabled"])
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
