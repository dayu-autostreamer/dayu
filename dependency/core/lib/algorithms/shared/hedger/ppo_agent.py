from typing import Optional, List, Dict, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_encoder import TopologyEncoders
from .ppo_network import DeploymentActor, OffloadActor, ValueHead, FeatureAdapter
from .hedger_config import DeploymentConstraintCfg, OffloadingConstraintCfg
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


def _build_mask_context(mask: Optional[torch.Tensor], h_p: torch.Tensor) -> torch.Tensor:
    """
    Summarize a service-device mask into one context vector.

    The mask is projected through physical node embeddings so the critic can
    observe deployment availability / current placement without depending on a
    fixed graph size.
    """
    if mask is None:
        return torch.zeros((1, h_p.size(-1)), device=h_p.device, dtype=h_p.dtype)

    mask_f = mask.float()
    if mask_f.numel() == 0:
        return torch.zeros((1, h_p.size(-1)), device=h_p.device, dtype=h_p.dtype)

    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    per_service_context = torch.matmul(mask_f, h_p) / counts
    return per_service_context.mean(dim=0, keepdim=True)


class HedgerDeploymentPPO(nn.Module):
    def __init__(self, encoder: TopologyEncoders, d_model=64, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lamda=0.95, clip_eps=0.2, update_encoder: bool = True, cloud_node_idx: int = -1,
                 constraint_cfg: DeploymentConstraintCfg = DeploymentConstraintCfg()):
        super().__init__()
        self.encoder = encoder
        self.actor = DeploymentActor(d_model)
        self.critic = ValueHead(d_model)
        self.service_adapter = FeatureAdapter(d_model)
        self.device_adapter = FeatureAdapter(d_model)

        adapter_params = list(self.service_adapter.parameters()) + list(self.device_adapter.parameters())
        encoder_params = list(self.encoder.parameters()) if update_encoder else []
        params_actor = list(self.actor.parameters()) + adapter_params + encoder_params
        self.actor_opt = torch.optim.Adam(params_actor, lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self._actor_train_params = params_actor

        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.cloud_idx = cloud_node_idx
        self.cfg = constraint_cfg

    def _static_allowed_mask(
        self,
        phys_feats: Dict[str, torch.Tensor],
        logic_feats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build the static deployment mask.

        This mask filters out physically impossible service-device pairs using
        only model memory footprint and total device memory.

        Returns:
            `static_allowed`: boolean tensor of shape `[Ms, Np]`, where
            `static_allowed[i, n]` indicates whether service `i` is statically
            deployable on device `n`.
        """
        cap = phys_feats["mem_capacity"].float()  # [Np]
        model_mem = logic_feats["model_mem"].float()  # [Ms]
        Ms = model_mem.size(0)
        Np = cap.size(0)

        # Shape: [Ms, Np].
        static_allowed = model_mem.view(Ms, 1) <= cap.view(1, Np)

        # Always keep the cloud as a feasible fallback.
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)
        static_allowed[:, cloud_idx] = True

        return static_allowed

    @staticmethod
    def topo_order(edge_index: torch.Tensor, num_nodes: int):
        row, col = edge_index
        indeg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        indeg.scatter_add_(0, col, torch.ones_like(col))
        q = [i for i in range(num_nodes) if indeg[i] == 0]
        order = []
        adj = [[] for _ in range(num_nodes)]
        for u, v in zip(row.tolist(), col.tolist()):
            adj[u].append(v)
        while q:
            u = q.pop(0)
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
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
            Compute the memory budget available to the current deployment step.

            - If `logic_feats` or `prev_deploy_mask` is missing, fall back to:
              `residual = mem_capacity * (1 - mem_util).`

            - If `prev_deploy_mask` is provided, assume the previous deployment
              is released before placing the new one:

              `total_used = cap * util`
              `prev_usage = sum_i prev_deploy_mask[i, n] * model_mem[i]`

              `baseline = max(total_used - prev_usage, 0)`  # memory used by non-RL workloads
              `residual_0 = max(cap - baseline, 0)`  # budget for the new deployment
        """
        cap = phys_feats["mem_capacity"].float()  # [Np] GB

        if logic_feats is None or prev_deploy_mask is None:
            util = phys_feats["mem_util_seq"][:, -1].float()  # Current total memory utilization.
            return cap * (1.0 - util)

        util = phys_feats["mem_util_seq"][:, -1].float()  # [Np]
        total_used = cap * util  # [Np], current total memory usage.

        model_mem = logic_feats["model_mem"].float()  # [Ms]
        # `prev_deploy_mask`: [Ms, Np] -> [Np], total memory used by the
        # previous RL deployment on each node.
        prev_usage = torch.matmul(prev_deploy_mask.float().t(), model_mem)  # [Np]
        baseline = torch.clamp(total_used - prev_usage, min=0.0)  # [Np]

        # Budget available to the new deployment round.
        residual = torch.clamp(cap - baseline, min=0.0)  # [Np]
        return residual

    def _adapt_embeddings(self, h_s: torch.Tensor, h_p: torch.Tensor):
        """Apply task-specific residual adapters to the shared encoder outputs."""
        h_s = self.service_adapter(h_s)
        h_p = self.device_adapter(h_p)
        return h_s, h_p

    def _deployment_context(self, h_p: torch.Tensor, prev_deploy_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if prev_deploy_mask is None:
            return torch.zeros((1, h_p.size(-1)), device=h_p.device, dtype=h_p.dtype)
        return _build_mask_context(self._enforce_cloud_replica(prev_deploy_mask.bool()), h_p)

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
        value = self.critic(h_s, h_p, self._deployment_context(h_p, prev_deploy_mask))
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
    ) -> Tuple[torch.Tensor, int, float]:
        """
        Project a sampled deployment mask back into memory-feasible space.

        The projection is deterministic given the sampled deployment and the
        actor probabilities for sampled replicas:

        - For each edge device, consider only the services whose raw Bernoulli
          sample selected that device.
        - Choose a subset that fits within the device residual-memory budget.
        - The primary objective is to maximize the retained policy preference,
          measured as the sum of Bernoulli log-odds `log(p / (1 - p))` of the
          kept sampled replicas.

        For small candidate sets, the best subset is found exactly. For larger
        candidate sets, a probability-priority greedy fallback is used. Previous
        deployment state and memory footprint are only used as tie-breakers.

        Besides the removed replica count, the projector also reports a
        correction cost. Each removed sampled replica contributes
        `-log(1 - p_ij)`, i.e. the negative log-probability of the corrected
        zero action under the raw Bernoulli policy. The final cost is normalized
        by the number of logical services so it can be used directly in reward
        shaping across DAGs of different sizes.
        """
        corrected = self._enforce_cloud_replica(raw_deploy_mask.bool())
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask)
        model_mem = logic_feats["model_mem"].float()
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (corrected.size(1) - 1)
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
            if total_selected_mem <= float(residual[device_idx].item()) + 1e-6:
                continue

            keep = self._select_device_subset(
                selected=selected,
                capacity=float(residual[device_idx].item()),
                model_mem=model_mem,
                raw_probs=raw_probs[:, device_idx],
                prev_selected=prev_edge_mask[:, device_idx],
            )

            for service_idx in selected:
                if service_idx not in keep:
                    corrected[service_idx, device_idx] = False
                    capacity_relax_cnt += 1
                    prob = float(raw_probs[service_idx, device_idx].item())
                    prob = min(max(prob, 1e-6), 1.0 - 1e-6)
                    capacity_relax_cost += -math.log(max(1.0 - prob, 1e-6))

        capacity_relax_cost /= float(num_services)
        return corrected, capacity_relax_cnt, capacity_relax_cost

    @staticmethod
    def _projection_priority_key(
            service_idx: int,
            score: float,
            prev_selected: bool,
            mem: float,
    ) -> tuple:
        return (
            score,
            1 if prev_selected else 0,
            -mem,
            -int(service_idx),
        )

    def _select_device_subset(
            self,
            selected: List[int],
            capacity: float,
            model_mem: torch.Tensor,
            raw_probs: torch.Tensor,
            prev_selected: torch.Tensor,
    ) -> set:
        """
        Select which sampled replicas to keep on one edge device.

        The objective is to preserve as much actor preference as possible under
        the memory budget. For current Hedger workloads, service counts are
        small, so exact search is practical and removes most projection bias.
        """
        if capacity <= 1e-6 or not selected:
            return set()

        candidates = []
        for service_idx in selected:
            prob = float(raw_probs[service_idx].item())
            prob = min(max(prob, 1e-6), 1.0 - 1e-6)
            score = float(torch.logit(torch.tensor(prob, dtype=torch.float32)).item())
            mem = float(model_mem[service_idx].item())
            was_selected = bool(prev_selected[service_idx].item())
            candidates.append((int(service_idx), score, mem, was_selected))

        def better(candidate_tuple, best_tuple):
            if best_tuple is None:
                return True
            cand_score, cand_count, cand_prev, cand_mem, cand_ids = candidate_tuple
            best_score, best_count, best_prev, best_mem, best_ids = best_tuple
            if cand_score > best_score + 1e-9:
                return True
            if cand_score < best_score - 1e-9:
                return False
            if cand_count > best_count:
                return True
            if cand_count < best_count:
                return False
            if cand_prev > best_prev:
                return True
            if cand_prev < best_prev:
                return False
            if cand_mem < best_mem - 1e-9:
                return True
            if cand_mem > best_mem + 1e-9:
                return False
            return cand_ids < best_ids

        # Exact subset search is affordable for the current small service DAGs.
        if len(candidates) <= 18:
            best_tuple = None
            best_keep = set()
            total_masks = 1 << len(candidates)
            for mask_bits in range(total_masks):
                used_mem = 0.0
                score_sum = 0.0
                prev_count = 0
                keep_ids = []
                feasible = True
                for bit_idx, (service_idx, score, mem, was_selected) in enumerate(candidates):
                    if not (mask_bits & (1 << bit_idx)):
                        continue
                    used_mem += mem
                    if used_mem > capacity + 1e-6:
                        feasible = False
                        break
                    score_sum += score
                    prev_count += 1 if was_selected else 0
                    keep_ids.append(service_idx)
                if not feasible:
                    continue
                keep_tuple = (score_sum, len(keep_ids), prev_count, used_mem, tuple(keep_ids))
                if better(keep_tuple, best_tuple):
                    best_tuple = keep_tuple
                    best_keep = set(keep_ids)
            return best_keep

        # Greedy fallback for unusually large service sets.
        remaining = capacity
        keep = set()
        ranked = sorted(
            candidates,
            key=lambda item: self._projection_priority_key(item[0], item[1], item[3], item[2]),
            reverse=True,
        )
        for service_idx, score, mem, was_selected in ranked:
            if mem <= remaining + 1e-6:
                keep.add(service_idx)
                remaining -= mem
        return keep

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index,
               phys_feats, topo_order: Optional[list] = None,
               prev_deploy_mask: Optional[torch.Tensor] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)

        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)

        # Static feasibility mask.
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]

        # Sample first from the static deployment distribution, then apply a
        # deterministic capacity projection. PPO is optimized on the raw sampled
        # action, while the environment executes the corrected one.
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
                probs_raw = self.actor(
                    h_s[service_idx:service_idx + 1],
                    h_p,
                    mask=stochastic_allowed.unsqueeze(0),
                )[0]
                probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
                probs_log = torch.where(stochastic_allowed, probs_log, torch.full_like(probs_log, eps))
                probs_sample = torch.where(stochastic_allowed, probs_raw, torch.zeros_like(probs_raw))
                dist = torch.distributions.Bernoulli(probs=probs_log)
                sampled_bits = torch.rand_like(probs_sample) < probs_sample
                sampled_row = torch.where(
                    stochastic_allowed,
                    sampled_bits,
                    torch.zeros_like(sampled_bits, dtype=torch.bool),
                )
                logp_sum += dist.log_prob(sampled_row.float()).sum()
                ent_terms.append(dist.entropy()[stochastic_allowed].mean())
                raw_probs[service_idx] = probs_raw
            sampled_row[cloud_idx] = True
            raw_deploy_mask[service_idx] = sampled_row
            raw_probs[service_idx, cloud_idx] = 1.0

        ent_sum = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=h_s.device)
        deploy_mask, capacity_relax_cnt, capacity_relax_cost = self._project_deployment_mask(
            raw_deploy_mask,
            raw_probs=raw_probs,
            logic_feats=logic_feats,
            phys_feats=phys_feats,
            prev_deploy_mask=prev_deploy_mask,
        )
        value = self.critic(h_s, h_p, self._deployment_context(h_p, prev_deploy_mask))

        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt,
            "capacity_relax_cost": capacity_relax_cost,
            "raw_deploy_mask": raw_deploy_mask,
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
    ):
        """
        Evaluate `deploy_mask` under the current parameters.

        PPO is evaluated against the raw pre-correction deployment sample, while
        the environment executes the corrected mask returned by `policy()`.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)

        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)
        deploy_mask = self._enforce_cloud_replica(deploy_mask.bool())
        eps = 1e-6

        logp_sum = torch.tensor(0.0, device=h_s.device)
        ent_terms = []
        for service_idx in topo_order:
            stochastic_allowed = static_allowed[service_idx].clone()
            stochastic_allowed[cloud_idx] = False
            if stochastic_allowed.any():
                probs_raw = self.actor(
                    h_s[service_idx:service_idx + 1],
                    h_p,
                    mask=stochastic_allowed.unsqueeze(0),
                )[0]
                probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
                probs_log = torch.where(stochastic_allowed, probs_log, torch.full_like(probs_log, eps))
                dist = torch.distributions.Bernoulli(probs=probs_log)
                acts_row = deploy_mask[service_idx].float()
                logp_sum += dist.log_prob(acts_row).masked_select(stochastic_allowed).sum()
                ent_terms.append(dist.entropy().masked_select(stochastic_allowed).mean())

        ent_sum = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=h_s.device)
        value = self.critic(h_s, h_p, self._deployment_context(h_p, prev_deploy_mask))
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
                "deploy_mask": tr.get("raw_deploy_mask", tr["deploy_mask"]).to(device),
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


class HedgerOffloadingPPO(nn.Module):
    def __init__(self, encoder: TopologyEncoders, d_model=64,
                 actor_lr=3e-4, critic_lr=1e-3, update_encoder: bool = True,
                 gamma=0.99, lamda=0.95, clip_eps=0.2,
                 source_node_idx: int = 0, cloud_node_idx: int = -1,
                 constraint_cfg: OffloadingConstraintCfg = OffloadingConstraintCfg()):
        super().__init__()
        self.encoder = encoder
        self.actor = OffloadActor(d_model)
        self.critic = ValueHead(d_model)
        self.service_adapter = FeatureAdapter(d_model)
        self.device_adapter = FeatureAdapter(d_model)

        adapter_params = list(self.service_adapter.parameters()) + list(self.device_adapter.parameters())
        encoder_params = list(self.encoder.parameters()) if update_encoder else []
        params_actor = list(self.actor.parameters()) + adapter_params + encoder_params
        self.actor_opt = torch.optim.Adam(params_actor, lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self._actor_train_params = params_actor

        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
        self.source = source_node_idx
        self.cloud_idx = cloud_node_idx
        self.cfg = constraint_cfg

    @staticmethod
    def topo_order(edge_index: torch.Tensor, num_nodes: int):
        row, col = edge_index
        indeg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        indeg.scatter_add_(0, col, torch.ones_like(col))
        q = [i for i in range(num_nodes) if indeg[i] == 0]
        order = []
        adj = [[] for _ in range(num_nodes)]
        for u, v in zip(row.tolist(), col.tolist()):
            adj[u].append(v)
        while q:
            u = q.pop(0)
            order.append(u)
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if len(order) < num_nodes:
            seen = set(order)
            order += [i for i in range(num_nodes) if i not in seen]
        return order

    def _adapt_embeddings(self, h_s: torch.Tensor, h_p: torch.Tensor):
        """Apply task-specific residual adapters to the shared encoder outputs."""
        h_s = self.service_adapter(h_s)
        h_p = self.device_adapter(h_p)
        return h_s, h_p

    def _offloading_context(self, h_p: torch.Tensor, static_mask: torch.Tensor) -> torch.Tensor:
        return _build_mask_context(static_mask.bool(), h_p)

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
        value = self.critic(h_s, h_p, self._offloading_context(h_p, static_mask))
        return value.squeeze(0)

    @staticmethod
    def _build_parents(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        row, col = edge_index
        parents = [[] for _ in range(num_nodes)]
        for u, v in zip(row.tolist(), col.tolist()):
            parents[v].append(u)
        return parents

    def _normalize_offloading_mask(self, base: torch.Tensor) -> torch.Tensor:
        """
        Normalize a per-service offloading mask.

        Assume that deployment always leaves at least one feasible replica for
        every service and that the cloud is a fallback. This extra guard prevents
        an all-False row from invalidating categorical sampling.
        """
        if base.any():
            return base

        allowed = torch.zeros_like(base)
        allowed[self.cloud_idx] = True
        return allowed

    def _correct_offloading_actions(
            self,
            raw_actions: torch.Tensor,
            parents: List[List[int]],
            topo_order: List[int],
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply the cloud-cascade rule after sampling.

        If any parent of a service is assigned to the cloud, the current service
        is corrected to the cloud as well. The correction is deterministic given
        the sampled actions and the DAG structure, so PPO can optimize the raw
        policy while the environment executes the corrected plan.
        """
        corrected = raw_actions.clone()
        correction_cnt = 0
        for node_idx in topo_order:
            parent_indices = parents[node_idx]
            if not parent_indices:
                continue
            if any(int(corrected[parent].item()) == self.cloud_idx for parent in parent_indices):
                if int(corrected[node_idx].item()) != self.cloud_idx:
                    corrected[node_idx] = self.cloud_idx
                    correction_cnt += 1
        return corrected, correction_cnt

    def _offloading_correction_cost(
            self,
            raw_probs: torch.Tensor,
            raw_actions: torch.Tensor,
            corrected_actions: torch.Tensor,
    ) -> float:
        """
        Measure how incompatible the executed corrected plan is with the raw policy.

        Each corrected service contributes the negative log-probability of the
        executed action under the raw categorical policy. The total cost is
        normalized by the number of services so reward scale stays comparable
        across DAG sizes.
        """
        num_services = max(1, int(raw_actions.numel()))
        correction_cost = 0.0
        for service_idx in range(num_services):
            corrected = int(corrected_actions[service_idx].item())
            raw = int(raw_actions[service_idx].item())
            if corrected == raw:
                continue
            prob = float(raw_probs[service_idx, corrected].item())
            prob = min(max(prob, 1e-6), 1.0)
            correction_cost += -math.log(prob)
        return correction_cost / float(num_services)

    def _count_switches(self, actions: torch.Tensor, parents: List[List[int]]) -> int:
        """
        Count execution-target changes along actual DAG edges.

        Root services are compared against the source node. Non-root services are
        compared against each parent on incoming logical edges.
        """
        switches = 0
        for node, parent_indices in enumerate(parents):
            cur = actions[node].item()
            if not parent_indices:
                if cur != self.source:
                    switches += 1
                continue
            for parent in parent_indices:
                if cur != actions[parent].item():
                    switches += 1
        return switches

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
               static_mask: torch.Tensor, topo_order: Optional[list] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)

        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)
        parents = self._build_parents(logic_edge_index, Ms)

        raw_actions = torch.empty(Ms, dtype=torch.long, device=h_p.device)
        raw_probs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        logp_sum = torch.tensor(0.0, device=h_p.device)
        ent_sum = torch.tensor(0.0, device=h_p.device)

        for i in topo_order:
            allowed = self._normalize_offloading_mask(static_mask[i].clone())
            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            raw_probs[i] = probs_i
            dist = torch.distributions.Categorical(probs=probs_i)
            raw_actions[i] = dist.sample()
            logp_sum += dist.log_prob(raw_actions[i])
            ent_sum += dist.entropy()

        actions, correction_cnt = self._correct_offloading_actions(raw_actions, parents, topo_order)
        correction_cost = self._offloading_correction_cost(raw_probs, raw_actions, actions)
        switches = self._count_switches(actions, parents)
        value = self.critic(h_s, h_p, self._offloading_context(h_p, static_mask))
        aux_cost = self.cfg.penalty_switch * switches + self.cfg.penalty_relax * correction_cost
        return actions, logp_sum, ent_sum, value.squeeze(0), {
            "switches": switches,
            "correction_cnt": correction_cnt,
            "correction_cost": correction_cost,
            "aux_cost": aux_cost,
            "raw_actions": raw_actions,
        }

    def evaluate(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
                 actions: torch.Tensor, static_mask: torch.Tensor, topo_order: Optional[list] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)

        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)

        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)
        parents = self._build_parents(logic_edge_index, Ms)
        corrected_actions, correction_cnt = self._correct_offloading_actions(actions, parents, topo_order)
        raw_probs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)

        for i in topo_order:
            allowed = self._normalize_offloading_mask(static_mask[i].clone())
            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            raw_probs[i] = probs_i
            dist = torch.distributions.Categorical(probs=probs_i)
            a = actions[i]
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()

        correction_cost = self._offloading_correction_cost(raw_probs, actions, corrected_actions)
        switches = self._count_switches(corrected_actions, parents)
        value = self.critic(h_s, h_p, self._offloading_context(h_p, static_mask))
        aux_cost = self.cfg.penalty_switch * switches + self.cfg.penalty_relax * correction_cost
        return logp_sum, ent_sum, value.squeeze(0), {
            "switches": switches,
            "correction_cnt": correction_cnt,
            "correction_cost": correction_cost,
            "aux_cost": aux_cost,
        }

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
                "actions": tr["actions"].to(device),
                "static_mask": tr["static_mask"].to(device),
                "topo_order": tr["topo_order"],
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
                                                    tr['actions'], tr['static_mask'], tr['topo_order'])
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
