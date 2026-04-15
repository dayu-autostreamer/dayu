from typing import Optional, List, Dict, Tuple

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

    def _dynamic_allowed_mask(
            self,
            static_allowed_row: torch.Tensor,
            residual: torch.Tensor,
            model_mem: torch.Tensor,
            cloud_idx: int,
    ) -> torch.Tensor:
        allowed = torch.logical_and(static_allowed_row, residual >= (model_mem - 1e-6))
        allowed = allowed.clone()
        allowed[cloud_idx] = True
        return allowed

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index,
               phys_feats, topo_order: Optional[list] = None,
               prev_deploy_mask: Optional[torch.Tensor] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)

        # Current deployment budget after accounting for baseline usage and the
        # memory that can be reclaimed from the previous deployment.
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask)  # [Np]
        model_mem = logic_feats["model_mem"].float()  # [Ms]
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)

        # Static feasibility mask.
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]

        # Sequentially mask out devices whose remaining memory budget can no
        # longer host the current service, so PPO trains on the actual feasible
        # action distribution instead of a post-hoc projection.
        deploy_mask = torch.zeros((Ms, Np), dtype=torch.bool, device=h_s.device)
        logp_sum = torch.tensor(0.0, device=h_s.device)
        ent_terms = []
        capacity_relax_cnt = 0
        eps = 1e-6
        residual = residual.clone()
        for service_idx in topo_order:
            allowed_row = self._dynamic_allowed_mask(
                static_allowed[service_idx],
                residual,
                model_mem[service_idx],
                cloud_idx,
            )
            stochastic_allowed = allowed_row.clone()
            stochastic_allowed[cloud_idx] = False
            acts_row = torch.zeros(Np, dtype=torch.bool, device=h_s.device)
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
                sampled_row = torch.rand_like(probs_sample) < probs_sample
                acts_row = torch.where(
                    stochastic_allowed,
                    sampled_row,
                    torch.zeros_like(sampled_row, dtype=torch.bool),
                )
                logp_sum += dist.log_prob(acts_row.float()).sum()
                ent_terms.append(dist.entropy()[stochastic_allowed].mean())
            acts_row[cloud_idx] = True
            deploy_mask[service_idx] = acts_row

            edge_replicas = acts_row.clone()
            edge_replicas[cloud_idx] = False
            if edge_replicas.any():
                residual[edge_replicas] = torch.clamp(
                    residual[edge_replicas] - model_mem[service_idx],
                    min=0.0,
                )

        ent_sum = torch.stack(ent_terms).mean() if ent_terms else torch.tensor(0.0, device=h_s.device)
        value = self.critic(h_s, h_p, self._deployment_context(h_p, prev_deploy_mask))

        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt
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

        The policy distribution matches the sequential feasible sampler used in
        `policy()`: service replicas are sampled row by row under the current
        residual memory budget.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)

        # Static feasibility mask.
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask).clone()
        model_mem = logic_feats["model_mem"].float()
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)
        deploy_mask = self._enforce_cloud_replica(deploy_mask.bool())
        eps = 1e-6

        logp_sum = torch.tensor(0.0, device=h_s.device)
        ent_terms = []
        for service_idx in topo_order:
            allowed_row = self._dynamic_allowed_mask(
                static_allowed[service_idx],
                residual,
                model_mem[service_idx],
                cloud_idx,
            )
            stochastic_allowed = allowed_row.clone()
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

            edge_replicas = deploy_mask[service_idx].clone()
            edge_replicas[cloud_idx] = False
            if edge_replicas.any():
                residual[edge_replicas] = torch.clamp(
                    residual[edge_replicas] - model_mem[service_idx],
                    min=0.0,
                )

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
        adv = torch.tensor(adv, device=device)
        rets = torch.tensor(rets, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
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
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._actor_train_params, 1.0)
                self.actor_opt.step()
                self.critic_opt.step()


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

    def _dynamic_offloading_mask(
            self,
            base_mask: torch.Tensor,
            parent_indices: List[int],
            actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the row-wise feasible mask for one service.

        Once any parent is assigned to the cloud, the current service is
        restricted to the cloud before sampling, so PPO trains on the real
        constrained distribution instead of repairing invalid actions afterward.
        """
        if parent_indices and any(int(actions[parent].item()) == self.cloud_idx for parent in parent_indices):
            allowed = torch.zeros_like(base_mask, dtype=torch.bool)
            allowed[self.cloud_idx] = True
            return allowed
        return self._normalize_offloading_mask(base_mask.clone())

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

        actions = torch.empty(Ms, dtype=torch.long, device=h_p.device)
        logp_sum = torch.tensor(0.0, device=h_p.device)
        ent_sum = torch.tensor(0.0, device=h_p.device)

        for i in topo_order:
            allowed = self._dynamic_offloading_mask(static_mask[i], parents[i], actions)
            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs=probs_i)
            actions[i] = dist.sample()
            logp_sum += dist.log_prob(actions[i])
            ent_sum += dist.entropy()

        correction_cnt = 0
        switches = self._count_switches(actions, parents)
        value = self.critic(h_s, h_p, self._offloading_context(h_p, static_mask))
        aux_cost = self.cfg.penalty_switch * switches + self.cfg.penalty_relax * correction_cnt
        return actions, logp_sum, ent_sum, value.squeeze(0), {
            "switches": switches,
            "correction_cnt": correction_cnt,
            "aux_cost": aux_cost,
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
        correction_cnt = 0

        for i in topo_order:
            allowed = self._dynamic_offloading_mask(static_mask[i], parents[i], actions)
            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs=probs_i)
            a = actions[i]
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()

        switches = self._count_switches(actions, parents)
        value = self.critic(h_s, h_p, self._offloading_context(h_p, static_mask))
        aux_cost = self.cfg.penalty_switch * switches + self.cfg.penalty_relax * correction_cnt
        return logp_sum, ent_sum, value.squeeze(0), {
            "switches": switches,
            "correction_cnt": correction_cnt,
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
        adv = torch.tensor(adv, device=device)
        rets = torch.tensor(rets, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
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

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._actor_train_params, 1.0)
                self.actor_opt.step()
                self.critic_opt.step()
