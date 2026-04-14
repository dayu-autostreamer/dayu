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
        cap = phys_feats["mem_capacity"].float()  # [Np] MB

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

    def _enforce_cloud_replica(self, deploy_mask: torch.Tensor) -> torch.Tensor:
        """Force every service to keep a cloud replica in the deployment mask."""
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (deploy_mask.size(1) - 1)
        deploy_mask = deploy_mask.clone()
        deploy_mask[:, cloud_idx] = True
        return deploy_mask

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

        # Mask out physically impossible placements before sampling.
        probs_raw = self.actor(h_s, h_p, mask=static_allowed)  # [Ms, Np], post-sigmoid replica probabilities

        # Clamp probabilities for numerical stability and avoid `log(0)` on masked-out positions.
        eps = 1e-6
        probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
        probs_log = torch.where(static_allowed, probs_log, torch.full_like(probs_log, eps))

        # Do not sample from statically invalid positions.
        probs_sample = torch.where(static_allowed, probs_raw, torch.zeros_like(probs_raw))

        # Sample a multilabel Bernoulli action. `probs_log` is used to build the distribution;
        # `probs_sample` is used for actual sampling.
        dist = torch.distributions.Bernoulli(probs=probs_log)
        acts = torch.rand_like(probs_sample) < probs_sample  # bool [Ms, Np]
        deploy_mask = self._enforce_cloud_replica(acts)

        # Project the sampled deployment back into the residual capacity budget.
        capacity_relax_cnt = 0
        for n in range(Np):
            if n == cloud_idx:
                # Cloud replicas are always retained and are not pruned here.
                continue

            while True:
                # Current total memory assigned to node `n`.
                used_n = (deploy_mask[:, n].float() * model_mem).sum()

                # Stop once the residual budget is satisfied.
                if used_n <= residual[n] + 1e-6:
                    break

                # All services currently placed on node `n`.
                candidates = torch.nonzero(deploy_mask[:, n], as_tuple=False).view(-1)
                if candidates.numel() == 0:
                    break

                # Remove the lowest-confidence replica until the placement is feasible.
                cand_probs = probs_log[candidates, n]
                drop_local_idx = torch.argmin(cand_probs)
                drop_service_idx = candidates[drop_local_idx]

                deploy_mask[drop_service_idx, n] = False
                capacity_relax_cnt += 1

        # Compute log-probabilities and entropy under the static Bernoulli distribution.
        act_float = deploy_mask.float()
        logp = dist.log_prob(act_float)  # [Ms, Np]
        ent = dist.entropy()  # [Ms, Np]

        logp_sum = logp.sum()
        ent_sum = ent.mean()

        value = self.critic(h_s, h_p)

        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt
        }

    def evaluate(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats, deploy_mask: torch.Tensor):
        """
        Evaluate `deploy_mask` under the current parameters.

        The policy distribution remains "static feasibility mask + independent
        Bernoulli", matching the sampling stage. Capacity correction is not
        replayed here and is instead treated as a deterministic environment-side
        projection.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)

        # Static feasibility mask.
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]

        probs_raw = self.actor(h_s, h_p, mask=static_allowed)
        eps = 1e-6
        probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
        probs_log = torch.where(static_allowed, probs_log, torch.full_like(probs_log, eps))

        dist = torch.distributions.Bernoulli(probs=probs_log)

        deploy_mask = self._enforce_cloud_replica(deploy_mask)
        act_float = deploy_mask.float()
        logp = dist.log_prob(act_float)  # [Ms, Np]
        ent = dist.entropy()  # [Ms, Np]

        logp_sum = logp.sum()
        ent_sum = ent.mean()

        value = self.critic(h_s, h_p)
        return logp_sum, ent_sum, value.squeeze(0)

    def ppo_update(self, transitions: List[dict], epochs=4, batch_size=16, clip_eps=None, entropy_coef=0.01,
                   value_coef=0.5):
        clip_eps = self.clip_eps if clip_eps is None else clip_eps
        device = next(self.parameters()).device
        old_logp = torch.stack([t['logp'].detach().to(device) for t in transitions])
        old_val = [_scalar_to_float(t['value']) for t in transitions]
        rewards = [float(t['reward']) for t in transitions]
        dones = [t['done'] for t in transitions]
        adv, rets = compute_returns_advantages(rewards, old_val, dones, self.gamma, self.lamda)
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
                    lp, ent, val = self.evaluate(tr['logic_edge_index'], tr['logic_feats'], tr['phys_edge_index'],
                                                 tr['phys_feats'], tr['deploy_mask'])
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
        actions: torch.Tensor,
        parents: List[List[int]],
        topo_order: List[int],
    ) -> Tuple[torch.Tensor, int]:
        """
        Apply the offloading correction operator.

        The sampled path is scanned in topological order. Once a service is
        placed on the cloud, downstream successors that still point to the edge
        are promoted to the cloud to keep the route consistent.
        """
        corrected = actions.clone()
        must_cloud = torch.zeros(actions.size(0), dtype=torch.bool, device=actions.device)
        correction_cnt = 0

        for node in topo_order:
            parent_indices = parents[node]
            parent_must_cloud = bool(must_cloud[parent_indices].any()) if parent_indices else False

            if parent_must_cloud and corrected[node].item() != self.cloud_idx:
                corrected[node] = self.cloud_idx
                correction_cnt += 1

            must_cloud[node] = parent_must_cloud or (corrected[node].item() == self.cloud_idx)

        return corrected, correction_cnt

    def _count_switches(self, actions: torch.Tensor, topo_order: List[int]) -> int:
        """
        Count device switches along the topological execution order.
        """
        last = self.source
        switches = 0
        for node in topo_order:
            cur = actions[node].item()
            if cur != last:
                switches += 1
                last = cur
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

        tentative_actions = torch.empty(Ms, dtype=torch.long, device=h_p.device)
        row_probs = torch.zeros(Ms, Np, device=h_p.device)
        ent_sum = torch.tensor(0.0, device=h_p.device)

        for i in topo_order:
            # The deployment-induced mask is the only pre-sampling feasibility constraint.
            allowed = self._normalize_offloading_mask(static_mask[i].clone())
            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs=probs_i)
            tentative_actions[i] = dist.sample()
            row_probs[i] = probs_i
            ent_sum += dist.entropy()

        # Apply the correction operator after sampling.
        actions, correction_cnt = self._correct_offloading_actions(tentative_actions, parents, topo_order)

        logp_sum = torch.tensor(0.0, device=h_p.device)
        for i in topo_order:
            dist = torch.distributions.Categorical(probs=row_probs[i])
            logp_sum += dist.log_prob(actions[i])

        switches = self._count_switches(actions, topo_order)
        value = self.critic(h_s, h_p)
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
        actions, correction_cnt = self._correct_offloading_actions(actions, parents, topo_order)

        for i in topo_order:
            allowed = self._normalize_offloading_mask(static_mask[i].clone())
            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs=probs_i)
            a = actions[i]
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()

        switches = self._count_switches(actions, topo_order)
        value = self.critic(h_s, h_p)
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
        adv, rets = compute_returns_advantages(rewards, old_val, dones, self.gamma, self.lamda)
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
