from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topology_encoder import TopologyEncoders
from .ppo_network import DeploymentActor, OffloadActor, ValueHead, FeatureAdapter
from .hedger_config import DeploymentConstraintCfg, OffloadingConstraintCfg
from .utils import bfs_hop_from_source, compute_returns_advantages


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
        To construct a "static physical mask": Filter out physically impossible deployment combinations based solely on the model size and the total memory of the device.

        Return:

        `static_allowed`: `[Ms, Np]` boolean array

        `static_allowed[i, n]` = `True` indicates that service `i` can be physically accommodated on device `n`.
        """
        cap = phys_feats["mem_capacity"].float()  # [Np]
        model_mem = logic_feats["model_mem"].float()  # [Ms]
        Ms = model_mem.size(0)
        Np = cap.size(0)

        # [Ms, Np]
        static_allowed = model_mem.view(Ms, 1) <= cap.view(1, Np)

        # Cloud nodes are always treated as "feasible alternatives."
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
            Calculate the "memory budget available for deploying the new model in this round."

            - If `logic_feats` or `prev_deploy_mask` are not provided, revert to the old logic:
              `residual = mem_capacity * (1 - mem_util).`

            - If `prev_deploy_mask` is provided, indicating that we will first unload the previous model before deploying the new one, calculate as follows:

              `total_used = cap * util`
              `prev_usage = sum_i prev_deploy_mask[i, n] * model_mem[i]`

              `baseline = max(total_used - prev_usage, 0)`  # Memory used by non-RL models
              `residual_0 = max(cap - baseline, 0)`  # Budget for the new model
        """
        cap = phys_feats["mem_capacity"].float()  # [Np] MB

        if logic_feats is None or prev_deploy_mask is None:
            util = phys_feats["mem_util_seq"][:, -1].float()  # 当前总 mem_util
            return cap * (1.0 - util)

        util = phys_feats["mem_util_seq"][:, -1].float()  # [Np]
        total_used = cap * util  # [Np]，当前总占用

        model_mem = logic_feats["model_mem"].float()  # [Ms]
        # prev_deploy_mask: [Ms, Np] -> [Np] (Total GPU memory usage of the old model on each node)
        prev_usage = torch.matmul(prev_deploy_mask.float().t(), model_mem)  # [Np]
        baseline = torch.clamp(total_used - prev_usage, min=0.0)  # [Np]

        # Budget allocated for the "new round of model deployment"
        residual = torch.clamp(cap - baseline, min=0.0)  # [Np]
        return residual

    def _adapt_embeddings(self, h_s: torch.Tensor, h_p: torch.Tensor):
        """
        Perform a task-specific residual transformation on the (h_s, h_p) provided by the shared encoder
        """
        h_s = h_s + self.s_adapter(h_s)
        h_p = h_p + self.p_adapter(h_p)
        return h_s, h_p

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index,
               phys_feats, topo_order: Optional[list] = None,
               prev_deploy_mask: Optional[torch.Tensor] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)

        # Current deployment budget (consider baseline + uninstall old models)
        residual = self._initial_residual_mem(phys_feats, logic_feats, prev_deploy_mask)  # [Np]
        model_mem = logic_feats["model_mem"].float()  # [Ms]
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)

        # Static physical mask
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]

        # Masking out physically impossible combinations using static masking
        # The `static_allowed` parameter is passed to the actor, which then applies masked_fill to the attention scores.
        probs_raw = self.actor(h_s, h_p, mask=static_allowed)  # [Ms, Np], probability after the sigmoid

        # To ensure numerical stability, perform a slight trimming on the probabilities used for log_prob;
        # For positions that are not allowed, replace 0 with a very small eps to avoid log(0)
        eps = 1e-6
        probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
        probs_log = torch.where(static_allowed, probs_log, torch.full_like(probs_log, eps))

        # Strictly Prohibit Sampling at Positions Marked as static_disallowed
        probs_sample = torch.where(static_allowed, probs_raw, torch.zeros_like(probs_raw))

        # Multilabel Bernoulli sampling
        # Here, probs_log is used to construct the distribution, and probs_sample is used for actual sampling
        # (positions that are not allowed will always sample 0)
        dist = torch.distributions.Bernoulli(probs=probs_log)
        acts = torch.rand_like(probs_sample) < probs_sample  # bool [Ms, Np]
        deploy_mask = acts.clone()

        # Each service must be deployed on at least one device (as a backup: deployed on the cloud).
        row_sum = deploy_mask.sum(dim=1)  # [Ms]
        need_cloud = (row_sum == 0)
        if need_cloud.any():
            deploy_mask[need_cloud, cloud_idx] = True

        # Post-sampling capacity projection (post-projection)
        capacity_relax_cnt = 0
        if self.cfg.enforce_capacity:
            # Check each device to see if it exceeds the budget, and if it does, delete some copies.
            for n in range(Np):
                if n == cloud_idx:
                    # The cloud side can be considered as "infinity" or controlled individually;
                    # here, no capacity trimming is performed.
                    continue

                while True:
                    # The current load of device n
                    used_n = (deploy_mask[:, n].float() * model_mem).sum()

                    # The budget has been met, exit the loop.
                    if used_n <= residual[n] + 1e-6:
                        break

                    # Find all services currently deployed on device n.
                    candidates = torch.nonzero(deploy_mask[:, n], as_tuple=False).view(-1)
                    if candidates.numel() == 0:
                        break

                    # Strategy: Delete the service with the "minimum probability" of strategy on this device
                    cand_probs = probs_log[candidates, n]
                    drop_local_idx = torch.argmin(cand_probs)
                    drop_service_idx = candidates[drop_local_idx]

                    deploy_mask[drop_service_idx, n] = False
                    capacity_relax_cnt += 1

            # Capacity adjustment may cause some services to have no replicas at all,
            # as a last resort: enforce cloud deployment.
            row_sum = deploy_mask.sum(dim=1)
            need_cloud = (row_sum == 0)
            if need_cloud.any():
                deploy_mask[need_cloud, cloud_idx] = True

        # Calculate logp and entropy under the "static Bernoulli distribution"
        act_float = deploy_mask.float()
        logp = dist.log_prob(act_float)       # [Ms, Np]
        ent = dist.entropy()                  # [Ms, Np]

        logp_sum = logp.sum()
        ent_sum = ent.mean()

        value = self.critic(h_s, h_p)

        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt
        }

    def evaluate(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
                 deploy_mask: torch.Tensor, topo_order: Optional[list] = None,
                 prev_deploy_mask: Optional[torch.Tensor] = None):
        """
        Under the current parameters, calculate log_prob, entropy, and value for the given deploy_mask:
        - Policy distribution: Static physical mask + independent Bernoulli (consistent with the policy stage)
        - The capacity constraint process is no longer replicated; capacity constraints are considered as a "projection" of the environment on actions.
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)

        # Static physical mask
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)  # [Ms, Np]

        probs_raw = self.actor(h_s, h_p, mask=static_allowed)
        eps = 1e-6
        probs_log = torch.clamp(probs_raw, eps, 1.0 - eps)
        probs_log = torch.where(static_allowed, probs_log, torch.full_like(probs_log, eps))

        dist = torch.distributions.Bernoulli(probs=probs_log)

        act_float = deploy_mask.float()
        logp = dist.log_prob(act_float)     # [Ms, Np]
        ent = dist.entropy()                # [Ms, Np]

        logp_sum = logp.sum()
        ent_sum = ent.mean()

        value = self.critic(h_s, h_p)
        return logp_sum, ent_sum, value.squeeze(0)

    def ppo_update(self, transitions: List[dict], epochs=4, batch_size=16, clip_eps=None, entropy_coef=0.01,
                   value_coef=0.5):
        clip_eps = self.clip_eps if clip_eps is None else clip_eps
        device = next(self.parameters()).device
        old_logp = torch.stack([t['logp'].detach().to(device) for t in transitions])
        old_val = [t['value'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        dones = [t['done'] for t in transitions]
        adv, rets = compute_returns_advantages(rewards, old_val, dones, self.gamma, self.lamda)
        adv = torch.tensor(adv, device=device)
        rets = torch.tensor(rets, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        T = len(transitions)
        for _ in range(epochs):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, batch_size):
                idx = perm[start:start + batch_size]
                new_logp_list = []
                new_val_list = []
                ent_list = []
                for j in idx:
                    tr = transitions[j]
                    lp, ent, val = self.evaluate(tr['logic_edge_index'], tr['logic_feats'],
                                                 tr['phys_edge_index'], tr['phys_feats'],
                                                 tr['deploy_mask'], tr['topo_order'],
                                                 tr['prev_deploy_mask'])
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

    def _build_metric(self, phys_edge_index: torch.Tensor, N: int):
        if not self.cfg.use_monotone_metric: return None
        return bfs_hop_from_source(phys_edge_index, N, self.source)

    def _adapt_embeddings(self, h_s: torch.Tensor, h_p: torch.Tensor):
        """
        Perform a task-specific residual transformation on the (h_s, h_p) provided by the shared encoder
        """
        h_s = h_s + self.s_adapter(h_s)
        h_p = h_p + self.p_adapter(h_p)
        return h_s, h_p

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
               static_mask: torch.Tensor, topo_order: Optional[list] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)

        actions = torch.empty(Ms, dtype=torch.long, device=h_p.device)
        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)

        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)
        metric = self._build_metric(phys_edge_index, Np)
        visited = torch.zeros(Np, dtype=torch.bool, device=h_p.device)
        last = self.source
        switches = 0
        relax_cnt = 0

        for i in topo_order:
            base = static_mask[i].clone()
            if self.cfg.cloud_sticky and (last == self.cloud_idx):
                allowed = torch.zeros_like(base)
                allowed[self.cloud_idx] = base[self.cloud_idx]
            else:
                allowed = base.clone()
                if self.cfg.forbid_return:
                    forbid = visited.clone()
                    if self.cfg.allow_stay: forbid[last] = False
                    allowed = allowed & (~forbid)
                if metric is not None:
                    if self.cfg.metric_non_decreasing:
                        allowed = allowed & (metric >= metric[last])
                    else:
                        allowed = allowed & (metric <= metric[last])
            if not allowed.any():
                relax_cnt += 1
                if self.cfg.allow_stay and base[last]:
                    allowed = torch.zeros_like(base)
                    allowed[last] = True
                elif base[self.cloud_idx]:
                    allowed = torch.zeros_like(base)
                    allowed[self.cloud_idx] = True
                else:
                    allowed = base

            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs_i)
            a = dist.sample()
            actions[i] = a
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()

            if a != last:
                switches += 1
                visited[last] = True
                last = a

        value = self.critic(h_s, h_p)
        aux_cost = self.cfg.penalty_switch * switches + self.cfg.penalty_relax * relax_cnt
        return actions, logp_sum, ent_sum, value.squeeze(0), {"switches": switches, "relax_cnt": relax_cnt,
                                                              "aux_cost": aux_cost}

    def evaluate(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
                 actions: torch.Tensor, static_mask: torch.Tensor, topo_order: Optional[list] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)

        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)

        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)
        metric = self._build_metric(phys_edge_index, Np)
        visited = torch.zeros(Np, dtype=torch.bool, device=h_p.device)
        last = self.source
        switches = 0
        relax_cnt = 0

        for i in topo_order:
            base = static_mask[i].clone()
            if self.cfg.cloud_sticky and (last == self.cloud_idx):
                allowed = torch.zeros_like(base)
                allowed[self.cloud_idx] = base[self.cloud_idx]
            else:
                allowed = base.clone()
                if self.cfg.forbid_return:
                    forbid = visited.clone()
                    if self.cfg.allow_stay: forbid[last] = False
                    allowed = allowed & (~forbid)
                if metric is not None:
                    if self.cfg.metric_non_decreasing:
                        allowed = allowed & (metric >= metric[last])
                    else:
                        allowed = allowed & (metric <= metric[last])

            if not allowed.any():
                relax_cnt += 1
                if self.cfg.allow_stay and base[last]:
                    allowed = torch.zeros_like(base)
                    allowed[last] = True
                elif base[self.cloud_idx]:
                    allowed = torch.zeros_like(base)
                    allowed[self.cloud_idx] = True
                else:
                    allowed = base

            probs_i = self.actor(h_s[i:i + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs_i)
            a = actions[i]
            logp_sum += dist.log_prob(a)
            ent_sum += dist.entropy()

            if a != last:
                switches += 1
                visited[last] = True
                last = a

        value = self.critic(h_s, h_p)
        aux_cost = self.cfg.penalty_switch * switches + self.cfg.penalty_relax * relax_cnt
        return logp_sum, ent_sum, value.squeeze(0), {"switches": switches, "relax_cnt": relax_cnt, "aux_cost": aux_cost}

    def ppo_update(self, transitions: List[dict], epochs=4, batch_size=32, clip_eps=None, entropy_coef=0.01,
                   value_coef=0.5):
        clip_eps = self.clip_eps if clip_eps is None else clip_eps
        device = next(self.parameters()).device
        old_logp = torch.stack([t['logp'].detach().to(device) for t in transitions])  # [T]
        old_val = [t['value'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        dones = [t['done'] for t in transitions]
        adv, rets = compute_returns_advantages(rewards, old_val, dones, self.gamma, self.lamda)
        adv = torch.tensor(adv, device=device)
        rets = torch.tensor(rets, device=device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        T = len(transitions)
        for _ in range(epochs):
            perm = torch.randperm(T, device=device)
            for start in range(0, T, batch_size):
                idx = perm[start:start + batch_size]
                new_logp_list = []
                new_val_list = []
                ent_list = []
                for j in idx:
                    tr = transitions[j]
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
