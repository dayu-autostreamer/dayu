from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from topology_encoder import TopologyEncoders
from ppo_network import DeploymentActor, OffloadActor, ValueHead
from hedger_agent_config import DeploymentConstraintCfg, OffloadingConstraintCfg
from utils import bfs_hop_from_source, compute_returns_advantages


class HedgerDeploymentPPO(nn.Module):
    """
    顺序按服务决策：对 service i，允许在若干边缘/云节点上“部署/不部署”，
    这里采用“逐节点 Bernoulli”的方式，并在解码时用“剩余显存”白盒约束过滤。
    云端默认可用/可部署。
    """

    def __init__(self, encoder: TopologyEncoders, d_model=64, actor_lr=3e-4, critic_lr=1e-3,
                 gamma=0.99, lamda=0.95, clip_eps=0.2, update_encoder: bool = True, cloud_node_idx: int = -1,
                 constraint_cfg: DeploymentConstraintCfg = DeploymentConstraintCfg()):
        super().__init__()
        self.encoder = encoder
        self.actor = DeploymentActor(d_model)
        self.critic = ValueHead(d_model)
        params_actor = list(self.actor.parameters()) + (list(self.encoder.parameters()) if update_encoder else [])
        self.actor_opt = torch.optim.Adam(params_actor, lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lamda = lamda
        self.clip_eps = clip_eps
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

    def _initial_residual_mem(self, phys_feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 估算“瞬时可用显存”= mem_capacity * (1 - 当前 mem_util)；实际系统可替换成更精确的可用显存统计
        cap = phys_feats["mem_capacity"].float()  # [Np] MB
        util = phys_feats["mem_util_seq"][:, -1].float()  # 当前时刻 mem_util
        return cap * (1.0 - util)  # [Np] MB

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
               prev_deploy_mask: Optional[torch.Tensor] = None,  # [Ms,Np] 上一版部署，计算重部署成本
               topo_order: Optional[list] = None):
        """
        返回：
          deploy_mask: [Ms,Np] 0/1
          logp_sum/ent_sum/value：用于 PPO
          aux: {capacity_relax_cnt}
        """
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)

        residual = self._initial_residual_mem(phys_feats)  # [Np]
        model_mem = logic_feats["model_mem"].float()  # [Ms] MB
        probs = self.actor(h_s, h_p, mask=None)  # [Ms,Np] 原始 Bernoulli 概率
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)

        deploy_mask = torch.zeros(Ms, Np, dtype=torch.bool, device=h_p.device)
        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)
        capacity_relax_cnt = 0

        for i in topo_order:
            # 基于容量的允许矩阵：residual >= model_mem[i]
            allowed = residual >= model_mem[i]
            # 云端总是允许（资源充足）
            allowed[cloud_idx] = True

            # 若 enforce_capacity=True，且除云外全 False，就只能选云；否则可多标签采样
            if self.cfg.enforce_capacity and (allowed.sum() == 0 or (allowed.sum() == 1 and allowed[cloud_idx])):
                # 仅云可用
                pm = torch.zeros(Np, device=h_p.device)
                pm[cloud_idx] = 1.0
                dist = torch.distributions.Bernoulli(probs=pm)
                act = (pm > 0.5)  # 直接选云
                # 记录（使 logp/entropy 有定义）
                logp_sum += dist.log_prob(act.float()).sum()
                ent_sum += dist.entropy().mean()
                deploy_mask[i] = act
            else:
                # 允许在 allowed 节点上 Bernoulli 采样（可多副本）
                pm = probs[i].clone()
                pm = torch.where(allowed, pm, torch.zeros_like(pm))
                # 特殊情况：如果全部被置零（极端），放宽一次 -> 仅云
                if pm.sum() == 0:
                    capacity_relax_cnt += 1
                    pm = torch.zeros_like(pm)
                    pm[cloud_idx] = 1.0
                dist = torch.distributions.Bernoulli(probs=pm)
                act = (torch.rand_like(pm) < pm).float()  # [Np]
                # 至少保证云端部署（兜底）
                if act.sum() == 0:
                    act[cloud_idx] = 1.0
                logp_sum += dist.log_prob(act).sum()
                ent_sum += dist.entropy().mean()
                deploy_mask[i] = act.bool()

            # 更新剩余显存（对非云端扣减）
            for n in range(Np):
                if n == cloud_idx: continue
                if deploy_mask[i, n]:
                    residual[n] = torch.clamp(residual[n] - model_mem[i], min=0.0)

        value = self.critic(h_s, h_p)
        return deploy_mask, logp_sum, ent_sum, value.squeeze(0), {"capacity_relax_cnt": capacity_relax_cnt}

    def evaluate(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
                 deploy_mask: torch.Tensor, topo_order: Optional[list] = None):
        # 评估给定 deploy_mask 的 logp/entropy/value（用于 PPO）
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        Ms, Np = h_s.size(0), h_p.size(0)
        residual = self._initial_residual_mem(phys_feats)
        model_mem = logic_feats["model_mem"].float()
        probs = self.actor(h_s, h_p, mask=None)
        cloud_idx = self.cloud_idx if self.cloud_idx >= 0 else (Np - 1)
        if topo_order is None: topo_order = self.topo_order(logic_edge_index, Ms)
        logp_sum = torch.tensor(0., device=h_p.device)
        ent_sum = torch.tensor(0., device=h_p.device)

        for i in topo_order:
            allowed = residual >= model_mem[i]
            allowed[cloud_idx] = True
            pm = torch.where(allowed, probs[i], torch.zeros_like(probs[i]))
            # 特例处理：若 pm 全 0，则把云端设为 1
            if pm.sum() == 0:
                pm = torch.zeros_like(pm)
                pm[cloud_idx] = 1.0
            dist = torch.distributions.Bernoulli(probs=pm)
            act = deploy_mask[i].float()
            logp_sum += dist.log_prob(act).sum()
            ent_sum += dist.entropy().mean()
            for n in range(Np):
                if n == cloud_idx: continue
                if deploy_mask[i, n]: residual[n] = torch.clamp(residual[n] - model_mem[i], min=0.0)

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
                                                 tr['deploy_mask'], tr['topo_order'])
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
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.encoder.parameters()), 1.0)
                self.actor_opt.step()
                self.critic_opt.step()


class HedgerOffloadPPO(nn.Module):
    def __init__(self, encoder: TopologyEncoders, d_model=64,
                 actor_lr=3e-4, critic_lr=1e-3, update_encoder: bool = True,
                 gamma=0.99, lamda=0.95, clip_eps=0.2,
                 source_node_idx: int = 0, cloud_node_idx: int = -1,
                 constraint_cfg: OffloadingConstraintCfg = OffloadingConstraintCfg()):
        super().__init__()
        self.encoder = encoder
        self.actor = OffloadActor(d_model)
        self.critic = ValueHead(d_model)
        params_actor = list(self.actor.parameters()) + (list(self.encoder.parameters()) if update_encoder else [])
        self.actor_opt = torch.optim.Adam(params_actor, lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
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

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index, phys_feats,
               static_mask: torch.Tensor, topo_order: Optional[list] = None):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
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

    # -------- PPO 更新（以“图级一步”为样本） --------
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
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.encoder.parameters()), 1.0)
                self.actor_opt.step()
                self.critic_opt.step()
