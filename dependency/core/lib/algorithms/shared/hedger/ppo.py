import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch

from .gnn import LogicGNN, PhysicalGNN


class DeploymentActor(nn.Module):
    """Multi-label actor for deployment: per (service, node) Bernoulli"""
    def __init__(self, dim):
        super().__init__()
        self.linear_s = nn.Linear(dim, dim)
        self.linear_p = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, h_s, h_p, mask=None):
        scores = torch.matmul(self.linear_s(h_s), self.linear_p(h_p).t()) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return torch.sigmoid(scores)

class OffloadActor(nn.Module):
    """Single-choice actor for offload: per service Categorical"""
    def __init__(self, dim):
        super().__init__()
        self.linear_s = nn.Linear(dim, dim)
        self.linear_p = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, h_s, h_p, mask=None):
        scores = torch.matmul(self.linear_s(h_s), self.linear_p(h_p).t()) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        return F.softmax(scores, dim=-1)


class HedgerDeploymentPPO(nn.Module):
    """
    PPO agent for model deployment (large timescale).
    Uses LogicGNN + PhysicalGNN + DeploymentActor.
    """
    def __init__(self, logic_in_feats, phys_in_feats, emb_dim=64,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, clip_eps=0.2):
        super().__init__()
        self.logic_encoder = LogicGNN(logic_in_feats, emb_dim, emb_dim)
        self.phys_encoder = PhysicalGNN(phys_in_feats, emb_dim, emb_dim)
        self.actor = DeploymentActor(emb_dim)
        self.critic = nn.Sequential(nn.Linear(2*emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim,1))
        self.actor_opt = torch.optim.Adam(
            list(self.logic_encoder.parameters()) +
            list(self.phys_encoder.parameters()) +
            list(self.actor.parameters()), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def encode(self, logic_data, phys_data):
        h_s = self.logic_encoder(logic_data.x, logic_data.edge_index)
        h_p = self.phys_encoder(phys_data.x, phys_data.edge_index)
        return h_s, h_p

    def forward(self, logic_data, phys_data, mask=None):
        h_s, h_p = self.encode(logic_data, phys_data)
        return self.actor(h_s, h_p, mask)

    def evaluate(self, logic_batch, phys_batch, actions, mask=None):
        probs = self.forward(logic_batch, phys_batch, mask)
        dist = torch.distributions.Bernoulli(probs)
        logp_node = dist.log_prob(actions).sum(dim=1)
        # aggregate per graph
        batch_vec = logic_batch.batch
        batch_size = logic_batch.num_graphs
        logp = torch.zeros(batch_size, device=probs.device)
        for i, g in enumerate(batch_vec):
            logp[g] += logp_node[i]
        entropy = dist.entropy().mean()
        vs = global_mean_pool(
            self.logic_encoder(logic_batch.x, logic_batch.edge_index), logic_batch.batch)
        vp = global_mean_pool(
            self.phys_encoder(phys_batch.x, phys_batch.edge_index), phys_batch.batch)
        value = self.critic(torch.cat([vs, vp], dim=-1)).squeeze(-1)
        return logp, entropy, value

    def compute_gae(self, rewards, values, dones):
        advs = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.clip_eps * (1 - dones[t]) * gae
            advs.insert(0, gae)
        return advs

    def ppo_update(self, transitions, epochs=4, batch_size=64):
        device = transitions[0]['action'].device
        # detach old log probs
        old_lp = torch.stack([t['log_prob'].detach().to(device) for t in transitions])
        values = [t['value'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        dones = [t['done'] for t in transitions]
        advs = self.compute_gae(rewards, values, dones)
        rets = [a + v for a, v in zip(advs, values)]
        adv_tensor = torch.tensor(advs, device=device)
        ret_tensor = torch.tensor(rets, device=device)
        T = len(transitions)
        for _ in range(epochs):
            perm = torch.randperm(T)
            for start in range(0, T, batch_size):
                idx = perm[start:start+batch_size]
                new_lp_list, ent_list, val_list = [], [], []
                for i in idx:
                    tr = transitions[i]
                    lb = Batch.from_data_list([tr['logic_data']])
                    pb = Batch.from_data_list([tr['phys_data']])
                    act = tr['action'].to(device)
                    lp, ent, val = self.evaluate(lb, pb, act)
                    new_lp_list.append(lp)
                    ent_list.append(ent)
                    val_list.append(val)
                new_lp = torch.stack(new_lp_list)  # [batch]
                ent_mean = torch.stack(ent_list).mean()  # scalar
                new_val = torch.stack(val_list).squeeze(-1)  # [batch]
                oldb = old_lp[idx]
                retb = ret_tensor[idx]
                advb = adv_tensor[idx]
                ratio = torch.exp(new_lp - oldb)
                s1 = ratio * advb
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advb
                ploss = -torch.min(s1, s2).mean()
                vloss = F.mse_loss(new_val, retb)
                loss = ploss + 0.5 * vloss - 0.01 * ent_mean
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()


class HedgerOffloadPPO(nn.Module):
    """
    PPO agent for task offloading (small timescale).
    Uses LogicGNN + PhysicalGNN + OffloadActor.
    """
    def __init__(self, logic_in_feats, phys_in_feats, emb_dim=64,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, clip_eps=0.2):
        super().__init__()
        self.logic_encoder = LogicGNN(logic_in_feats, emb_dim, emb_dim)
        self.phys_encoder = PhysicalGNN(phys_in_feats, emb_dim, emb_dim)
        self.actor = OffloadActor(emb_dim)
        self.critic = nn.Sequential(nn.Linear(2*emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim,1))
        self.actor_opt = torch.optim.Adam(
            list(self.logic_encoder.parameters()) +
            list(self.phys_encoder.parameters()) +
            list(self.actor.parameters()), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.clip_eps = clip_eps

    def encode(self, logic_data, phys_data):
        h_s = self.logic_encoder(logic_data.x, logic_data.edge_index)
        h_p = self.phys_encoder(phys_data.x, phys_data.edge_index)
        return h_s, h_p

    def forward(self, logic_data, phys_data, mask=None):
        h_s, h_p = self.encode(logic_data, phys_data)
        return self.actor(h_s, h_p, mask)

    def evaluate(self, logic_batch, phys_batch, actions, mask=None):
        probs = self.forward(logic_batch, phys_batch, mask)
        dist = torch.distributions.Categorical(probs)
        logp_node = dist.log_prob(actions)
        batch_vec = logic_batch.batch
        batch_size = logic_batch.num_graphs
        logp = torch.zeros(batch_size, device=probs.device)
        for i, g in enumerate(batch_vec):
            logp[g] += logp_node[i]
        entropy = dist.entropy().mean()
        vs = global_mean_pool(
            self.logic_encoder(logic_batch.x, logic_batch.edge_index), logic_batch.batch)
        vp = global_mean_pool(
            self.phys_encoder(phys_batch.x, phys_batch.edge_index), phys_batch.batch)
        value = self.critic(torch.cat([vs, vp], dim=-1)).squeeze(-1)
        return logp, entropy, value

    def compute_gae(self, rewards, values, dones):
        advs = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.clip_eps * (1 - dones[t]) * gae
            advs.insert(0, gae)
        return advs

    def ppo_update(self, transitions, epochs=4, batch_size=64):
        device = transitions[0]['action'].device
        old_lp = torch.stack([t['log_prob'].detach().to(device) for t in transitions])
        values = [t['value'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        dones = [t['done'] for t in transitions]
        advs = self.compute_gae(rewards, values, dones)
        rets = [a + v for a, v in zip(advs, values)]
        adv_tensor = torch.tensor(advs, device=device)
        ret_tensor = torch.tensor(rets, device=device)
        T = len(transitions)
        for _ in range(epochs):
            perm = torch.randperm(T)
            for start in range(0, T, batch_size):
                idx = perm[start:start+batch_size]
                new_lp_list, ent_list, val_list = [], [], []
                for i in idx:
                    tr = transitions[i]
                    lb = Batch.from_data_list([tr['logic_data']])
                    pb = Batch.from_data_list([tr['phys_data']])
                    act = tr['action'].to(device)
                    lp, ent, val = self.evaluate(lb, pb, act)
                    new_lp_list.append(lp)
                    ent_list.append(ent)
                    val_list.append(val)
                new_lp = torch.stack(new_lp_list)
                ent_mean = torch.stack(ent_list).mean()
                new_val = torch.stack(val_list).squeeze(-1)
                oldb = old_lp[idx]
                retb = ret_tensor[idx]
                advb = adv_tensor[idx]
                ratio = torch.exp(new_lp - oldb)
                s1 = ratio * advb
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advb
                ploss = -torch.min(s1, s2).mean()
                vloss = F.mse_loss(new_val, retb)
                loss = ploss + 0.5 * vloss - 0.01 * ent_mean
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                self.actor_opt.step()
                self.critic_opt.step()