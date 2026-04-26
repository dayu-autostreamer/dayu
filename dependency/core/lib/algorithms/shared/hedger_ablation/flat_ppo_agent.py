from typing import Optional, List, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lib.algorithms.shared.hedger.hedger_config import DeploymentConstraintCfg, OffloadingConstraintCfg
from core.lib.algorithms.shared.hedger.ppo_agent import (
    _DeploymentBackbonePPO,
    _mean_or_zero,
    _move_tensor_dict_to_device,
    _parameters_grad_norm,
    _scalar_to_float,
    _std_or_zero,
    _tensor_std_float,
)
from core.lib.algorithms.shared.hedger.ppo_network import OffloadActor
from core.lib.algorithms.shared.hedger.topology_encoder import TopologyEncoders
from core.lib.algorithms.shared.hedger.utils import compute_returns_advantages

__all__ = ("HedgerFlatPPO",)


class HedgerFlatPPO(_DeploymentBackbonePPO):
    """
    Flat ablation policy for joint deployment/offloading decisions.

    This keeps Hedger's graph encoder and action corrections, but removes the
    two-timescale hierarchy: one PPO module samples both deployment replicas and
    offloading targets, then optimizes the joint log-probability with one critic.
    """

    def __init__(
            self,
            encoder: TopologyEncoders,
            d_model=64,
            actor_lr=3e-4,
            critic_lr=1e-3,
            gamma=0.99,
            lamda=0.95,
            clip_eps=0.2,
            update_encoder: bool = True,
            cloud_node_idx: int = -1,
            deployment_constraint_cfg: DeploymentConstraintCfg = DeploymentConstraintCfg(),
            offloading_constraint_cfg: OffloadingConstraintCfg = OffloadingConstraintCfg(),
    ):
        super().__init__(
            encoder=encoder,
            d_model=d_model,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            lamda=lamda,
            clip_eps=clip_eps,
            update_encoder=update_encoder,
            cloud_node_idx=cloud_node_idx,
            constraint_cfg=deployment_constraint_cfg,
        )
        self.offload_actor = OffloadActor(d_model)
        self.cloud_idx = cloud_node_idx
        self.offloading_cfg = offloading_constraint_cfg
        self._rebuild_actor_optimizer(extra_actor_modules=[self.offload_actor])

    @staticmethod
    def _build_parents(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        row, col = edge_index
        parents = [[] for _ in range(num_nodes)]
        for u, v in zip(row.tolist(), col.tolist()):
            parents[v].append(u)
        return parents

    def _normalize_offloading_mask(self, base: torch.Tensor) -> torch.Tensor:
        if base.any():
            return base
        allowed = torch.zeros_like(base)
        allowed[self._cloud_index(base.numel())] = True
        return allowed

    def _correct_offloading_actions(
            self,
            raw_actions: torch.Tensor,
            parents: List[List[int]],
            topo_order: List[int],
    ) -> Tuple[torch.Tensor, int]:
        corrected = raw_actions.clone()
        correction_cnt = 0
        cloud_idx = int(self.cloud_idx)
        if cloud_idx < 0:
            cloud_idx = int(raw_actions.max().item()) if raw_actions.numel() else -1
        for node_idx in topo_order:
            parent_indices = parents[node_idx]
            if not parent_indices:
                continue
            if any(int(corrected[parent].item()) == cloud_idx for parent in parent_indices):
                if int(corrected[node_idx].item()) != cloud_idx:
                    corrected[node_idx] = cloud_idx
                    correction_cnt += 1
        return corrected, correction_cnt

    @staticmethod
    def _offloading_correction_cost(
            raw_probs: torch.Tensor,
            raw_actions: torch.Tensor,
            corrected_actions: torch.Tensor,
    ) -> float:
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

    @torch.no_grad()
    def policy(self, logic_edge_index, logic_feats, phys_edge_index,
               phys_feats, topo_order: Optional[list] = None,
               prev_deploy_mask: Optional[torch.Tensor] = None,
               deterministic: bool = False):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)

        cloud_idx = self._cloud_index(Np)
        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)

        raw_deploy_mask = torch.zeros((Ms, Np), dtype=torch.bool, device=h_s.device)
        raw_deploy_probs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_s.device)
        deploy_logp = torch.tensor(0.0, device=h_s.device)
        deploy_ent_terms = []
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
                sampled_bits = probs_sample >= 0.5 if deterministic else torch.rand_like(probs_sample) < probs_sample
                sampled_row = torch.where(
                    stochastic_allowed,
                    sampled_bits,
                    torch.zeros_like(sampled_bits, dtype=torch.bool),
                )
                deploy_logp += dist.log_prob(sampled_row.float()).sum()
                deploy_ent_terms.append(dist.entropy()[stochastic_allowed].mean())
                raw_deploy_probs[service_idx] = probs_raw
            sampled_row[cloud_idx] = True
            raw_deploy_mask[service_idx] = sampled_row
            raw_deploy_probs[service_idx, cloud_idx] = 1.0

        (
            deploy_mask,
            capacity_relax_cnt,
            capacity_relax_cost,
            edge_cover_repair_cnt,
            edge_cover_repair_cost,
            edge_cover_unmet,
        ) = self._project_deployment_mask(
            raw_deploy_mask,
            raw_probs=raw_deploy_probs,
            logic_feats=logic_feats,
            phys_feats=phys_feats,
            prev_deploy_mask=prev_deploy_mask,
            static_allowed=static_allowed,
        )

        parents = self._build_parents(logic_edge_index, Ms)
        raw_actions = torch.empty(Ms, dtype=torch.long, device=h_p.device)
        raw_offloading_probs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        off_logp = torch.tensor(0.0, device=h_p.device)
        off_ent_terms = []
        for service_idx in topo_order:
            allowed = self._normalize_offloading_mask(deploy_mask[service_idx].clone())
            probs_i = self.offload_actor(h_s[service_idx:service_idx + 1], h_p, allowed.unsqueeze(0))[0]
            raw_offloading_probs[service_idx] = probs_i
            dist = torch.distributions.Categorical(probs=probs_i)
            raw_actions[service_idx] = torch.argmax(probs_i) if deterministic else dist.sample()
            off_logp += dist.log_prob(raw_actions[service_idx])
            off_ent_terms.append(dist.entropy())

        actions, off_correction_cnt = self._correct_offloading_actions(raw_actions, parents, topo_order)
        off_correction_cost = self._offloading_correction_cost(raw_offloading_probs, raw_actions, actions)
        off_aux_cost = float(self.offloading_cfg.penalty_relax) * off_correction_cost

        deploy_entropy = (
            torch.stack(deploy_ent_terms).mean()
            if deploy_ent_terms else torch.tensor(0.0, device=h_s.device)
        )
        off_entropy = (
            torch.stack(off_ent_terms).mean()
            if off_ent_terms else torch.tensor(0.0, device=h_s.device)
        )
        value = self.critic(h_s, h_p, self._deployment_context(h_p, prev_deploy_mask))
        return deploy_mask, actions, deploy_logp + off_logp, (deploy_entropy + off_entropy) * 0.5, value.squeeze(0), {
            "capacity_relax_cnt": capacity_relax_cnt,
            "capacity_relax_cost": capacity_relax_cost,
            "edge_cover_repair_cnt": edge_cover_repair_cnt,
            "edge_cover_repair_cost": edge_cover_repair_cost,
            "edge_cover_unmet": edge_cover_unmet,
            "raw_deploy_mask": raw_deploy_mask,
            "raw_actions": raw_actions,
            "correction_cnt": off_correction_cnt,
            "correction_cost": off_correction_cost,
            "aux_cost": off_aux_cost,
        }

    def evaluate(
            self,
            logic_edge_index,
            logic_feats,
            phys_edge_index,
            phys_feats,
            raw_deploy_mask: torch.Tensor,
            raw_actions: torch.Tensor,
            static_mask: torch.Tensor,
            prev_deploy_mask: Optional[torch.Tensor] = None,
            topo_order: Optional[list] = None,
    ):
        h_s, h_p = self.encoder.encode(logic_edge_index, logic_feats, phys_edge_index, phys_feats)
        h_s, h_p = self._adapt_embeddings(h_s, h_p)
        Ms, Np = h_s.size(0), h_p.size(0)
        if topo_order is None:
            topo_order = self.topo_order(logic_edge_index, Ms)

        static_allowed = self._static_allowed_mask(phys_feats, logic_feats)
        cloud_idx = self._cloud_index(Np)
        raw_deploy_mask = self._enforce_cloud_replica(raw_deploy_mask.bool())
        eps = 1e-6

        deploy_logp = torch.tensor(0.0, device=h_s.device)
        deploy_ent_terms = []
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
                acts_row = raw_deploy_mask[service_idx].float()
                deploy_logp += dist.log_prob(acts_row).masked_select(stochastic_allowed).sum()
                deploy_ent_terms.append(dist.entropy().masked_select(stochastic_allowed).mean())

        parents = self._build_parents(logic_edge_index, Ms)
        corrected_actions, correction_cnt = self._correct_offloading_actions(raw_actions, parents, topo_order)
        raw_probs = torch.zeros((Ms, Np), dtype=torch.float32, device=h_p.device)
        off_logp = torch.tensor(0.0, device=h_p.device)
        off_ent_terms = []
        for service_idx in topo_order:
            allowed = self._normalize_offloading_mask(static_mask[service_idx].clone())
            probs_i = self.offload_actor(h_s[service_idx:service_idx + 1], h_p, allowed.unsqueeze(0))[0]
            raw_probs[service_idx] = probs_i
            dist = torch.distributions.Categorical(probs=probs_i)
            off_logp += dist.log_prob(raw_actions[service_idx])
            off_ent_terms.append(dist.entropy())

        deploy_entropy = (
            torch.stack(deploy_ent_terms).mean()
            if deploy_ent_terms else torch.tensor(0.0, device=h_s.device)
        )
        off_entropy = (
            torch.stack(off_ent_terms).mean()
            if off_ent_terms else torch.tensor(0.0, device=h_s.device)
        )
        correction_cost = self._offloading_correction_cost(raw_probs, raw_actions, corrected_actions)
        aux_cost = float(self.offloading_cfg.penalty_relax) * correction_cost
        value = self.critic(h_s, h_p, self._deployment_context(h_p, prev_deploy_mask))
        return deploy_logp + off_logp, (deploy_entropy + off_entropy) * 0.5, value.squeeze(0), {
            "correction_cnt": correction_cnt,
            "correction_cost": correction_cost,
            "aux_cost": aux_cost,
        }

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
                "raw_deploy_mask": tr["raw_deploy_mask"].to(device),
                "raw_actions": tr["raw_actions"].to(device),
                "static_mask": tr["static_mask"].to(device),
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
                    lp, ent, val, _ = self.evaluate(
                        tr['logic_edge_index'],
                        tr['logic_feats'],
                        tr['phys_edge_index'],
                        tr['phys_feats'],
                        tr['raw_deploy_mask'],
                        tr['raw_actions'],
                        tr['static_mask'],
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
