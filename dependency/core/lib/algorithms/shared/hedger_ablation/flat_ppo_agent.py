from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.lib.algorithms.shared.hedger.hedger_config import DeploymentConstraintCfg
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

    This keeps Hedger's graph encoder and action projection, but removes the
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

    def _dynamic_allowed_row(
            self,
            base_allowed: torch.Tensor,
            parents: List[List[int]],
            service_idx: int,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        allowed = self._normalize_offloading_mask(base_allowed)
        cloud_idx = self._cloud_index(allowed.numel())
        if cloud_idx < 0:
            return allowed
        if any(int(actions[parent].item()) == cloud_idx for parent in parents[service_idx]):
            cloud_only = torch.zeros_like(allowed)
            cloud_only[cloud_idx] = True
            return cloud_only
        return allowed

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
        effective_positive_mask = deploy_mask & static_allowed.bool()
        if cloud_idx >= 0:
            negative_action_mask[:, cloud_idx] = False
            effective_positive_mask[:, cloud_idx] = False
        (
            deploy_logp,
            deploy_entropy,
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

        parents = self._build_parents(logic_edge_index, Ms)
        actions = torch.empty(Ms, dtype=torch.long, device=h_p.device)
        off_logp = torch.tensor(0.0, device=h_p.device)
        off_ent_terms = []
        for service_idx in topo_order:
            allowed = self._dynamic_allowed_row(deploy_mask[service_idx].clone(), parents, service_idx, actions)
            probs_i = self.offload_actor(h_s[service_idx:service_idx + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs=probs_i)
            actions[service_idx] = torch.argmax(probs_i) if deterministic else dist.sample()
            off_logp += dist.log_prob(actions[service_idx])
            off_ent_terms.append(dist.entropy())

        off_entropy = (
            torch.stack(off_ent_terms).mean()
            if off_ent_terms else torch.tensor(0.0, device=h_s.device)
        )
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        actor_debug = dict(pair_ctx)
        actor_debug.update(executed_pair_ctx or {})
        actor_debug.update(option_debug)
        actor_debug.update({
            "capacity_removed_mask": capacity_removed_mask,
            "positive_mask": positive_mask,
            "negative_mask": negative_mask,
        })
        if cloud_idx > 0:
            raw_edge_count = (raw_deploy_mask[:, :cloud_idx] & static_allowed[:, :cloud_idx].bool()).float().sum(dim=1)
            executed_edge_count = deploy_mask[:, :cloud_idx].float().sum(dim=1)
            lost_edge_service = (raw_edge_count > 0.0) & (executed_edge_count <= 0.0)
            projection_last_edge_removed_mask = capacity_removed_mask.clone()
            projection_last_edge_removed_mask[:, :cloud_idx] = (
                capacity_removed_mask[:, :cloud_idx] & lost_edge_service.view(-1, 1)
            )
            projection_last_edge_removed_mask[:, cloud_idx:] = False
        else:
            projection_last_edge_removed_mask = torch.zeros_like(capacity_removed_mask, dtype=torch.bool)
        actor_debug["projection_last_edge_removed_mask"] = projection_last_edge_removed_mask
        return deploy_mask, actions, deploy_logp + off_logp, (deploy_entropy + off_entropy) * 0.5, value.squeeze(0), {
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
            "actor_debug": actor_debug,
        }

    def evaluate(
            self,
            logic_edge_index,
            logic_feats,
            phys_edge_index,
            phys_feats,
            raw_deploy_mask: torch.Tensor,
            actions: torch.Tensor,
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
        static_mask = static_mask.bool()
        (
            _q_embedding,
            _k_embedding,
            _qk_scores,
            _qk_feature,
            _pair_adjustment,
            candidate_features,
            select_logits,
            _pair_ctx,
        ) = self._deployment_actor_terms(
            h_s,
            h_p,
            logic_edge_index,
            logic_feats,
            phys_feats,
            static_allowed,
            prev_deploy_mask=prev_deploy_mask,
        )
        negative_action_mask = raw_deploy_mask & ~static_mask & static_allowed.bool()
        effective_positive_mask = static_mask & static_allowed.bool()
        if cloud_idx >= 0:
            negative_action_mask[:, cloud_idx] = False
            effective_positive_mask[:, cloud_idx] = False
        (
            deploy_logp,
            deploy_entropy,
            _positive_logp,
            _negative_logp,
            _positive_mask,
            _negative_mask,
        ) = self._deployment_select_logp_entropy(
            select_logits,
            static_mask,
            static_allowed,
            topo_order=topo_order,
            positive_mask=effective_positive_mask,
            negative_mask=negative_action_mask,
        )

        actions = actions.long()

        parents = self._build_parents(logic_edge_index, Ms)
        off_logp = torch.tensor(0.0, device=h_p.device)
        off_ent_terms = []
        for service_idx in topo_order:
            allowed = self._dynamic_allowed_row(static_mask[service_idx].clone(), parents, service_idx, actions)
            probs_i = self.offload_actor(h_s[service_idx:service_idx + 1], h_p, allowed.unsqueeze(0))[0]
            dist = torch.distributions.Categorical(probs=probs_i)
            off_logp += dist.log_prob(actions[service_idx])
            off_ent_terms.append(dist.entropy())

        off_entropy = (
            torch.stack(off_ent_terms).mean()
            if off_ent_terms else torch.tensor(0.0, device=h_s.device)
        )
        value = self.critic(h_s, h_p, candidate_features, static_allowed)
        return deploy_logp + off_logp, (deploy_entropy + off_entropy) * 0.5, value.squeeze(0), {}

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
                "actions": tr["actions"].to(device),
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
                        tr['actions'],
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
