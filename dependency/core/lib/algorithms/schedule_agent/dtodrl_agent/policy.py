import copy
import math
import os
import tempfile

import torch
from torch import nn

__all__ = ("DTODRLPolicy", "GraphAttentionActorCritic")


class GraphAttentionActorCritic(nn.Module):
    """Small graph-attention actor with one categorical head per service."""

    def __init__(self, node_feature_dim, candidate_feature_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.node_encoder = nn.Linear(node_feature_dim, self.hidden_dim)
        self.query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.node_update = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_feature_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        node_features,
        adjacency,
        candidate_features,
        candidate_mask,
    ):
        node_state = torch.tanh(self.node_encoder(node_features))
        query = self.query(node_state)
        key = self.key(node_state)
        score = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
            self.hidden_dim
        )
        node_count = node_features.shape[1]
        self_edges = torch.eye(
            node_count,
            dtype=torch.bool,
            device=node_features.device,
        ).unsqueeze(0)
        attention_mask = adjacency.bool() | self_edges
        score = score.masked_fill(~attention_mask, -1e9)
        attention = torch.softmax(score, dim=-1)
        context = torch.matmul(attention, self.value(node_state))
        node_state = torch.tanh(
            self.node_update(torch.cat((node_state, context), dim=-1))
        )

        candidate_state = self.candidate_encoder(candidate_features)
        expanded_node = node_state.unsqueeze(2).expand(
            -1,
            -1,
            candidate_state.shape[2],
            -1,
        )
        logits = self.actor(
            torch.cat((expanded_node, candidate_state), dim=-1)
        ).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), -1e9)
        graph_state = node_state.mean(dim=1)
        state_value = self.critic(graph_state).squeeze(-1)
        return logits, state_value


class DTODRLPolicy:
    CHECKPOINT_VERSION = 1

    @staticmethod
    def _normalize_weight_signature(signature):
        """Return only fields that define DTODRL weight compatibility.

        Legacy checkpoints included the video configuration even though it
        does not change the GAT/PPO tensor schema or action space.  Keep
        accepting those checkpoints while retaining strict checks for every
        model-relevant context field.
        """
        normalized = copy.deepcopy(signature)
        if isinstance(normalized, dict):
            normalized.pop("configuration", None)
        return normalized

    def __init__(
        self,
        signature,
        checkpoint_path,
        mode,
        hidden_dim=64,
        learning_rate=3e-4,
        ppo_clip=0.2,
        entropy_weight=0.01,
        ppo_epochs=4,
        random_seed=0,
        load_checkpoint=False,
    ):
        # DTODRL evaluates a very small graph (single-digit services and only a
        # few candidates per service).  PyTorch's default CPU thread pool costs
        # substantially more than the matrix work itself on the cloud worker,
        # and can make an otherwise lightweight inference miss the source
        # cadence.  A single intra-op thread preserves the exact model while
        # avoiding that implementation artefact.  Inter-op threads can only be
        # configured before PyTorch starts parallel work, so tolerate a prior
        # initialization in a reused process.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        self.signature = self._normalize_weight_signature(signature)
        self.checkpoint_path = str(checkpoint_path)
        self.mode = str(mode)
        self.hidden_dim = max(8, int(hidden_dim))
        self.ppo_clip = min(0.9, max(0.01, float(ppo_clip)))
        self.entropy_weight = max(0.0, float(entropy_weight))
        self.ppo_epochs = max(1, int(ppo_epochs))
        self.device = torch.device("cpu")
        torch.manual_seed(int(random_seed))
        self.model = GraphAttentionActorCritic(
            node_feature_dim=5,
            candidate_feature_dim=5,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(learning_rate),
        )
        self.update_count = 0

        should_load = self.mode == "inference" or bool(load_checkpoint)
        if should_load:
            self.load()

    def _batch_tensors(self, states):
        return (
            torch.tensor(
                [state["node_features"] for state in states],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [state["adjacency"] for state in states],
                dtype=torch.bool,
                device=self.device,
            ),
            torch.tensor(
                [state["candidate_features"] for state in states],
                dtype=torch.float32,
                device=self.device,
            ),
            torch.tensor(
                [state["candidate_mask"] for state in states],
                dtype=torch.bool,
                device=self.device,
            ),
        )

    def _distribution(self, states):
        tensors = self._batch_tensors(states)
        logits, values = self.model(*tensors)
        return torch.distributions.Categorical(logits=logits), values

    def select(self, state, deterministic):
        self.model.eval()
        with torch.no_grad():
            if deterministic:
                tensors = self._batch_tensors([state])
                logits, value = self.model(*tensors)
                actions = logits.argmax(dim=-1)
                # Inference never consumes the old-policy probability.  Avoid
                # building a Categorical object solely for a discarded value.
                log_probability = logits.new_zeros((logits.shape[0],))
            else:
                distribution, value = self._distribution([state])
                actions = distribution.sample()
                log_probability = distribution.log_prob(actions).sum(dim=-1)
        return (
            [int(item) for item in actions[0].tolist()],
            float(log_probability.item()),
            float(value.item()),
        )

    def update(self, transitions):
        if not transitions:
            return {}
        states = [transition["state"] for transition in transitions]
        actions = torch.tensor(
            [transition["actions"] for transition in transitions],
            dtype=torch.long,
            device=self.device,
        )
        old_log_probability = torch.tensor(
            [transition["old_log_probability"] for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        old_value = torch.tensor(
            [transition["old_value"] for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.tensor(
            [transition["reward"] for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        advantage = returns - old_value
        if len(transitions) > 1 and float(advantage.std(unbiased=False)) > 1e-6:
            advantage = (
                advantage - advantage.mean()
            ) / (advantage.std(unbiased=False) + 1e-6)

        self.model.train()
        metrics = {}
        for _ in range(self.ppo_epochs):
            distribution, value = self._distribution(states)
            log_probability = distribution.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(log_probability - old_log_probability)
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio,
                1.0 - self.ppo_clip,
                1.0 + self.ppo_clip,
            ) * advantage
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.square(value - returns).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - self.entropy_weight * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            metrics = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": float(entropy.detach()),
                "reward": float(returns.mean().detach()),
            }
        self.update_count += 1
        return metrics

    def save(self):
        directory = os.path.dirname(self.checkpoint_path) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {
            "version": self.CHECKPOINT_VERSION,
            "signature": copy.deepcopy(self.signature),
            "hidden_dim": self.hidden_dim,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }
        handle = tempfile.NamedTemporaryFile(
            prefix=".dtodrl-",
            suffix=".tmp",
            dir=directory,
            delete=False,
        )
        temporary_path = handle.name
        handle.close()
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, self.checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load(self):
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"DTODRL checkpoint does not exist: {self.checkpoint_path}"
            )
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        if payload.get("version") != self.CHECKPOINT_VERSION:
            raise ValueError("DTODRL checkpoint version is incompatible")
        checkpoint_signature = self._normalize_weight_signature(
            payload.get("signature")
        )
        if checkpoint_signature != self.signature:
            raise ValueError(
                "DTODRL checkpoint does not match the active scheduling context"
            )
        if int(payload.get("hidden_dim") or 0) != self.hidden_dim:
            raise ValueError("DTODRL checkpoint hidden_dim does not match configuration")
        self.model.load_state_dict(payload["model"])
        if self.mode == "train" and payload.get("optimizer"):
            self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = max(0, int(payload.get("update_count") or 0))
        self.model.eval()
