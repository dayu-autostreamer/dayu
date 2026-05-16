import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PairQNetwork(nn.Module):
    """Plain MLP Q network over service-device pair features."""

    def __init__(self, pair_feature_dim, hidden_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.deploy_head = nn.Linear(hidden_dim, 1)
        self.offload_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Args:
            state: [batch, service, device, feature]
        Returns:
            deploy_q, offload_q: [batch, service, device]
        """
        batch_size, num_services, num_devices, feature_dim = state.shape
        x = state.reshape(batch_size * num_services * num_devices, feature_dim)
        h = self.body(x)
        deploy_q = self.deploy_head(h).view(batch_size, num_services, num_devices)
        offload_q = self.offload_head(h).view(batch_size, num_services, num_devices)
        return deploy_q, offload_q


class DQNAgent:
    """A compact DQN baseline with pair-wise deployment and offloading heads."""

    def __init__(
        self,
        num_services,
        num_devices,
        pair_feature_dim,
        device_service_limits,
        learning_rate=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.985,
        target_update_freq=10,
        hidden_dim=128,
        min_replicas_per_service=1,
        max_replicas_per_service=2,
        replica_score_threshold=0.0,
        deploy_q_weight=0.5,
        offload_q_weight=1.0,
        device=None,
    ):
        self.num_services = int(num_services)
        self.num_devices = int(num_devices)
        self.pair_feature_dim = int(pair_feature_dim)
        self.device_service_limits = np.asarray(device_service_limits, dtype=np.int64)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.target_update_freq = int(target_update_freq)
        self.min_replicas_per_service = max(1, int(min_replicas_per_service))
        self.max_replicas_per_service = max(self.min_replicas_per_service, int(max_replicas_per_service))
        self.replica_score_threshold = float(replica_score_threshold)
        self.deploy_q_weight = float(deploy_q_weight)
        self.offload_q_weight = float(offload_q_weight)
        self.update_counter = 0

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q_network = PairQNetwork(self.pair_feature_dim, hidden_dim).to(self.device)
        self.target_network = PairQNetwork(self.pair_feature_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, deterministic=False):
        if not deterministic and np.random.rand() < self.epsilon:
            return self._random_action()

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            deploy_q, offload_q = self.q_network(state_tensor)
            deploy_q = deploy_q.squeeze(0).detach().cpu().numpy()
            offload_q = offload_q.squeeze(0).detach().cpu().numpy()
        return self._decode_action(deploy_q, offload_q)

    def _random_action(self):
        mask = np.zeros((self.num_services, self.num_devices), dtype=np.float32)
        device_counts = np.zeros(self.num_devices, dtype=np.int64)

        service_order = np.random.permutation(self.num_services)
        for service_idx in service_order:
            replica_count = np.random.randint(
                self.min_replicas_per_service,
                self.max_replicas_per_service + 1,
            )
            candidates = list(np.random.permutation(self.num_devices))
            placed = 0
            for device_idx in candidates:
                if placed >= replica_count:
                    break
                if device_counts[device_idx] >= self.device_service_limits[device_idx]:
                    continue
                mask[service_idx, device_idx] = 1.0
                device_counts[device_idx] += 1
                placed += 1
            if placed == 0:
                device_idx = int(np.argmin(device_counts / np.maximum(self.device_service_limits, 1)))
                mask[service_idx, device_idx] = 1.0
                device_counts[device_idx] += 1

        offloading_targets = np.zeros(self.num_services, dtype=np.int64)
        for service_idx in range(self.num_services):
            candidates = np.flatnonzero(mask[service_idx] > 0)
            offloading_targets[service_idx] = int(np.random.choice(candidates)) if candidates.size else 0
        return {
            "deployment_mask": mask,
            "offloading_targets": offloading_targets,
            "deploy_q": None,
            "offload_q": None,
        }

    def _decode_action(self, deploy_q, offload_q):
        mask = self._decode_deployment(deploy_q)
        offloading_targets = np.zeros(self.num_services, dtype=np.int64)
        for service_idx in range(self.num_services):
            candidates = np.flatnonzero(mask[service_idx] > 0)
            if candidates.size == 0:
                candidates = np.arange(self.num_devices)
            best_local = int(candidates[np.argmax(offload_q[service_idx, candidates])])
            offloading_targets[service_idx] = best_local
        return {
            "deployment_mask": mask,
            "offloading_targets": offloading_targets,
            "deploy_q": deploy_q,
            "offload_q": offload_q,
        }

    def _decode_deployment(self, deploy_q):
        mask = np.zeros((self.num_services, self.num_devices), dtype=np.float32)
        device_counts = np.zeros(self.num_devices, dtype=np.int64)

        best_scores = deploy_q.max(axis=1)
        service_order = np.argsort(-best_scores)
        for service_idx in service_order:
            ranked_devices = np.argsort(-deploy_q[service_idx])
            placed = 0
            for device_idx in ranked_devices:
                if placed >= self.min_replicas_per_service:
                    break
                if device_counts[device_idx] >= self.device_service_limits[device_idx]:
                    continue
                mask[service_idx, device_idx] = 1.0
                device_counts[device_idx] += 1
                placed += 1
            if placed == 0:
                device_idx = self._least_loaded_device(device_counts)
                mask[service_idx, device_idx] = 1.0
                device_counts[device_idx] += 1

        extra_candidates = []
        for service_idx in range(self.num_services):
            center = float(np.mean(deploy_q[service_idx]))
            current_replicas = int(mask[service_idx].sum())
            if current_replicas >= self.max_replicas_per_service:
                continue
            for device_idx in range(self.num_devices):
                if mask[service_idx, device_idx] > 0:
                    continue
                margin = float(deploy_q[service_idx, device_idx] - center)
                if margin >= self.replica_score_threshold:
                    extra_candidates.append((margin, service_idx, device_idx))

        for _, service_idx, device_idx in sorted(extra_candidates, reverse=True):
            if int(mask[service_idx].sum()) >= self.max_replicas_per_service:
                continue
            if device_counts[device_idx] >= self.device_service_limits[device_idx]:
                continue
            mask[service_idx, device_idx] = 1.0
            device_counts[device_idx] += 1
        return mask

    def _least_loaded_device(self, device_counts):
        capacity = np.maximum(self.device_service_limits, 1)
        load_ratio = device_counts / capacity
        return int(np.argmin(load_ratio))

    def train(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return {"loss": 0.0, "epsilon": self.epsilon}

        states, deployment_masks, offloading_targets, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        deployment_masks = torch.as_tensor(deployment_masks, dtype=torch.float32, device=self.device)
        offloading_targets = torch.as_tensor(offloading_targets, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        deploy_q, offload_q = self.q_network(states)
        current_q = self._joint_q_value(deploy_q, offload_q, deployment_masks, offloading_targets)

        with torch.no_grad():
            next_deploy_q, next_offload_q = self.target_network(next_states)
            next_masks = []
            next_targets = []
            for batch_idx in range(next_states.shape[0]):
                decoded = self._decode_action(
                    next_deploy_q[batch_idx].detach().cpu().numpy(),
                    next_offload_q[batch_idx].detach().cpu().numpy(),
                )
                next_masks.append(decoded["deployment_mask"])
                next_targets.append(decoded["offloading_targets"])
            next_masks = torch.as_tensor(np.asarray(next_masks), dtype=torch.float32, device=self.device)
            next_targets = torch.as_tensor(np.asarray(next_targets), dtype=torch.long, device=self.device)
            next_q = self._joint_q_value(next_deploy_q, next_offload_q, next_masks, next_targets)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return {
            "loss": float(loss.item()),
            "epsilon": float(self.epsilon),
            "q_mean": float(current_q.detach().mean().item()),
            "target_q_mean": float(target_q.detach().mean().item()),
        }

    def _joint_q_value(self, deploy_q, offload_q, deployment_masks, offloading_targets):
        selected_deploy_sum = (deploy_q * deployment_masks).sum(dim=(1, 2))
        selected_deploy_count = deployment_masks.sum(dim=(1, 2)).clamp_min(1.0)
        deploy_value = selected_deploy_sum / selected_deploy_count

        batch_size = offload_q.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        service_indices = torch.arange(self.num_services, device=self.device).unsqueeze(0)
        offload_value = offload_q[batch_indices, service_indices, offloading_targets].mean(dim=1)
        return self.deploy_q_weight * deploy_value + self.offload_q_weight * offload_value

    def save(self, save_dir, episode):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"dqn_model_episode_{episode}.pth")
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "update_counter": self.update_counter,
        }, save_path)

    def load(self, save_dir, episode):
        load_path = os.path.join(save_dir, f"dqn_model_episode_{episode}.pth")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"DeepVA model file not found: {load_path}")
        checkpoint = torch.load(load_path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.update_counter = int(checkpoint.get("update_counter", self.update_counter))
