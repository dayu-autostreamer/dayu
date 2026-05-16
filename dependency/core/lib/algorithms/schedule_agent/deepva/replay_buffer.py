from collections import deque

import numpy as np


class ReplayBuffer:
    """Replay buffer for DeepVA pair-state DQN transitions."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, state, deployment_mask, offloading_targets, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(deployment_mask, dtype=np.float32),
            np.asarray(offloading_targets, dtype=np.int64),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), int(batch_size), replace=False)
        states, masks, targets, rewards, next_states, dones = [], [], [], [], [], []
        for idx in indices:
            state, mask, target, reward, next_state, done = self.buffer[idx]
            states.append(state)
            masks.append(mask)
            targets.append(target)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
            np.asarray(targets, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
