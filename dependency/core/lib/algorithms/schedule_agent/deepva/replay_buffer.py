import numpy as np
from collections import deque


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        """
        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        添加一条经验
        
        Args:
            state: (num_services+4, num_devices) numpy array
            action: (num_services,) numpy array
            reward: float
            next_state: (num_services+4, num_devices) numpy array
            done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            states: (batch_size, num_services+4, num_devices)
            actions: (batch_size, num_services)
            rewards: (batch_size,)
            next_states: (batch_size, num_services+4, num_devices)
            dones: (batch_size,)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

