import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


class QNetwork(nn.Module):
    """Q网络：输入state，输出每个服务在每个设备上的Q值"""
    
    def __init__(self, num_services, num_devices, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.num_services = num_services
        self.num_devices = num_devices
        
        # 输入维度：(num_services + 4) * num_devices
        input_dim = (num_services + 4) * num_devices
        
        # 全连接网络
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_services * num_devices)
        
    def forward(self, state):
        """
        Args:
            state: (batch_size, num_services+4, num_devices)
        Returns:
            q_values: (batch_size, num_services, num_devices)
        """
        batch_size = state.shape[0]
        
        # Flatten state
        x = state.view(batch_size, -1)
        
        # Forward pass
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to (batch_size, num_services, num_devices)
        q_values = x.view(batch_size, self.num_services, self.num_devices)
        
        return q_values


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, num_services, num_devices, device_service_limits, 
                 learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, 
                 epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=10,
                 hidden_dim=128):
        """
        Args:
            num_services: 服务数量
            num_devices: 设备数量
            device_service_limits: 每个设备可部署的最大服务数量 (list)
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: epsilon-greedy初始值
            epsilon_end: epsilon-greedy最小值
            epsilon_decay: epsilon衰减率
            target_update_freq: 目标网络更新频率
            hidden_dim: 隐藏层维度
        """
        self.num_services = num_services
        self.num_devices = num_devices
        self.device_service_limits = device_service_limits
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建Q网络和目标网络
        self.q_network = QNetwork(num_services, num_devices, hidden_dim).to(self.device)
        self.target_network = QNetwork(num_services, num_devices, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def select_action(self, state, deterministic=False):
        """
        选择动作（服务部署方案）
        
        Args:
            state: (num_services+4, num_devices) numpy array
            deterministic: 是否使用确定性策略
            
        Returns:
            action: (num_services,) numpy array，每个元素表示该服务部署到哪个设备
        """
        # epsilon-greedy策略
        if not deterministic and np.random.rand() < self.epsilon:
            # 随机探索
            action = self._random_valid_action()
        else:
            # 利用Q网络
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)  # (1, num_services, num_devices)
                q_values = q_values.squeeze(0).cpu().numpy()  # (num_services, num_devices)
                
                # 为每个服务选择Q值最大的设备
                action = np.argmax(q_values, axis=1)  # (num_services,)
                
                # 应用mask确保满足约束
                action = self._apply_constraints(action)
        
        return action
    
    def _random_valid_action(self):
        """生成一个随机但满足约束的动作"""
        action = np.zeros(self.num_services, dtype=int)
        
        # 为每个服务随机选择设备
        for i in range(self.num_services):
            action[i] = np.random.randint(0, self.num_devices)
        
        # 应用约束
        action = self._apply_constraints(action)
        
        return action
    
    def _apply_constraints(self, action):
        """
        应用约束条件：
        1. 每个设备部署的服务数不超过限制
        2. 每个服务至少部署到一个设备
        """
        action = action.copy()
        
        # 统计每个设备当前部署的服务数
        device_counts = np.zeros(self.num_devices, dtype=int)
        for device_id in action:
            device_counts[device_id] += 1
        
        # 约束1: 如果设备部署服务数超限，按服务index顺序移除
        for device_id in range(self.num_devices):
            if device_counts[device_id] > self.device_service_limits[device_id]:
                # 找到部署在该设备上的服务
                services_on_device = [i for i in range(self.num_services) if action[i] == device_id]
                
                # 按index顺序移除多余的服务
                excess = device_counts[device_id] - self.device_service_limits[device_id]
                for i in range(excess):
                    service_to_move = services_on_device[i]
                    # 找到第一个有余量的设备
                    for target_device in range(self.num_devices):
                        if device_counts[target_device] < self.device_service_limits[target_device]:
                            action[service_to_move] = target_device
                            device_counts[device_id] -= 1
                            device_counts[target_device] += 1
                            break
        
        # 约束2: 确保每个服务都部署到某个设备（实际上已经满足，但做个检查）
        for service_id in range(self.num_services):
            if action[service_id] < 0 or action[service_id] >= self.num_devices:
                # 找到第一个有余量的设备
                for device_id in range(self.num_devices):
                    if device_counts[device_id] < self.device_service_limits[device_id]:
                        action[service_id] = device_id
                        device_counts[device_id] += 1
                        break
        
        return action
    
    def train(self, replay_buffer, batch_size=32):
        """
        使用replay buffer训练Q网络
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            
        Returns:
            loss: 当前批次的损失
        """
        if len(replay_buffer) < batch_size:
            return 0.0
        
        # 从replay buffer采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        q_values = self.q_network(states)  # (batch_size, num_services, num_devices)
        
        # 获取每个服务选择的设备对应的Q值
        # actions: (batch_size, num_services)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, self.num_services).to(self.device)
        service_indices = torch.arange(self.num_services).unsqueeze(0).expand(batch_size, -1).to(self.device)
        current_q = q_values[batch_indices, service_indices, actions]  # (batch_size, num_services)
        
        # 对每个服务的Q值求平均作为状态-动作对的Q值
        current_q = current_q.mean(dim=1)  # (batch_size,)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # (batch_size, num_services, num_devices)
            next_q_max = next_q_values.max(dim=2)[0]  # (batch_size, num_services)
            next_q = next_q_max.mean(dim=1)  # (batch_size,)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = self.loss_fn(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, save_dir, episode):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'dqn_model_episode_{episode}.pth')
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
        }, save_path)
        print(f'Model saved to {save_path}')
    
    def load(self, save_dir, episode):
        """加载模型"""
        load_path = os.path.join(save_dir, f'dqn_model_episode_{episode}.pth')
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.update_counter = checkpoint['update_counter']
            print(f'Model loaded from {load_path}')
        else:
            print(f'Model file not found: {load_path}')

