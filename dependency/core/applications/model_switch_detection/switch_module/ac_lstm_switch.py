from .base_switch import BaseSwitch
import numpy as np
import time
import threading
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class LSTMActorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, lstm_layers: int = 1):
        super(LSTMActorNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]
        x = F.relu(self.fc1(lstm_out))
        return F.softmax(self.fc2(x), dim=-1)

class LSTMCriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, lstm_layers: int = 1):
        super(LSTMCriticNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]
        x = F.relu(self.fc1(lstm_out))
        return self.fc2(x)

class Transition:
    def __init__(self, state_seq, action, reward, next_state_seq, done):
        self.state_seq = state_seq  # Sequence of states
        self.action = action
        self.reward = reward
        self.next_state_seq = next_state_seq  # Sequence of next states
        self.done = done

class LSTMACSwitch(BaseSwitch):
    def __init__(self, 
                 decision_interval: int,
                 detector_instance: object,
                 gamma: float = 0.99,
                 actor_lr: float = 0.001,
                 critic_lr: float = 0.002,
                 buffer_size: int = 1000,
                 batch_size: int = 16,
                 queue_high_threshold_length: int = 20,
                 queue_max_length: int = 50,
                 seq_length: int = 5,  # Number of timesteps to consider
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 1,
                 *args, **kwargs):
        """
        初始化LSTM-Actor-Critic切换器
        
        Args:
            decision_interval: 决策间隔(秒)
            detector_instance: 检测器实例
            gamma: 折扣因子
            actor_lr: Actor学习率
            critic_lr: Critic学习率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批量大小
            queue_high_threshold_length: 队列长度高阈值，超过此值开始考虑强制降级
            queue_max_length: 队列最大长度，用于计算降级概率
            seq_length: 时序序列长度，即考虑最近几帧
            lstm_hidden_dim: LSTM隐藏层维度
            lstm_layers: LSTM层数
        """
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.gamma = gamma
        self.batch_size = batch_size
        self.queue_high_threshold_length = queue_high_threshold_length
        self.queue_max_length = queue_max_length
        self.seq_length = seq_length
        
        # 特征维度: [queue_length, cur_model_accuracy, processing_latency, target_nums, 
        #            avg_confidence, std_confidence, avg_size, std_size, brightness, contrast]
        # 新增: 模型one-hot编码
        self.feature_dim = 10 + self.models_num
        
        # 初始化LSTM-Actor和LSTM-Critic网络
        self.actor = LSTMActorNetwork(
            input_dim=self.feature_dim, 
            output_dim=self.models_num, 
            hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers
        )
        self.critic = LSTMCriticNetwork(
            input_dim=self.feature_dim,
            hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers
        )
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 初始化经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 记录当前状态序列和动作
        self.current_state_seq = None
        self.current_action = None
        
        # 训练相关
        self.train_iterations = 0
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.last_switch_time = time.time()
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()
        
    def _extract_feature(self, stats_entry) -> np.ndarray:
        """从单个统计信息条目中提取特征"""
        if stats_entry is None:
            # 返回全零向量作为补充
            base_features = np.zeros(10)
            model_onehot = np.zeros(self.models_num)
            # 假设默认使用模型0
            model_onehot[0] = 1
            return np.concatenate([base_features, model_onehot])
            
        # 构建基础特征向量
        base_features = np.array([
            stats_entry.queue_length,            # 系统负载指标
            stats_entry.cur_model_accuracy,      # 模型准确率
            stats_entry.processing_latency,      # 处理延迟
            stats_entry.target_nums,             # 场景复杂度
            stats_entry.avg_confidence,          # 检测置信度
            stats_entry.std_confidence,          # 置信度std
            stats_entry.avg_size,                # 目标尺寸
            stats_entry.std_size,                # 尺寸std
            stats_entry.brightness,              # 图像亮度
            stats_entry.contrast                 # 图像对比度
        ])
        
        # 创建模型one-hot编码
        model_onehot = np.zeros(self.models_num)
        model_onehot[stats_entry.cur_model_index] = 1
        
        # 合并特征
        features = np.concatenate([base_features, model_onehot])
        
        return features
        
    def _extract_sequence(self, stats_list) -> np.ndarray:
        """从统计信息列表中提取特征序列"""
        # 安全地获取最近的seq_length个统计信息
        seq_stats = []
        if stats_list:
            # 确保不会越界，从后往前取最近的seq_length个stats
            start_idx = max(0, len(stats_list) - self.seq_length)
            seq_stats = list(stats_list)[start_idx:]
        
        # 提取每个时间步的特征
        features_seq = []
        
        # 如果统计信息不足seq_length，用零向量填充前面的部分
        padding_count = max(0, self.seq_length - len(seq_stats))
        for _ in range(padding_count):
            features_seq.append(self._extract_feature(None))
            
        # 添加实际的统计信息特征
        for stats_entry in seq_stats:
            features_seq.append(self._extract_feature(stats_entry))
            
        return np.array(features_seq)
        
    def _compute_reward(self, stats_list) -> float:
        """
        计算动态奖励值
        使用最近的统计信息计算奖励
        """
        if not stats_list:
            return 0.0
            
        # 使用最近的统计信息
        latest_stats = stats_list[-1]
        
        # 计算队列比例和动态权重
        queue_ratio = latest_stats.queue_length / 10.0  # 使用10作为队列阈值
        w_accuracy = max(1 - queue_ratio, 0)  # 准确率权重
        w_latency = min(queue_ratio, 5)  # 延迟权重
        
        # 计算奖励
        reward = (2 * w_accuracy * (latest_stats.cur_model_accuracy/100.0 + latest_stats.avg_confidence) - 
                 2 * w_latency * latest_stats.processing_latency)
        
        print(f"""Reward calculation:
            Queue Length: {latest_stats.queue_length} (Ratio: {queue_ratio:.2f})
            Weights: accuracy={w_accuracy:.2f}, latency={w_latency:.2f}
            Accuracy: {latest_stats.cur_model_accuracy:.1f}
            Confidence: {latest_stats.avg_confidence:.2f}
            Latency: {latest_stats.processing_latency:.3f}s
            Final Reward: {reward:.3f}""")
        
        return reward
    
    def _compute_sequence_reward(self, stats_list) -> float:
        """
        计算序列奖励值
        取最近几个状态的奖励平均值
        """
        if not stats_list:
            return 0.0
            
        # 安全地获取最近的seq_length个统计信息
        recent_stats = []
        if stats_list:
            start_idx = max(0, len(stats_list) - self.seq_length)
            recent_stats = list(stats_list)[start_idx:]
        
        # 如果没有最近的统计信息，返回0
        if not recent_stats:
            return 0.0
            
        # 计算每个时间步的奖励，然后取平均值
        rewards = []
        for i in range(len(recent_stats)):
            # 为了计算第i个时间步的奖励，我们使用到第i个的所有统计信息
            temp_stats_list = recent_stats[:i+1]
            if temp_stats_list:
                rewards.append(self._compute_reward(temp_stats_list))
                
        # 返回平均奖励
        if rewards:
            return sum(rewards) / len(rewards)
        return 0.0
         
    def _switch_loop(self):
        """主循环"""
        while True:
            if time.time() - self.last_switch_time > self.decision_interval:
                self._make_decision()
                self.last_switch_time = time.time()
            time.sleep(0.1)
            
    def _select_action(self, state_seq):
        """根据当前状态序列选择动作"""
        state_tensor = torch.FloatTensor(state_seq).unsqueeze(0)  # [1, seq_len, feature_dim]
        
        # 以一定概率进行随机探索
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.models_num)
        else:
            with torch.no_grad():
                action_probs = self.actor(state_tensor).cpu().numpy().flatten()
                action_idx = np.random.choice(self.models_num, p=action_probs)
        
        # 获取最新状态的队列长度（序列中的最后一个状态）
        queue_length = state_seq[-1][0]  # 第一个特征是队列长度
        
        # 新增逻辑：如果队列超过阈值，强行以一定概率采用最低配置
        switch_prob = 0.0
        if queue_length >= self.queue_high_threshold_length:
            switch_prob = (queue_length - self.queue_high_threshold_length) / (self.queue_max_length - self.queue_high_threshold_length)
            # switch_prob = min(switch_prob, 0.9)  # 限制最大概率为90%
        
        if random.random() < switch_prob:
            # 强制降档到最小模型（假设模型0是最轻量的）
            action_idx = 0
            print(f"Queue length high ({queue_length}), forcing downgrade to model 0 with probability {switch_prob:.2f}")
            
        return action_idx
    
    def _update_networks(self):
        """训练LSTM-Actor和LSTM-Critic网络"""
        if len(self.memory) < self.batch_size:
            return
            
        # 随机抽样
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # 构建训练张量
        states_seq = torch.FloatTensor([t.state_seq for t in batch])  # [batch_size, seq_len, feature_dim]
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states_seq = torch.FloatTensor([t.next_state_seq for t in batch])
        dones = torch.FloatTensor([t.done for t in batch])
        
        # 计算目标值
        next_values = self.critic(next_states_seq).squeeze(1).detach()
        target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # 更新Critic
        current_values = self.critic(states_seq).squeeze(1)
        critic_loss = F.mse_loss(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 计算优势值
        advantages = target_values - current_values.detach()
        
        # 更新Actor
        action_probs = self.actor(states_seq)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-10)
        actor_loss = -(action_log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # 增加训练迭代计数
        self.train_iterations += 1
        
        if self.train_iterations % 100 == 0:
            print(f"Training iteration {self.train_iterations}, Actor loss: {actor_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}, Epsilon: {self.epsilon:.4f}")
    
    def _make_decision(self):
        """做出切换决策"""
        # 获取当前状态序列
        stats = self.detector_instance.stats_manager.stats
        if not stats:
            return
            
        # 提取状态序列特征
        next_state_seq = self._extract_sequence(stats)
        
        # 如果有上一个状态序列和动作，则计算奖励并存储经验
        if self.current_state_seq is not None and self.current_action is not None:
            # 计算序列奖励
            reward = self._compute_sequence_reward(stats)
            done = False  # 在连续任务中，done通常为False
            
            # 存储经验
            self.memory.append(Transition(
                self.current_state_seq,
                self.current_action,
                reward,
                next_state_seq,
                done
            ))
            
            # 更新网络
            self._update_networks()
        
        # 选择动作
        action = self._select_action(next_state_seq)
        
        # 更新当前状态序列和动作
        self.current_state_seq = next_state_seq
        self.current_action = action
        
        # 如果选择的模型与当前模型不同，则切换
        current_model = stats[-1].cur_model_index if stats else 0
        if action != current_model:
            print(f'LSTM-Actor-Critic switched model from {current_model} to {action}')
            self.switch_model(action)
            
    def switch_model(self, index: int):
        """切换到指定的模型"""
        self.detector_instance.switch_model(index)
        
    def get_detector_stats(self, *args, **kwargs):
        return super().get_detector_stats(**kwargs)