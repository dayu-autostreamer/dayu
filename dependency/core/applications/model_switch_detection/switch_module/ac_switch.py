from .base_switch import BaseSwitch
import numpy as np
import time
import threading
import random
from typing import List, Dict
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class ActorNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class ACSwitch(BaseSwitch):
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
                 *args, **kwargs):
        """
        初始化Actor-Critic切换器
        
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
        """
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.gamma = gamma
        self.batch_size = batch_size
        self.queue_high_threshold_length = queue_high_threshold_length
        self.queue_max_length = queue_max_length
        
        # 特征维度: [queue_length, cur_model_accuracy, processing_latency, target_nums, 
        #            avg_confidence, std_confidence, avg_size, std_size, brightness, contrast]
        self.context_dim = 10
        
        # 初始化Actor和Critic网络
        self.actor = ActorNetwork(self.context_dim, self.models_num)
        self.critic = CriticNetwork(self.context_dim)
        
        # 初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 初始化经验回放缓冲区
        self.memory = deque(maxlen=buffer_size)
        
        # 记录当前状态和动作
        self.current_state = None
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
        
    def _extract_context(self, stats_list) -> np.ndarray:
        """从统计信息中提取特征"""
        if not stats_list:
            return np.zeros(self.context_dim)
            
        # 使用最近的统计信息
        latest_stats = stats_list[-1]
        
        # 构建特征向量
        features = np.array([
            latest_stats.queue_length,            # 系统负载指标
            latest_stats.cur_model_accuracy,      # 模型准确率
            latest_stats.processing_latency,      # 处理延迟
            latest_stats.target_nums,             # 场景复杂度
            latest_stats.avg_confidence,          # 检测置信度
            latest_stats.std_confidence,          # 置信度std
            latest_stats.avg_size,                # 目标尺寸
            latest_stats.std_size,                # 尺寸std
            latest_stats.brightness,              # 图像亮度
            latest_stats.contrast                 # 图像对比度
        ])
        
        # 特征归一化
        # features = np.clip(features, -10, 10)  # 截断异常值
        return features
        
    def _compute_reward(self, stats_list) -> float:
        """
        计算动态奖励值
        - 当队列长度较短时，更注重准确率
        - 当队列长度较长时，更注重处理延迟
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
         
    def _switch_loop(self):
        """主循环"""
        while True:
            if time.time() - self.last_switch_time > self.decision_interval:
                self._make_decision()
                self.last_switch_time = time.time()
            time.sleep(0.1)
            
    def _select_action(self, state):
        """根据当前状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 以一定概率进行随机探索
        if random.random() < self.epsilon:
            action_idx = random.randrange(self.models_num)
        else:
            with torch.no_grad():
                action_probs = self.actor(state_tensor).cpu().numpy().flatten()
                action_idx = np.random.choice(self.models_num, p=action_probs)
        
        # 获取当前状态的统计信息
        current_stats = {
            'queue_length': state[0]
        }
        
        # 新增逻辑：如果队列超过阈值，强行以一定概率采用最低配置
        switch_prob = 0.0
        if current_stats['queue_length'] >= self.queue_high_threshold_length:
            switch_prob = (current_stats['queue_length'] - self.queue_high_threshold_length) / (self.queue_max_length - self.queue_high_threshold_length)
            # switch_prob = min(switch_prob, 0.9)  # 限制最大概率为90%
        
        if random.random() < switch_prob:
            # 强制降档到最小模型（假设模型0是最轻量的）
            action_idx = 0
            print(f"Queue length high ({current_stats['queue_length']}), forcing downgrade to model 0 with probability {switch_prob:.2f}")
            
        return action_idx
    
    def _update_networks(self):
        """训练Actor和Critic网络"""
        if len(self.memory) < self.batch_size:
            return
            
        # 随机抽样
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([t.state for t in batch])
        actions = torch.LongTensor([t.action for t in batch])
        rewards = torch.FloatTensor([t.reward for t in batch])
        next_states = torch.FloatTensor([t.next_state for t in batch])
        dones = torch.FloatTensor([t.done for t in batch])
        
        # 计算目标值
        next_values = self.critic(next_states).squeeze(1).detach()
        target_values = rewards + (1 - dones) * self.gamma * next_values
        
        # 更新Critic
        current_values = self.critic(states).squeeze(1)
        critic_loss = F.mse_loss(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 计算优势值
        advantages = target_values - current_values.detach()
        
        # 更新Actor
        action_probs = self.actor(states)
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
        # 获取当前状态
        stats = self.detector_instance.stats_manager.stats
        if not stats:
            return
            
        # 提取特征
        next_state = self._extract_context(stats)
        
        # 如果有上一个状态和动作，则计算奖励并存储经验
        if self.current_state is not None and self.current_action is not None:
            reward = self._compute_reward(stats)
            done = False  # 在连续任务中，done通常为False
            
            # 存储经验
            self.memory.append(Transition(
                self.current_state,
                self.current_action,
                reward,
                next_state,
                done
            ))
            
            # 更新网络
            self._update_networks()
        
        # 选择动作
        action = self._select_action(next_state)
        
        # 更新当前状态和动作
        self.current_state = next_state
        self.current_action = action
        
        # 如果选择的模型与当前模型不同，则切换
        current_model = stats[-1].cur_model_index
        if action != current_model:
            print(f'Actor-Critic switched model from {current_model} to {action}')
            self.switch_model(action)
            
    def switch_model(self, index: int):
        """切换到指定的模型"""
        self.detector_instance.switch_model(index)
        
    def get_detector_stats(self, *args, **kwargs):
        return super().get_detector_stats(**kwargs)