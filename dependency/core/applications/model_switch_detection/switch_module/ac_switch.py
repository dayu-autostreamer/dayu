from .base_switch import BaseSwitch
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple, deque
import os

# 定义经验元组
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'log_prob', 'value'))

class LSTMActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMActorCritic, self).__init__()
        
        # 分离的LSTM层，一个用于Actor，一个用于Critic
        self.actor_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.critic_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Actor网络（策略）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic网络（价值函数）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 隐藏状态初始化
        self.actor_hidden = None
        self.critic_hidden = None
        
    def reset_hidden(self, batch_size=1):
        """重置LSTM的隐藏状态"""
        device = next(self.parameters()).device
        self.actor_hidden = (
            torch.zeros(1, batch_size, self.actor_lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.actor_lstm.hidden_size).to(device)
        )
        self.critic_hidden = (
            torch.zeros(1, batch_size, self.critic_lstm.hidden_size).to(device),
            torch.zeros(1, batch_size, self.critic_lstm.hidden_size).to(device)
        )
        
    def forward(self, x, reset_hidden=False):
        """前向传播"""
        batch_size = x.size(0)
        
        # 重置隐藏状态（如果需要）
        if reset_hidden or self.actor_hidden is None or self.critic_hidden is None:
            self.reset_hidden(batch_size)
        
        # 检查隐藏状态batch_size是否与输入匹配
        if self.actor_hidden[0].size(1) != batch_size:
            self.reset_hidden(batch_size)
            
        # Actor LSTM前向传播
        actor_lstm_out, self.actor_hidden = self.actor_lstm(x, self.actor_hidden)
        actor_features = actor_lstm_out[:, -1, :]  # 取LSTM最后一个时间步的输出
        
        # Critic LSTM前向传播
        critic_lstm_out, self.critic_hidden = self.critic_lstm(x, self.critic_hidden)
        critic_features = critic_lstm_out[:, -1, :]  # 取LSTM最后一个时间步的输出
        
        # 计算动作概率和状态价值
        action_probs = self.actor(actor_features)
        state_value = self.critic(critic_features)
        
        return action_probs, state_value
    
    def act(self, state, exploration_rate=0.0):
        """根据当前状态选择动作"""
        # 确保隐藏状态与当前输入batch匹配
        self.reset_hidden(batch_size=state.size(0))
        
        # 前向传播获取动作概率和状态价值
        action_probs, state_value = self(state)
        
        print(f"Action probs: {action_probs}")
        
        # 应用探索 - 有时选择随机动作
        if np.random.random() < exploration_rate:
            action_idx = torch.randint(0, action_probs.size(-1), (1,)).item()
            log_prob = torch.log(action_probs[0, action_idx])
            return action_idx, log_prob, state_value
        
        # 创建一个分类分布并从中采样
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # 返回动作索引、对数概率和状态价值
        return action.item(), dist.log_prob(action), state_value


class ACSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 queue_low_threshold_length: int = 10,
                 state_history_length: int = 5,
                 hidden_dim: int = 32,
                 *args, **kwargs):
        """
        LSTM Actor-Critic模型切换器
        
        Args:
            decision_interval: 模型切换决策的时间间隔(秒)
            detector_instance: 检测器实例
            queue_low_threshold_length: 队列长度阈值，用于奖励计算
            state_history_length: LSTM的时间步长
            hidden_dim: LSTM隐藏层的维度
        """
        # 基本属性初始化
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.queue_low_threshold_length = queue_low_threshold_length
        self.state_history_length = state_history_length
        
        # 队列阈值参数
        self.queue_high_threshold_length = 2 * self.queue_low_threshold_length
        
        # 当前使用的模型
        self.current_model_index = None
        
        # 特征数据相关
        self.input_dim = 11  # StatsEntry中除去timestamp的特征数量
        
        # 决策与学习相关参数
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 探索增强参数
        self.exploration_rate = 0.9  # 初始探索率
        self.min_exploration_rate = 0.1  # 最小探索率
        self.exploration_decay = 0.999  # 探索率衰减因子
        
        # 初始化LSTM Actor-Critic网络
        self.network = LSTMActorCritic(
            input_dim=self.input_dim, 
            hidden_dim=self.hidden_dim, 
            output_dim=self.models_num
        ).to(self.device)
        
        # 使用分离的优化器
        self.actor_optimizer = optim.Adam([p for n, p in self.network.named_parameters() 
                                          if 'actor' in n or 'actor_lstm' in n], lr=0.005)
        self.critic_optimizer = optim.Adam([p for n, p in self.network.named_parameters() 
                                           if 'critic' in n or 'critic_lstm' in n], lr=0.001)
        
        # 跟踪最后选择的动作和状态
        self.previous_action = None
        self.previous_state = None
        self.previous_log_prob = None
        self.previous_value = None
        self.last_switch_time = time.time()
        
        # 经验回放相关
        self.replay_buffer = deque(maxlen=100)
        self.min_samples_before_update = 8
        self.batch_size = 8
        self.update_frequency = 1
        self.update_counter = 0
        
        # Actor-Critic超参数
        self.gamma = 0.5  # 折扣因子
        self.entropy_beta = 0.1  # 熵正则化系数
        self.critic_loss_coef = 0.5  # 价值函数损失系数
        
        # 统计信息
        self.episodes = 0
        self.training_steps = 0
        
        # 模型保存路径
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, "lstm_ac_switch_model.pt")
        
        # 尝试加载预训练模型
        self.load_model(self.model_path)
        
        # 启动决策线程
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()
        
        print(f"LSTM Actor-Critic Switch initialized with {self.models_num} models, device: {self.device}")

    def _switch_loop(self):
        """模型切换决策的主循环"""
        while True:
            try:
                current_time = time.time()
                
                # 如果达到了决策间隔时间
                if current_time - self.last_switch_time > self.decision_interval:
                    # 获取间隔统计数据，用于LSTM输入
                    stats_list = self.get_detector_interval_stats(nums=self.state_history_length, interval=1.0)
                    
                    # 获取当前统计数据以计算奖励
                    current_stats = self.get_detector_stats()
                    
                    # 检查是否需要紧急响应
                    emergency_mode = False
                    if current_stats and current_stats[0]:
                        stats_dict = self.stats_entry_to_dict(current_stats[0])
                        if stats_dict.get('queue_length', 0) >= self.queue_high_threshold_length:
                            emergency_mode = True
                            # 选择最轻量的模型（索引0）
                            self.handle_emergency()
                            time.sleep(self.decision_interval)
                            continue
                    
                    if stats_list and len(stats_list) > 0:
                        # 预处理统计数据序列用于网络输入
                        current_state = self.preprocess_stats(stats_list)
                        
                        # 如果有上一个动作和状态，计算奖励并进行学习
                        if self.previous_action is not None and self.previous_state is not None and current_stats and current_stats[0]:
                            # 计算奖励
                            stats_dict = self.stats_entry_to_dict(current_stats[0])
                            reward = self.calculate_reward(stats_dict)
                            
                            # 创建经验元组
                            transition = Transition(
                                state=self.previous_state,
                                action=self.previous_action,
                                next_state=current_state,
                                reward=reward,
                                log_prob=self.previous_log_prob,
                                value=self.previous_value
                            )
                            
                            # 添加到经验回放缓冲区
                            self.replay_buffer.append(transition)
                            
                            # 计数器递增，并在达到更新频率时进行网络更新
                            self.update_counter += 1
                            if self.update_counter >= self.update_frequency and len(self.replay_buffer) >= self.min_samples_before_update:
                                self.update_network(self.batch_size)
                                self.update_counter = 0
                        
                        # 选择新动作
                        if current_state is not None:
                            action_idx, log_prob, value = self.select_action(current_state)
                            
                            # 执行动作（切换模型）
                            if self.current_model_index is None or action_idx != self.current_model_index:
                                self.switch_model(action_idx)
                                self.current_model_index = action_idx
                                print(f'LSTM Actor-Critic switched model to {action_idx}')
                            else:
                                print(f'Keeping current model: {action_idx}')
                            
                            # 保存当前状态和动作以便下次使用
                            self.previous_state = current_state
                            self.previous_action = action_idx
                            self.previous_log_prob = log_prob
                            self.previous_value = value
                            self.last_switch_time = current_time
                            
                            # 增加episode计数
                            self.episodes += 1
                            
                            # 定期保存模型
                            if self.episodes % 50 == 0:
                                self.save_model(self.model_path)
                        else:
                            print("无效的状态，跳过此次决策")
                    else:
                        print("No valid state sequence available")
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"决策循环中出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # 出错时等待较长时间再重试

    def handle_emergency(self):
        """处理紧急情况，如队列长度超过最大阈值"""
        # 紧急情况下选择最轻量的模型（索引0）
        selected_model = 0
        self.switch_model(selected_model)
        self.current_model_index = selected_model
        print(f"EMERGENCY: Queue length exceeded threshold. Selecting lightest model: {selected_model}")
        return selected_model

    def stats_entry_to_dict(self, stats):
        """将StatsEntry转换为字典"""
        if stats is None:
            return {}
        
        return {
            'timestamp': stats.timestamp,
            'queue_length': stats.queue_length,
            'cur_model_index': stats.cur_model_index,
            'cur_model_accuracy': stats.cur_model_accuracy,
            'processing_latency': stats.processing_latency,
            'target_nums': stats.target_nums,
            'avg_confidence': stats.avg_confidence,
            'std_confidence': stats.std_confidence,
            'avg_size': stats.avg_size,
            'std_size': stats.std_size,
            'brightness': stats.brightness,
            'contrast': stats.contrast
        }
    
    def preprocess_stats(self, stats_list):
        """预处理统计数据用于网络输入"""
        if not stats_list:
            return None
            
        # 提取特征
        features = []
        for stats in stats_list:
            if stats is None:
                continue
                
            stats_dict = self.stats_entry_to_dict(stats)
            
            # 从StatsEntry中提取特征（除去timestamp）
            feature = [
                float(stats_dict.get('queue_length', 0)) / self.queue_high_threshold_length,  # 队列长度归一化
                float(stats_dict.get('cur_model_index', 0)),  # 当前模型索引
                float(stats_dict.get('cur_model_accuracy', 0)) / 100.0,  # 当前模型准确率归一化
                float(stats_dict.get('processing_latency', 0)),  # 处理延迟
                float(stats_dict.get('target_nums', 0)) / 10.0,  # 目标数量归一化，假设平均不超过10
                float(stats_dict.get('avg_confidence', 0)),  # 平均置信度（已在0-1范围）
                float(stats_dict.get('std_confidence', 0)),  # 置信度标准差
                float(stats_dict.get('avg_size', 0)),  # 平均尺寸
                float(stats_dict.get('std_size', 0)),  # 尺寸标准差
                float(stats_dict.get('brightness', 0)) / 255.0,  # 亮度归一化
                float(stats_dict.get('contrast', 0)) / 255.0  # 对比度归一化
            ]
            features.append(feature)
            
        # 如果没有有效特征，返回None
        if len(features) == 0:
            print("警告: 没有有效特征")
            return None
        
        # 将特征转换为numpy数组 [sequence_length, features]
        features = np.array(features, dtype=np.float32)
        
        # 确保形状正确 (batch_size, sequence_length, features)
        # 对于LSTM，我们需要 [batch_size=1, sequence_length, features]
        features = np.expand_dims(features, axis=0)
            
        # 转换为PyTorch张量
        state_tensor = torch.FloatTensor(features).to(self.device)
        
        return state_tensor

    def calculate_reward(self, stats):
        """计算奖励值"""
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
        # 权重计算
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        # 奖励计算
        reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                 w2 * (stats['processing_latency'])
                 
        print(f"计算奖励: {reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")
        return reward

    def select_action(self, state):
        """选择动作（模型）"""
        with torch.no_grad():
            action_idx, log_prob, state_value = self.network.act(state, exploration_rate=self.exploration_rate)
        
        # 衰减探索率
        self.exploration_rate = max(self.exploration_rate * self.exploration_decay, 
                                    self.min_exploration_rate)
        
        print(f"选择动作: {action_idx}, 探索率: {self.exploration_rate:.4f}")
        return action_idx, log_prob, state_value

    def sample_from_replay_buffer(self, batch_size):
        """从回放缓冲区采样经验"""
        if len(self.replay_buffer) < batch_size:
            return list(self.replay_buffer)
        
        # 随机采样不放回
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]

    def update_network(self, batch_size=32):
        """使用经验回放缓冲区中的样本批量更新LSTM Actor-Critic网络"""
        try:
            # 如果缓冲区中的样本不足，则跳过更新
            if len(self.replay_buffer) < batch_size:
                print(f"Skip update: Not enough samples in replay buffer ({len(self.replay_buffer)}/{batch_size})")
                return
                
            # 从缓冲区中随机采样一批经验
            transitions = self.sample_from_replay_buffer(batch_size)
            
            # 验证样本有效性并过滤无效样本
            valid_transitions = []
            for t in transitions:
                if t.log_prob is not None and isinstance(t.log_prob, torch.Tensor):
                    valid_transitions.append(t)
            
            if not valid_transitions:
                print("No valid transitions to update network")
                return
                
            # 提取批数据
            batch_size = len(valid_transitions)
            
            # 确保所有状态有一致的序列长度
            seq_lengths = set()
            for t in valid_transitions:
                if t.state is not None and len(t.state.shape) >= 2:
                    seq_lengths.add(t.state.shape[1])
            
            if len(seq_lengths) > 1:
                print(f"Warning: Inconsistent sequence lengths {seq_lengths}, skipping update")
                return
            
            states = torch.cat([t.state for t in valid_transitions], dim=0)
            actions = torch.tensor([t.action for t in valid_transitions], dtype=torch.long).to(self.device)
            rewards = torch.tensor([t.reward for t in valid_transitions], dtype=torch.float32).view(-1, 1).to(self.device)
            
            # 处理next_states (有些可能为None)
            next_states_list = []
            masks = []
            for t in valid_transitions:
                if t.next_state is not None:
                    next_states_list.append(t.next_state)
                    masks.append(1.0)
                else:
                    # 如果next_state为None，使用零张量代替
                    next_states_list.append(torch.zeros_like(t.state))
                    masks.append(0.0)
            
            next_states = torch.cat(next_states_list, dim=0)
            masks = torch.tensor(masks, dtype=torch.float32).view(-1, 1).to(self.device)
            
            # 更新Critic网络
            self.critic_optimizer.zero_grad()
            
            # 计算当前状态的价值
            self.network.reset_hidden(batch_size)
            _, current_values = self.network(states)
            
            # 计算下一个状态的价值
            self.network.reset_hidden(batch_size)
            _, next_values = self.network(next_states)
            next_values = next_values.detach()  # 停止梯度
            
            # 计算目标值和critic损失
            target_values = rewards + self.gamma * next_values * masks
            critic_loss = F.mse_loss(current_values, target_values)
            
            # 更新critic
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.critic_optimizer.step()
            
            # 更新Actor网络
            self.actor_optimizer.zero_grad()
            
            # 重新计算策略
            self.network.reset_hidden(batch_size)
            action_probs, values = self.network(states)
            
            # 计算优势函数，使用当前网络的值
            advantages = (target_values - values).detach()
            
            print(f"Advantages stats: mean={advantages.mean().item():.6f}, std={advantages.std().item():.6f}, min={advantages.min().item():.6f}, max={advantages.max().item():.6f}")
            
            # 创建一个分类分布
            dist = Categorical(action_probs)
            
            # 计算所选动作的对数概率
            action_log_probs = dist.log_prob(actions)
            
            # 在更新前检查log_probs和advantages中是否有NaN或Inf
            if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
                print("Warning: NaN or Inf detected in log_probs")
                
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print("Warning: NaN or Inf detected in advantages")
            
            # 计算actor损失
            actor_loss = -(action_log_probs * advantages.squeeze()).mean()
            
            # 添加熵正则化
            entropy_loss = dist.entropy().mean()
            actor_loss -= self.entropy_beta * entropy_loss
            
            # === 针对极小概率的特殊损失 ===
            # 设定最小可接受概率阈值
            min_prob_threshold = 5e-2  # 5%
            
            # 检查每个批次样本的动作概率分布
            min_probs_per_batch = action_probs.min(dim=1)[0]  # 每个批次中的最小概率
            
            # 计算有多少批次样本中存在极小概率
            extreme_prob_batches = (min_probs_per_batch < min_prob_threshold).float()
            num_extreme_batches = extreme_prob_batches.sum().item()
            
            # 只有当存在极小概率时才添加惩罚
            if num_extreme_batches > 0:
                # 检查每个动作的最小概率
                min_action_probs = action_probs.min(dim=0)[0]  # 每个动作的最小概率
                
                # 找出小于阈值的动作
                small_prob_actions = (min_action_probs < min_prob_threshold).float()
                num_small_actions = small_prob_actions.sum().item()
                
                if num_small_actions > 0:
                    # 计算对数惩罚：-log(p)当p接近0时会变得非常大
                    prob_penalty = -torch.log(min_action_probs + 1e-10) * small_prob_actions
                    
                    # 只对小于阈值的动作应用惩罚
                    extreme_prob_penalty = prob_penalty.sum() / max(1.0, num_small_actions)
                    
                    # 动态调整惩罚系数，概率越小惩罚越大
                    penalty_coef = 0.1 * (1.0 - min_action_probs.min().item() / min_prob_threshold)
                    penalty_coef = min(penalty_coef, 0.5)  # 限制最大惩罚系数
                    
                    # 添加到actor损失
                    actor_loss = actor_loss + penalty_coef * extreme_prob_penalty
                    
                    print(f"Applied probability penalty: {penalty_coef:.4f} * {extreme_prob_penalty.item():.4f} for {num_small_actions} actions below threshold")
            
            # 更新actor
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # 重置LSTM隐藏状态，准备下一次预测
            self.network.reset_hidden(batch_size=1)
            
            # 更新统计信息
            self.training_steps += 1
            
            print(f"Network updated - Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}")
        except Exception as e:
            print(f"Error during network update: {e}")
            import traceback
            traceback.print_exc()

    def save_model(self, path):
        """保存模型到文件"""
        try:
            torch.save({
                'model_state_dict': self.network.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'episodes': self.episodes,
                'training_steps': self.training_steps
            }, path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, path):
        """从文件加载模型"""
        try:
            if not os.path.exists(path):
                print(f"Model file not found: {path}")
                return False
                
            checkpoint = torch.load(path, map_location=self.device)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            
            if 'actor_optimizer_state_dict' in checkpoint and 'critic_optimizer_state_dict' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
                
            self.exploration_rate = checkpoint.get('exploration_rate', self.exploration_rate)
            self.episodes = checkpoint.get('episodes', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def switch_model(self, index: int):
        """切换到指定索引的模型"""
        self.detector_instance.switch_model(index)

    def get_detector_stats(self):
        """获取检测器的最新统计信息"""
        stats = self.detector_instance.stats_manager.get_latest_stats()
        return stats
    
    def get_detector_interval_stats(self, nums: int = 5, interval: float = 1.0):
        """按间隔获取检测器的统计信息"""
        stats = self.detector_instance.stats_manager.get_interval_stats(nums, interval)
        return stats