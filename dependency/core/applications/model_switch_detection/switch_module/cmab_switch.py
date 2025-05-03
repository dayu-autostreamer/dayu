from .base_switch import BaseSwitch
import time
import threading
import numpy as np
from scipy import linalg


class CMABSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 queue_low_threshold_length: int = 10,
                 lambda_reg: float = 1.0,
                 noise_variance: float = 0.1,
                 exploration_rate: float = 0.1,
                 *args, **kwargs):
        """
        线性上下文汤普森采样(CMAB)模型切换器
        
        Args:
            decision_interval: 模型切换决策的时间间隔(秒)
            detector_instance: 检测器实例
            queue_low_threshold_length: 队列长度阈值，用于奖励计算
            context_dimension: 特征向量的维度
            lambda_reg: 正则化参数
            noise_variance: 观测噪声方差
            exploration_rate: 随机探索的概率
        """
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.queue_low_threshold_length = queue_low_threshold_length
        
        # 特征维度和贝叶斯线性回归参数
        self.context_dimension = 10  # 固定为10个特征（9个实际特征 + 1个偏置项）
        self.lambda_reg = lambda_reg
        self.noise_variance = noise_variance
        
        # 探索参数
        self.exploration_rate = exploration_rate
        self.exploration_counter = 0
        self.force_exploration_count = 15  # 每15次决策强制探索一次
        
        # 每个模型的参数
        self.models = {}
        self.init_thompson_sampling_models()
        
        # 跟踪最后选择的臂和切换时间
        self.last_selected_arm = None
        self.last_switch_time = time.time()
        self.last_context = None
        
        # 启动决策线程
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()
        
        print(f"CMAB Linear Thompson Sampling initialized with {self.models_num} models")

    def init_thompson_sampling_models(self):
        """初始化每个模型的线性Thompson Sampling参数"""
        for i in range(self.models_num):
            self.models[i] = {
                # 参数均值向量 (零向量)
                'mu': np.zeros(self.context_dimension),
                
                # 参数协方差矩阵 (单位矩阵乘以正则化参数的逆)
                'Sigma': np.eye(self.context_dimension) / self.lambda_reg,
                
                # 足够统计量: X^T X
                'precision': self.lambda_reg * np.eye(self.context_dimension),
                
                # 足够统计量: X^T y
                'precision_mean': np.zeros(self.context_dimension),
                
                # 观察计数
                'count': 0,
                
                # 观察的奖励总和
                'sum_reward': 0,
                
                # 奖励样本方差估计
                'reward_variance': 1.0
            }

    def _switch_loop(self):
        """模型切换决策的主循环"""
        while True:
            current_time = time.time()
            
            # 如果达到了决策间隔时间
            if current_time - self.last_switch_time > self.decision_interval:
                # 获取当前统计数据以计算奖励
                current_stats = self.get_detector_stats()
                
                if current_stats and self.last_selected_arm is not None and self.last_context is not None:
                    # 将StatsEntry转换为字典用于奖励计算
                    stats_dict = self.stats_entry_to_dict(current_stats[0])
                    
                    # 计算上一次选择的臂的奖励
                    reward = self.calculate_reward(stats_dict)
                    
                    # 更新上一次选择的臂的参数
                    self.update_thompson_sampling(self.last_selected_arm, self.last_context, reward)
                
                # 提取当前上下文特征
                if current_stats:
                    context = self.extract_features(current_stats[0])
                    
                    # 使用Thompson采样选择新臂
                    selected_arm = self.select_model_thompson_sampling(context)
                    
                    # 切换到所选模型
                    self.switch_model(selected_arm)
                    print(f'Thompson Sampling switched model to {selected_arm}')
                    
                    # 更新追踪变量
                    self.last_selected_arm = selected_arm
                    self.last_context = context
                    self.last_switch_time = current_time
                    
                    # 打印当前臂的统计信息
                    self.print_arm_stats()
            
            time.sleep(0.1)

    def extract_features(self, stats_entry):
        """从StatsEntry中提取特征向量"""
        stats = self.stats_entry_to_dict(stats_entry)
        
        # 提取特征，只对需要的特征进行归一化
        # 固定特征数为10（9个特征 + 1个偏置项）
        features = np.zeros(10)
        
        # 只对queue_length, brightness和contrast归一化
        features[0] = float(stats['queue_length']) / self.queue_low_threshold_length  # 队列长度归一化
        features[1] = float(stats['processing_latency'])  # 不归一化
        features[2] = float(stats['target_nums'])  # 不归一化
        features[3] = float(stats['avg_confidence'])  # 不归一化
        features[4] = float(stats['std_confidence'])  # 不归一化
        features[5] = float(stats['avg_size'])  # 不归一化
        features[6] = float(stats['std_size'])  # 不归一化
        features[7] = float(stats['brightness']) / 255.0  # 亮度归一化
        features[8] = float(stats['contrast']) / 255.0  # 对比度归一化
        
        # 添加一个常数特征作为偏置项
        features[9] = 1.0
            
        # 打印提取的特征
        print("Extracted features:")
        for i, value in enumerate(features):
            print(f"  feature_{i}: {value:.4f}")
            
        return features

    def sample_parameter(self, arm):
        """从模型的后验分布中采样参数向量"""
        model_data = self.models[arm]
        
        try:
            # 使用Cholesky分解从多元正态分布中采样，提高数值稳定性
            L = linalg.cholesky(model_data['Sigma'], lower=True)
            
            # 采样标准正态分布
            standard_normal = np.random.standard_normal(self.context_dimension)
            
            # 变换为目标分布
            theta_sample = model_data['mu'] + L @ standard_normal
            
            return theta_sample
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"警告: 模型{arm}采样错误: {e}")
            print("使用均值向量替代采样")
            # 如果采样失败，返回均值向量
            return model_data['mu']

    def select_model_thompson_sampling(self, context):
        """使用Thompson Sampling策略选择模型"""
        # 检查是否是强制探索回合
        self.exploration_counter += 1
        force_exploration = self.exploration_counter >= self.force_exploration_count
        
        # 如果是强制探索回合，重置计数器并随机选择
        if force_exploration:
            self.exploration_counter = 0
            # 随机选择一个模型
            selected_arm = np.random.randint(0, self.models_num)
            print(f"强制探索: 随机选择模型 {selected_arm}")
            return selected_arm
        
        # ε-greedy随机探索
        if np.random.random() < self.exploration_rate:
            selected_arm = np.random.randint(0, self.models_num)
            print(f"随机探索: 选择模型 {selected_arm}")
            return selected_arm
        
        # Thompson Sampling决策
        expected_rewards = {}
        sampled_params = {}
        
        # 为每个模型采样参数并计算期望奖励
        for arm in range(self.models_num):
            # 从后验分布采样参数向量
            theta = self.sample_parameter(arm)
            sampled_params[arm] = theta
            
            # 计算期望奖励
            expected_reward = np.dot(theta, context)
            expected_rewards[arm] = float(expected_reward)  # 确保是Python浮点数
        
        # 选择期望奖励最高的模型
        selected_arm = max(expected_rewards, key=expected_rewards.get)
        
        print(f"Thompson sampling选择模型: {selected_arm} (期望奖励={expected_rewards[selected_arm]:.4f})")
        
        # 打印所有模型的预期奖励
        for arm, reward in expected_rewards.items():
            print(f"  模型{arm}: 期望奖励={reward:.4f}, "
                  f"参数范数={np.linalg.norm(sampled_params[arm]):.4f}")
        
        return selected_arm

    def update_thompson_sampling(self, arm, context, reward):
        """更新Thompson Sampling模型的参数"""
        model_data = self.models[arm]
        
        # 累计计数和奖励
        model_data['count'] += 1
        model_data['sum_reward'] += reward
        
        # 更新协方差和精度矩阵
        context_2d = context.reshape(-1, 1)  # 列向量
        
        # 更新精度矩阵 (X^T X)
        model_data['precision'] += context_2d @ context_2d.T
        
        # 更新精度均值 (X^T y)
        model_data['precision_mean'] += context * reward
        
        # 重新计算均值向量和协方差矩阵
        try:
            # 计算协方差矩阵 (Sigma)
            model_data['Sigma'] = np.linalg.inv(model_data['precision'])
            
            # 计算均值向量 (mu = Sigma * precision_mean)
            model_data['mu'] = model_data['Sigma'] @ model_data['precision_mean']
            
            # 更新奖励方差估计
            if model_data['count'] > 1:
                avg_reward = model_data['sum_reward'] / model_data['count']
                # 简单估计方差
                model_data['reward_variance'] = max(0.1, self.noise_variance)
            
            print(f"更新模型{arm}的Thompson Sampling参数:")
            print(f"  count={model_data['count']}, avg_reward={model_data['sum_reward']/model_data['count']:.4f}")
            print(f"  mu_norm={np.linalg.norm(model_data['mu']):.4f}, var={model_data['reward_variance']:.4f}")
        except np.linalg.LinAlgError:
            print(f"警告: 无法计算模型{arm}的精度矩阵逆。使用先前的值。")

    def calculate_reward(self, stats):
        """计算模型的奖励值"""
        queue_ratio = stats['queue_length'] / self.queue_low_threshold_length
        
        # 权重计算
        w1 = max(1 - queue_ratio, 0)  # 准确率权重
        w2 = queue_ratio  # 延迟权重
        
        # 奖励计算
        raw_reward = w1 * (stats['cur_model_accuracy']/100.0 + stats['avg_confidence']) - \
                w2 * (stats['processing_latency'])
        
        print(f"计算奖励: {raw_reward:.4f} (w1={w1:.2f}, w2={w2:.2f})")
        
        return raw_reward

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
    
    def stats_entry_to_dict(self, stats_entry):
        """将StatsEntry对象转换为字典"""
        if stats_entry is None:
            return {}
        
        return {
            'timestamp': stats_entry.timestamp,
            'queue_length': stats_entry.queue_length,
            'cur_model_index': stats_entry.cur_model_index,
            'cur_model_accuracy': stats_entry.cur_model_accuracy,
            'processing_latency': stats_entry.processing_latency,
            'target_nums': stats_entry.target_nums,
            'avg_confidence': stats_entry.avg_confidence,
            'std_confidence': stats_entry.std_confidence,
            'avg_size': stats_entry.avg_size,
            'std_size': stats_entry.std_size,
            'brightness': stats_entry.brightness,
            'contrast': stats_entry.contrast
        }
    
    def print_arm_stats(self):
        """打印所有臂的当前统计信息"""
        print("\n线性Thompson Sampling模型参数:")
        print("------------------------")
        for arm in range(self.models_num):
            model_data = self.models[arm]
            if model_data['count'] > 0:
                avg_reward = model_data['sum_reward'] / model_data['count']
            else:
                avg_reward = 0.0
            print(f"模型 {arm}: count={model_data['count']}, avg_reward={avg_reward:.4f}, "
                  f"param_norm={np.linalg.norm(model_data['mu']):.4f}")
        print("------------------------\n")