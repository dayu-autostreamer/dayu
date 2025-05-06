from .base_switch import BaseSwitch
import time
import threading
import numpy as np
from scipy import linalg
import numpy as np

class DummyStrategy:
    """占位符策略类，将被实际的策略实现替换"""
    
    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        self.queue_low_threshold_length = queue_low_threshold_length
        
    def select_model(self, stats, current_model_index):
        """返回占位符模型决策"""
        return current_model_index


class QueueLengthStrategy:
    """基于冷却机制的自适应队列长度策略"""
    
    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        
        # 队列阈值
        self.queue_threshold = queue_low_threshold_length * 2.0
        
        # 稳定性相关参数
        self.stability_counter = 0
        self.base_stability_threshold = 3
        
        # 冷却机制参数
        self.cooling_factor = 1
        self.max_cooling_factor = 5
        self.cooling_recovery_rate = 1
    
    def select_model(self, stats, current_model_index):
        """基于队列长度做出模型切换决策"""
        queue_length = stats.get('queue_length', 0)
        
        # 更新系统稳定状态
        is_currently_stable = queue_length <= self.queue_threshold
        
        # 如果系统不稳定(队列超过阈值)，立即考虑降级
        if not is_currently_stable:
            self.stability_counter = 0  # 重置稳定计数
            
            # 如果有更轻量级的模型可用，则降级
            if current_model_index > 0:
                next_model = current_model_index - 1
                print(f"队列长度({queue_length})超过阈值({self.queue_threshold})。从模型{current_model_index}降级到{next_model}")
                return next_model
            else:
                print(f"队列长度({queue_length})超过阈值，但已经使用最轻量级模型。")
                return current_model_index
        
        # 如果系统稳定(队列低于阈值)，考虑是否可以升级
        else:
            # 增加稳定计数
            self.stability_counter += 1
            print(f"系统稳定: 稳定计数={self.stability_counter}/{self.base_stability_threshold * self.cooling_factor}")
            
            # 降低冷却系数
            if self.cooling_factor > 1 and self.stability_counter % 2 == 0:
                self.cooling_factor = max(1, self.cooling_factor - self.cooling_recovery_rate)
                print(f"冷却因子降低为: {self.cooling_factor}")
            
            # 连续稳定次数超过阈值时才升级
            current_stability_threshold = int(self.base_stability_threshold * self.cooling_factor)
            if self.stability_counter >= current_stability_threshold:
                # 如果有更高级的模型可用，则升级
                if current_model_index < self.models_num - 1:
                    next_model = current_model_index + 1
                    print(f"系统连续稳定({self.stability_counter}/{current_stability_threshold})。从模型{current_model_index}升级到{next_model}")
                    self.stability_counter = 0
                    return next_model
                else:
                    print("系统稳定，但已经使用最高级模型。")
                    self.stability_counter = 0
            
        # 保持当前模型
        return current_model_index


class ProbabilisticStrategy:
    """基于概率的自适应策略
    
    根据队列长度的不同，以不同概率做出模型切换决策:
    1. 队列长度低于低阈值时，高概率升级
    2. 队列长度在低阈值与高阈值之间时，按比例概率决策
    3. 队列长度高于高阈值时，高概率降级
    """
    
    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        
        # 定义队列阈值
        self.queue_low = queue_low_threshold_length
        self.queue_high = queue_low_threshold_length * 2
        
        # 概率参数
        self.upgrade_prob_base = 0.7  # 基础升级概率
        self.downgrade_prob_base = 0.8  # 基础降级概率
        
        # 稳定计数器，用于模型稳定性
        self.stability_counter = 0
        self.min_stability_period = 3  # 最小稳定周期
    
    def select_model(self, stats, current_model_index):
        """根据队列长度和概率做出模型切换决策"""
        # 获取当前队列长度
        queue_length = stats.get('queue_length', 0)
        
        # 打印当前状态
        print(f"概率策略分析 - 当前队列长度: {queue_length}, 低阈值: {self.queue_low}, 高阈值: {self.queue_high}")
        print(f"当前模型: {current_model_index}, 总模型数: {self.models_num}")
        
        # 增加稳定计数器
        self.stability_counter += 1
        
        # 检查是否应该考虑切换（稳定期内不切换）
        if self.stability_counter < self.min_stability_period:
            print(f"处于稳定期 ({self.stability_counter}/{self.min_stability_period})，维持当前模型")
            return current_model_index
        
        # 计算升级和降级的基础概率
        if queue_length <= self.queue_low:
            # 队列负载低，考虑升级
            upgrade_probability = self.upgrade_prob_base
            downgrade_probability = 0.0
            print(f"队列负载低 - 升级概率: {upgrade_probability:.2f}, 降级概率: {downgrade_probability:.2f}")
            
        elif queue_length >= self.queue_high:
            # 队列负载高，考虑降级
            upgrade_probability = 0.0
            downgrade_probability = self.downgrade_prob_base
            print(f"队列负载高 - 升级概率: {upgrade_probability:.2f}, 降级概率: {downgrade_probability:.2f}")
            
        else:
            # 队列负载适中，根据负载程度调整概率
            load_ratio = (queue_length - self.queue_low) / (self.queue_high - self.queue_low)
            upgrade_probability = self.upgrade_prob_base * (1 - load_ratio)
            downgrade_probability = self.downgrade_prob_base * load_ratio
            print(f"队列负载适中(比率: {load_ratio:.2f}) - 升级概率: {upgrade_probability:.2f}, 降级概率: {downgrade_probability:.2f}")
        
        # 根据当前模型位置调整概率（最高/最低级模型不能再升级/降级）
        if current_model_index == 0:  # 最低级模型
            downgrade_probability = 0.0
        if current_model_index == self.models_num - 1:  # 最高级模型
            upgrade_probability = 0.0
        
        # 生成随机数并做出决策
        random_value = np.random.random()
        
        if random_value < upgrade_probability:
            # 触发升级
            if current_model_index < self.models_num - 1:
                next_model = current_model_index + 1
                print(f"概率策略 - 随机值({random_value:.2f}) < 升级概率({upgrade_probability:.2f})，升级至模型 {next_model}")
                self.stability_counter = 0  # 重置稳定计数器
                return next_model
            
        elif random_value < upgrade_probability + downgrade_probability:
            # 触发降级
            if current_model_index > 0:
                next_model = current_model_index - 1
                print(f"概率策略 - 随机值({random_value:.2f}) < 降级概率({downgrade_probability:.2f})，降级至模型 {next_model}")
                self.stability_counter = 0  # 重置稳定计数器
                return next_model
        
        # 默认保持当前模型
        print(f"概率策略 - 保持当前模型 {current_model_index}")
        return current_model_index


class DistributionBasedStrategy:
    """基于目标数量和大小分布的自适应策略"""
    
    def __init__(self, models_num, queue_low_threshold_length):
        self.models_num = models_num
        self.queue_low_threshold = queue_low_threshold_length
        
        # 统计窗口大小
        self.window_size = 100
        
        # 初始化统计窗口
        self.target_nums_history = []  # 目标数量历史
        self.target_size_history = []  # 目标大小历史
        
        # 统计数据
        self.target_nums_mean = None
        self.target_nums_std = None
        self.target_size_mean = None
        self.target_size_std = None
        
        # 偏离标准差的阈值
        self.std_threshold = 1.0
        
        # 稳定性计数器
        self.upgrade_counter = 0
        self.downgrade_counter = 0
        self.stability_threshold = 3
    
    def _update_statistics(self, target_nums, target_size):
        """更新统计数据"""
        # 更新历史数据
        self.target_nums_history.append(target_nums)
        self.target_size_history.append(target_size)
        
        # 保持窗口大小
        if len(self.target_nums_history) > self.window_size:
            self.target_nums_history.pop(0)
            self.target_size_history.pop(0)
        
        # 只有当收集了足够的样本时才计算统计数据
        if len(self.target_nums_history) >= 10:
            self.target_nums_mean = np.mean(self.target_nums_history)
            self.target_nums_std = np.std(self.target_nums_history) or 1.0  # 避免除零
            
            self.target_size_mean = np.mean(self.target_size_history)
            self.target_size_std = np.std(self.target_size_history) or 1.0  # 避免除零
    
    def select_model(self, stats, current_model_index):
        """根据目标数量和大小的分布选择模型"""
        target_nums = stats.get('target_nums', 0)
        avg_size = stats.get('avg_size', 0)
        
        # 更新统计数据
        self._update_statistics(target_nums, avg_size)
        
        # 如果统计数据尚未初始化，保持当前模型
        if self.target_nums_mean is None:
            print("统计数据尚未初始化，保持当前模型")
            return current_model_index
        
        # 计算当前状态与均值的偏差（以标准差为单位）
        nums_deviation = (target_nums - self.target_nums_mean) / self.target_nums_std
        size_deviation = (avg_size - self.target_size_mean) / self.target_size_std
        
        print(f"目标数量: {target_nums} (均值: {self.target_nums_mean:.2f}, 偏差: {nums_deviation:.2f}σ)")
        print(f"目标大小: {avg_size:.2f} (均值: {self.target_size_mean:.2f}, 偏差: {size_deviation:.2f}σ)")
        
        # 检查是否需要升级模型
        need_upgrade = (
            nums_deviation > self.std_threshold or  # 目标数量超过均值+标准差
            size_deviation < -self.std_threshold    # 目标大小小于均值-标准差
        )
        
        # 检查是否需要降级模型
        need_downgrade = (
            nums_deviation < -self.std_threshold or  # 目标数量小于均值-标准差
            size_deviation > self.std_threshold      # 目标大小大于均值+标准差
        )
        
        # 决策逻辑
        if need_upgrade:
            self.upgrade_counter += 1
            self.downgrade_counter = 0
            print(f"可能需要升级: {self.upgrade_counter}/{self.stability_threshold}")
            
            if self.upgrade_counter >= self.stability_threshold:
                if current_model_index < self.models_num - 1:
                    next_model = current_model_index + 1
                    print(f"持续需要升级，切换至模型 {next_model}")
                    self.upgrade_counter = 0
                    return next_model
                else:
                    print("需要升级，但已经使用最高级模型")
                    self.upgrade_counter = 0
        
        elif need_downgrade:
            self.downgrade_counter += 1
            self.upgrade_counter = 0
            print(f"可能需要降级: {self.downgrade_counter}/{self.stability_threshold}")
            
            if self.downgrade_counter >= self.stability_threshold:
                if current_model_index > 0:
                    next_model = current_model_index - 1
                    print(f"持续需要降级，切换至模型 {next_model}")
                    self.downgrade_counter = 0
                    return next_model
                else:
                    print("需要降级，但已经使用最轻量级模型")
                    self.downgrade_counter = 0
        
        # 如果没有明确的升级/降级信号，保持当前模型
        return current_model_index
    
class MetaSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 queue_low_threshold_length: int = 10,
                 lambda_reg: float = 1.0,
                 noise_variance: float = 0.1,
                 exploration_rate: float = 0.1,
                 *args, **kwargs):
        """
        元策略切换器，使用Thompson采样在多个基础策略之间进行选择
        
        Args:
            decision_interval: 模型切换决策的时间间隔(秒)
            detector_instance: 检测器实例
            queue_low_threshold_length: 队列长度阈值，用于奖励计算
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
        
        # 初始化基础策略
        self.strategies = self._initialize_strategies()
        
        # 元策略Thompson Sampling参数
        self.meta_models = {}
        self.init_meta_thompson_sampling()
        
        # 跟踪最后选择的策略、模型和切换时间
        self.last_selected_strategy = None
        self.last_selected_arm = None
        self.last_switch_time = time.time()
        self.last_context = None
        
        # 启动决策线程
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.daemon = True
        self.switch_thread.start()
        
        print(f"MetaSwitch initialized with {self.models_num} models and {len(self.strategies)} strategies")

    def _initialize_strategies(self):
        """初始化基础策略集合"""
        # 这里可以添加各种策略，目前使用占位符策略
        strategies = {
            'strategy1': QueueLengthStrategy(self.models_num, self.queue_low_threshold_length),
            'strategy2': ProbabilisticStrategy(self.models_num, self.queue_low_threshold_length),
            'strategy3': DistributionBasedStrategy(self.models_num, self.queue_low_threshold_length)
        }
        return strategies

    def init_meta_thompson_sampling(self):
        """初始化元策略的Thompson Sampling参数"""
        for strategy_name in self.strategies.keys():
            self.meta_models[strategy_name] = {
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
                
                if current_stats and self.last_selected_strategy is not None and self.last_context is not None and self.last_selected_arm is not None:
                    # 将StatsEntry转换为字典用于奖励计算
                    stats_dict = self.stats_entry_to_dict(current_stats[0])
                    
                    # 计算上一次的奖励
                    reward = self.calculate_reward(stats_dict)
                    
                    # 更新上一次选择的策略的参数
                    self.update_meta_thompson_sampling(self.last_selected_strategy, self.last_context, reward)
                
                # 提取当前上下文特征
                if current_stats:
                    context = self.extract_features(current_stats[0])
                    
                    # 使用元Thompson采样选择策略
                    selected_strategy = self.select_meta_strategy(context)
                    
                    # 使用选定的策略获取模型建议
                    stats_dict = self.stats_entry_to_dict(current_stats[0])
                    current_model_idx = stats_dict['cur_model_index']
                    selected_arm = self.strategies[selected_strategy].select_model(stats_dict, current_model_idx)
                    
                    # 切换到所选模型
                    self.switch_model(selected_arm)
                    print(f'MetaSwitch: strategy {selected_strategy} switched model to {selected_arm}')
                    
                    # 更新追踪变量
                    self.last_selected_strategy = selected_strategy
                    self.last_selected_arm = selected_arm
                    self.last_context = context
                    self.last_switch_time = current_time
                    
                    # 打印当前元策略统计信息
                    self.print_meta_stats()
            
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

    def sample_meta_parameter(self, strategy_name):
        """从元策略的后验分布中采样参数向量"""
        model_data = self.meta_models[strategy_name]
        
        try:
            # 使用Cholesky分解从多元正态分布中采样，提高数值稳定性
            L = linalg.cholesky(model_data['Sigma'], lower=True)
            
            # 采样标准正态分布
            standard_normal = np.random.standard_normal(self.context_dimension)
            
            # 变换为目标分布
            theta_sample = model_data['mu'] + L @ standard_normal
            
            return theta_sample
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"警告: 策略{strategy_name}采样错误: {e}")
            print("使用均值向量替代采样")
            # 如果采样失败，返回均值向量
            return model_data['mu']

    def select_meta_strategy(self, context):
        """使用元Thompson Sampling策略选择基础策略"""
        # 检查是否是强制探索回合
        self.exploration_counter += 1
        force_exploration = self.exploration_counter >= self.force_exploration_count
        
        # 如果是强制探索回合，重置计数器并随机选择
        if force_exploration:
            self.exploration_counter = 0
            # 随机选择一个策略
            selected_strategy = np.random.choice(list(self.strategies.keys()))
            print(f"强制探索: 随机选择策略 {selected_strategy}")
            return selected_strategy
        
        # ε-greedy随机探索
        if np.random.random() < self.exploration_rate:
            selected_strategy = np.random.choice(list(self.strategies.keys()))
            print(f"随机探索: 选择策略 {selected_strategy}")
            return selected_strategy
        
        # 元Thompson Sampling决策
        expected_rewards = {}
        sampled_params = {}
        
        # 为每个策略采样参数并计算期望奖励
        for strategy_name in self.strategies.keys():
            # 从后验分布采样参数向量
            theta = self.sample_meta_parameter(strategy_name)
            sampled_params[strategy_name] = theta
            
            # 计算期望奖励
            expected_reward = np.dot(theta, context)
            expected_rewards[strategy_name] = float(expected_reward)  # 确保是Python浮点数
        
        # 选择期望奖励最高的策略
        selected_strategy = max(expected_rewards, key=expected_rewards.get)
        
        print(f"Meta Thompson sampling选择策略: {selected_strategy} (期望奖励={expected_rewards[selected_strategy]:.4f})")
        
        # 打印所有策略的预期奖励
        for strategy, reward in expected_rewards.items():
            print(f"  策略{strategy}: 期望奖励={reward:.4f}, "
                  f"参数范数={np.linalg.norm(sampled_params[strategy]):.4f}")
        
        return selected_strategy

    def update_meta_thompson_sampling(self, strategy_name, context, reward):
        """更新元Thompson Sampling模型的参数"""
        model_data = self.meta_models[strategy_name]
        
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
            
            print(f"更新策略{strategy_name}的元Thompson Sampling参数:")
            print(f"  count={model_data['count']}, avg_reward={model_data['sum_reward']/model_data['count']:.4f}")
            print(f"  mu_norm={np.linalg.norm(model_data['mu']):.4f}, var={model_data['reward_variance']:.4f}")
        except np.linalg.LinAlgError:
            print(f"警告: 无法计算策略{strategy_name}的精度矩阵逆。使用先前的值。")

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
    
    def print_meta_stats(self):
        """打印所有元策略的当前统计信息"""
        print("\n元Thompson Sampling策略参数:")
        print("------------------------")
        for strategy_name in self.strategies.keys():
            model_data = self.meta_models[strategy_name]
            if model_data['count'] > 0:
                avg_reward = model_data['sum_reward'] / model_data['count']
            else:
                avg_reward = 0.0
            print(f"策略 {strategy_name}: count={model_data['count']}, avg_reward={avg_reward:.4f}, "
                  f"param_norm={np.linalg.norm(model_data['mu']):.4f}")
        print("------------------------\n")