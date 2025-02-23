from .base_switch import BaseSwitch
import numpy as np
import time
import threading
from typing import List, Dict
from dataclasses import dataclass
import math

class LinUCBArm:
    def __init__(self, context_dim: int, alpha: float = 1.0):
        self.alpha = alpha
        self.A = np.identity(context_dim)  # d x d
        self.b = np.zeros((context_dim, 1))  # d x 1
        self.theta = np.zeros((context_dim, 1))  # d x 1
        
    def update(self, context: np.ndarray, reward: float):
        """更新模型参数"""
        context = context.reshape(-1, 1)  # d x 1
        self.A += context @ context.T
        self.b += reward * context
        self.theta = np.linalg.solve(self.A, self.b)
        
    def get_ucb_score(self, context: np.ndarray) -> float:
        """计算UCB分数"""
        context = context.reshape(-1, 1)  # d x 1
        A_inv = np.linalg.inv(self.A)
        mean = float(context.T @ self.theta)
        var = float(self.alpha * np.sqrt(context.T @ A_inv @ context))
        return mean + var

class CMABSwitch(BaseSwitch):
    def __init__(self, 
                 decision_interval: int,
                 detector_instance: object,
                 alpha: float = 1.0,
                 min_samples: int = 10,
                 *args, **kwargs):
        """
        初始化CMAB切换器
        
        Args:
            decision_interval: 决策间隔(秒)
            detector_instance: 检测器实例
            alpha: UCB探索参数
            min_samples: 每个臂最少需要的样本数
        """
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.alpha = alpha
        self.min_samples = min_samples
        
        # 特征维度: [queue_length, cur_model_accuracy, processing_latency, target_nums, 
        #            avg_confidence, std_confidence, avg_size, std_size, brightness, contrast]
        self.context_dim = 10
        
        # 为每个模型初始化一个LinUCB实例
        self.arms = [LinUCBArm(self.context_dim, alpha) for _ in range(self.models_num)]
        
        # 记录每个模型的使用次数
        self.model_counts = np.zeros(self.models_num)
        
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
        features = np.clip(features, -10, 10)  # 截断异常值
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
            
    def _make_decision(self):
        """做出切换决策"""
        # 获取当前状态
        stats = self.detector_instance.stats_manager.stats
        if not stats:
            return
            
        # 提取特征和计算奖励
        context = self._extract_context(stats)
        reward = self._compute_reward(stats)
        
        # 更新当前模型的参数
        current_model = stats[-1].cur_model_index
        self.arms[current_model].update(context, reward)
        self.model_counts[current_model] += 1
        
        # 计算每个模型的UCB分数
        ucb_scores = []
        for i in range(self.models_num):
            if self.model_counts[i] < self.min_samples:
                # 如果样本不足，给予高分以鼓励探索
                score = float('inf')
            else:
                score = self.arms[i].get_ucb_score(context)
            ucb_scores.append(score)
            
        # 选择得分最高的模型
        best_model = np.argmax(ucb_scores)
        
        # 如果最佳模型与当前模型不同，则切换
        if best_model != current_model:
            print(f'CMAB switched model from {current_model} to {best_model} (UCB scores: {ucb_scores})')
            self.switch_model(best_model)
            
    def switch_model(self, index: int):
        """切换到指定的模型"""
        self.detector_instance.switch_model(index)
        
    def get_detector_stats(self, *args, **kwargs):
        return super().get_detector_stats(**kwargs)