from .base_switch import BaseSwitch
from typing import List
import time
import threading
import numpy as np

class RuleBasedSwitch(BaseSwitch):
    def __init__(self, decision_interval: int, 
                 detector_instance: object,
                 *args, **kwargs):
        '''
        Initialize the random switcher.
        '''
        self.models_num = detector_instance.get_models_num()
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance

        self.queue_threshold = 10  # 队列积压警戒值
        self.current_pressure = 0  # 系统压力计数器
        self.pressure_threshold = 2  # 连续压力次数阈值

        self.last_switch_time = time.time()
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.start()

    def _switch_loop(self):
        while True:
            if time.time() - self.last_switch_time > self.decision_interval:
                model_index = self._get_model_index()
                self.switch_model(model_index)
                print(f'Rule based switched model to {model_index}')
                self.last_switch_time = time.time()
            time.sleep(0.1)

    def _get_model_index(self) -> int:
        """基于规则决定使用哪个模型"""
        stats = self.detector_instance.stats_manager.stats
        if not stats:
            return self.detector_instance.current_model_index
            
        # 获取最近的统计信息
        latest_stats = stats[-1]
        current_index = latest_stats.cur_model_index
        
        # 检查系统压力
        if latest_stats.queue_length > self.queue_threshold:
            self.current_pressure += 1
        else:
            self.current_pressure = max(0, self.current_pressure - 1)
            
        # 决策逻辑
        if self.current_pressure >= self.pressure_threshold:
            # 系统压力大，切换到更轻量级模型
            self.current_pressure = 0  # 重置压力计数器
            return max(0, current_index - 2)
        elif self.current_pressure == 0:
            # 系统压力小，尝试切换到更大模型
            if current_index < self.models_num - 1:
                return current_index + 1
                
        # 保持当前模型
        return current_index

    def switch_model(self, index: int):
        '''
        Switch the model to the one specified in the arguments.
        '''
        self.detector_instance.switch_model(index)

    def get_detector_stats(*args, **kwargs):
        return super().get_detector_stats(**kwargs)
        