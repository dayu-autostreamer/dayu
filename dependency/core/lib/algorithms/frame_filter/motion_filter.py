import abc
import time
import cv2
import numpy as np
from core.lib.common import ClassFactory, ClassType
from .base_filter import BaseFilter

__all__ = ('MotionFilter',)


@ClassFactory.register(ClassType.GEN_FILTER, alias='motion')
class MotionFilter(BaseFilter, abc.ABC):
    def __init__(self, 
                 min_fps=1, 
                 max_fps=None, 
                 motion_threshold_min=0.001, 
                 motion_threshold_max=0.05,
                 smoothing_factor=0.9,
                 history=500,
                 var_threshold=16):
        """
        基于运动检测的自适应帧率过滤器。
        
        Args:
            min_fps: 最低帧率，当场景无运动时使用此帧率
            max_fps: 最高帧率，当场景大动作时使用此帧率，None则使用原始帧率
            motion_threshold_min: 最小运动阈值，低于此值使用最低帧率
            motion_threshold_max: 最大运动阈值，高于此值使用最高帧率
            smoothing_factor: 平滑因子(0-1)，越大越平滑，但响应越慢
            history: 背景模型的历史帧数
            var_threshold: 背景模型的方差阈值
        """
        # 初始化基类
        super().__init__()
        
        # 帧率控制参数
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.motion_threshold_min = motion_threshold_min
        self.motion_threshold_max = motion_threshold_max
        self.smoothing_factor = smoothing_factor
        
        # 背景模型参数
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel = np.ones((5, 5), np.uint8)
        
        # 状态变量
        self.frame_count = 0
        self.prev_frame = None
        self.current_fps = min_fps
        self.last_frame_time = time.time()
        self.previous_decisions = []  # 存储最近的决策，用于平滑

    def __call__(self, system, frame) -> bool:
        """
        根据当前帧的运动情况决定是否保留该帧。
        
        Args:
            system: 系统对象，包含元数据
            frame: 当前帧
            
        Returns:
            bool: True表示保留该帧，False表示丢弃该帧
        """
        # 递增帧计数
        self.frame_count += 1
        
        # 获取原始帧率和目标帧率
        fps_raw = int(system.raw_meta_data['fps'])
        fps_config = int(system.meta_data['fps'])
        self.max_fps = min(fps_raw, fps_config)

        
        # 计算运动量
        motion_ratio = self._calculate_motion(frame)
        
        # 根据运动量调整目标帧率
        target_fps = self._calculate_target_fps(motion_ratio)
        
        # 平滑帧率变化
        self.current_fps = self.smoothing_factor * self.current_fps + (1 - self.smoothing_factor) * target_fps

        print(f'frame_count: {self.frame_count}, fps_raw: {fps_raw}, max_fps: {self.max_fps}, target_fps: {target_fps}, current_fps: {self.current_fps}, motion_ratio: {motion_ratio}')
        
        # 根据当前帧率决定是否保留该帧
        fps_mode, skip_frame_interval, remain_frame_interval = self.get_fps_adjust_mode(fps_raw, round(self.current_fps))
        
        decision = True
        if fps_mode == 'skip' and self.frame_count % skip_frame_interval == 0:
            decision = False
        
        if fps_mode == 'remain' and self.frame_count % remain_frame_interval != 0:
            decision = False

        print(f'fps_mode: {fps_mode}, skip_frame_interval: {skip_frame_interval}, remain_frame_interval: {remain_frame_interval}, decision: {decision}')
        
        return decision
    
    def _calculate_motion(self, frame):
        """计算当前帧的运动量"""
        # 应用背景减除
        fgmask = self.bg_subtractor.apply(frame)
        
        # 应用形态学操作去噪
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        
        # 计算运动量 (前景像素占比)
        motion_ratio = np.count_nonzero(fgmask) / float(fgmask.size)
        
        return motion_ratio
    
    def _calculate_target_fps(self, motion_ratio):
        """根据运动量计算目标帧率"""
        if motion_ratio < self.motion_threshold_min:
            return self.min_fps
        elif motion_ratio > self.motion_threshold_max:
            return self.max_fps
        else:
            # 在最小和最大阈值之间进行线性插值
            motion_scale = (motion_ratio - self.motion_threshold_min) / (self.motion_threshold_max - self.motion_threshold_min)
            return self.min_fps + motion_scale * (self.max_fps - self.min_fps)
    
    @staticmethod
    def get_fps_adjust_mode(fps_raw, fps):
        """
        计算帧率调整模式
        
        Args:
            fps_raw: 原始帧率
            fps: 目标帧率
            
        Returns:
            fps_mode: 帧率调整模式，'same'表示不调整，'skip'表示跳帧，'remain'表示保留特定帧
            skip_frame_interval: 跳帧间隔
            remain_frame_interval: 保留帧间隔
        """
        skip_frame_interval = 0
        remain_frame_interval = 0
        
        if fps >= fps_raw:
            fps_mode = 'same'
        elif fps < fps_raw // 2:
            fps_mode = 'remain'
            remain_frame_interval = fps_raw // fps
        else:
            fps_mode = 'skip'
            skip_frame_interval = fps_raw // (fps_raw - fps)
        
        return fps_mode, skip_frame_interval, remain_frame_interval