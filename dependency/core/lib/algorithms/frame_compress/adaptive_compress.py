"""
Adaptive video encoding:
    It proposes a region-adaptive video coding algorithm for cloud-edge collaborative object detection.
    The algorithm achieves efficient candidate region extraction through lightweight foreground detection and historical inference result analysis. 
    use a reinforcement learning-based adaptive coding strategy that dynamically adjusts coding parameters by combining network state awareness and region complexity assessment. 

Bingyun Yang, Jingyi Ning, Wenhui Zhou, Chuyu Wang, Lei Xie, Zhenjie Lin, Liming Wang. Adaptive Region-aware Video Encoding for Real-time Cloud-edge Collaborative Object Detection. Accepted to Appear In Proceeding of IEEE International Conference on Distributed Computing Systems (ICDCS 2025, poster paper).
"""

import abc
import os

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context
from .base_compress import BaseCompress
import cv2
import pickle

__all__ = ('AdaptiveCompress',)


@ClassFactory.register(ClassType.GEN_COMPRESS, alias='adaptive')
class AdaptiveCompress(BaseCompress, abc.ABC):
    def __init__(self):
        # 多任务可能存在问题
        self.past_acc = 0
        self.past_latency = 0
        self.past_qp = 45
        self.agent_file = Context.get_file_path('model.pkl')
        self.agent = self.load_model(self.agent_file)
        self.agent.epsilon = 0
        self.performace_gt = [
            (20, 0.37, 739.63671875, 0.76),
            (21, 0.37, 658.810546875, 0.76),
            (22, 0.37, 577.5419921875, 0.76),
            (23, 0.37, 481.552734375, 0.76),
            (24, 0.37, 416.9892578125, 0.76),
            (25, 0.37, 371.4248046875, 0.76),
            (26, 0.37, 300.8740234375, 0.76),
            (27, 0.37, 260.12109375, 0.76),
            (28, 0.37, 221.7900390625, 0.76),
            (29, 0.37, 182.3203125, 0.76),
            (30, 0.37, 155.515625, 0.76),
            (31, 0.37, 138.966796875, 0.76),
            (32, 0.37, 113.9091796875, 0.76),
            (33, 0.37, 98.8515625, 0.75),
            (34, 0.37, 85.9111328125, 0.75),
            (35, 0.37, 73.7109375, 0.75),
            (36, 0.37, 64.5966796875, 0.74),
            (37, 0.37, 58.94140625, 0.74),
            (38, 0.37, 50.30078125, 0.73),
            (39, 0.37, 45.97265625, 0.73),
            (40, 0.37, 41.02734375, 0.73),
            (41, 0.37, 36.8876953125, 0.71),
            (42, 0.37, 33.4677734375, 0.69),
            (43, 0.37, 32.0673828125, 0.67),
            (44, 0.37, 28.5615234375, 0.64),
            (45, 0.37, 26.3505859375, 0.61),
            (46, 0.37, 24.4951171875, 0.57),
            (47, 0.37, 22.208984375, 0.53),
            (48, 0.37, 20.7158203125, 0.49),
            (49, 0.37, 19.6279296875, 0.47),
            (50, 0.37, 17.6650390625, 0.42),
            (51, 0.37, 16.8994140625, 0.39)
        ]

    def __call__(self, system, frame_buffer, source_id, task_id):
        import subprocess

        assert frame_buffer, 'frame buffer is empty!'

        frames = [data[0] for data in frame_buffer]
        rois = [data[1] for data in frame_buffer]

        height, width, _ = frames[0].shape
        yuv_path = self.generate_yuv_temp_path(source_id, task_id)
        self.init_yuv_temp_path(yuv_path, frames)

        h264_path = self.generate_file_path(source_id, task_id)

        complexity_all, complexity_roi = self.analyze_packet_content(frames, rois)

        cqp = self.adjust_qp(
                    self.performace_gt, complexity_all, complexity_roi, 
                    self.agent,
                    self.past_qp
                )
        roi_path = self.generate_roi_path(source_id, task_id)
        command = (
                    f'./video_encode {yuv_path} {width} {height} H264 {h264_path} '
                    f'--econstqp -qpi {cqp} {cqp} {cqp} '
                    f'--roi -roi {roi_path} '
                    f'--input-metadata --blocking-mode 0'
                )
        process = subprocess.Popen(command, shell=True)
        process.wait()

        LOGGER.debug(f'[Generator Compress] compress the buffer frame, bkg QP: {cqp}')

        FileOps.remove_file(roi_path)
        FileOps.remove_file(yuv_path)


        return h264_path

    @staticmethod
    def load_model(filename='agent_model.pkl'):
        with open(filename, 'rb') as f:
            agent = pickle.load(f)
        LOGGER.info(f"Model loaded from {filename}")
        return agent

    @staticmethod
    def calculate_edge_density(frame, roi=None):
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            gray = gray[roi[1]:roi[3], roi[0]:roi[2]]
        edges = cv2.Canny(gray, 100, 200)
        # edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        edge_density = np.sum(edges)
        return edge_density
    
    @staticmethod
    def calculate_texture_complexity(frame, roi=None):
        from skimage.feature import graycomatrix, graycoprops
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            gray = gray[roi[1]:roi[3], roi[0]:roi[2]]
        glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')
        return contrast[0, 0]

    def analyze_packet_content(self, frames, rois=None, roi_weight=1.2):
        """
        通过计算帧的开头、中间和结尾部分的复杂度，
        然后取平均并乘以帧数来减少计算量。
        返回：
            total_complexity: 所有帧的总复杂度
            roi_complexity: 所有帧的 ROI 区域复杂度
        """
        def process_frame_both(frame, current_rois):
            total_complexity = 0
            roi_complexity = 0

            # 全帧复杂度
            total_edge_density = self.calculate_edge_density(frame)
            total_texture_complexity = self.calculate_texture_complexity(frame)
            total_complexity = (total_edge_density + total_texture_complexity) / 2

            # ROI复杂度
            if current_rois:
                for roi in current_rois:
                    edge_density = self.calculate_edge_density(frame, roi)
                    texture_complexity = self.calculate_texture_complexity(frame, roi)
                    edge_density *= roi_weight
                    texture_complexity *= roi_weight
                    roi_complexity += (edge_density + texture_complexity) / 2

            return total_complexity, roi_complexity

        selected_indices = [0, len(frames)//2, -1]
        selected_frames = [frames[i] for i in selected_indices]
        selected_rois = [rois[i] if rois else None for i in selected_indices]

        total_complexities = []
        roi_complexities = []

        for frame, frame_rois in zip(selected_frames, selected_rois):
            current_rois = frame_rois if frame_rois else []
            
            total_c, roi_c = process_frame_both(frame, current_rois)
            total_complexities.append(total_c)
            roi_complexities.append(roi_c)

        avg_total_complexity = sum(total_complexities) / len(total_complexities)
        avg_roi_complexity = sum(roi_complexities) / len(roi_complexities)

        # 乘以帧数来估算所有帧的复杂度
        total_complexity = avg_total_complexity * len(frames)
        roi_complexity = avg_roi_complexity * len(frames)

        return total_complexity, roi_complexity

    @staticmethod
    def generate_yuv_temp_path(source_id, task_id):
        return f'video_source_{source_id}_task_{task_id}_tmp.yuv'
    
    @staticmethod
    def init_yuv_temp_path(file_path, frame_buffer):
        import mmap
        import cv2

        batch_size = len(frame_buffer)
        height, width, _ = frame_buffer[0].shape
        yuv_size = width * height + 2 * (width // 2) * (height // 2)
        with open(file_path, 'wb') as f:
            f.write(b'\x00' * yuv_size * batch_size)

        with open(file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)

            try:
                for i, frame in enumerate(frame_buffer):
                    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                    mm.seek(i * yuv_size)
                    mm.write(yuv_frame)
            finally:
                mm.close()
        return
    @staticmethod
    def generate_file_path(source_id, task_id):
        return f'video_source_{source_id}_task_{task_id}.h264'
    
    @staticmethod
    def generate_roi_path(source_id, task_id):
        return f'roi_{source_id}_task_{task_id}.txt'

    @staticmethod
    def get_bandwidth():
        return 1000
    
    @staticmethod
    def estimate_performace_with_qp(performace_gt, cqp, delta_qp1, delta_qp_2, percent1, percent2):
        """预测生成的文件大小"""
        beta = [-1.75827058,  0.775,      54.54545455]
        file_size_ori = performace_gt[cqp]['file_size']
        file_size_enhance = performace_gt[cqp + delta_qp1]['file_size']
        file_size = (beta[0] * file_size_ori + beta[1] * file_size_enhance + beta[2] * percent1) * 2
        return file_size

    def adjust_qp(self, performace_gt,total_complexity, roi_complexity, agent, past_qp, latency=0.8):
        chosen_qp = 45

        try:
            # bandwidth = get_bandwidth_from_packet()  # 从内存中获取带宽
            bandwidth = self.get_bandwidth()  # 从内存中获取带宽
            
            new_state = State(
                total_complexity=total_complexity,  # 示例：全帧复杂度
                roi_complexity=roi_complexity,    # 示例：ROI复杂度
                bandwidth=bandwidth,          # 当前带宽（kbps）
                past_qp=past_qp,  # 历史QP
                latency=self.past_latency,            # 当前总时延（秒）
                acc=self.past_acc                # 当前精度
            )
            action = agent.choose_action(new_state.to_array())
            action_space = list(range(30,52))
            chosen_qp = action_space[action]
            percentage = roi_complexity / total_complexity
            file_size = self.estimate_performace_with_qp(performace_gt, chosen_qp, -10, -5, percentage, percentage)
            transmission_time = file_size * 8 / bandwidth / 1000

            # simulate
            self.past_latency = 0.37 + transmission_time + 0.35

            self.past_acc = performace_gt[chosen_qp]['accuracy']
            self.past_qp = chosen_qp

        except ValueError as e:
            LOGGER.error(f"选择状态失败: {e}")
        return chosen_qp

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.9, lr=0.001, epsilon=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.update_target_freq = 10
        self.steps = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q(s, a) for the current state
        q_values = self.model(states).gather(1, actions).squeeze()

        # Q(s', a') for the next state using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        
        # Bellman equation
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.steps % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.steps += 1

class State:
    def __init__(self, total_complexity, roi_complexity, bandwidth, past_qp, latency, acc):
        self.total_complexity = total_complexity 
        self.roi_complexity = roi_complexity     
        self.bandwidth = bandwidth           
        self.past_qp = past_qp                
        self.latency = latency              
        self.acc = acc                         
    
    def to_array(self):
        # 将状态转化为数组
        return np.array([self.total_complexity, self.roi_complexity, self.bandwidth, self.past_qp, self.latency, self.acc])