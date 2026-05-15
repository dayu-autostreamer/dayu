import numpy as np
import threading
from collections import deque


class StateBuffer:
    """状态缓冲区：收集和管理环境状态信息"""
    
    def __init__(self, num_services, num_devices, delay_window_size=10):
        """
        Args:
            num_services: 服务数量
            num_devices: 设备数量
            delay_window_size: 用于计算平均时延的窗口大小
        """
        self.num_services = num_services
        self.num_devices = num_devices
        self.delay_window_size = delay_window_size
        
        # 资源信息缓冲
        self.cpu_loads = {}  # device -> cpu load (%)
        self.memory_loads = {}  # device -> memory load (%)
        self.bandwidths = {}  # device -> bandwidth (Mbps)
        
        # 时延缓冲
        self.delay_buffer = deque(maxlen=delay_window_size)
        
        # 当前部署方案
        self.current_deployment = None  # (num_services,) array, 每个元素是设备id
        
        # 线程锁
        self.lock = threading.Lock()
        
    def update_resource(self, device, cpu_load, memory_load, bandwidth):
        """
        更新设备资源信息
        
        Args:
            device: 设备标识
            cpu_load: CPU负载百分比 (0-100)
            memory_load: 内存负载百分比 (0-100)
            bandwidth: 网络带宽 (Mbps)
        """
        with self.lock:
            self.cpu_loads[device] = cpu_load
            self.memory_loads[device] = memory_load
            self.bandwidths[device] = bandwidth
    
    def update_delay(self, delay):
        """
        更新任务时延
        
        Args:
            delay: 任务端到端时延 (秒)
        """
        with self.lock:
            self.delay_buffer.append(delay)
    
    def update_deployment(self, deployment):
        """
        更新当前部署方案
        
        Args:
            deployment: (num_services,) numpy array 或 dict {service_name: device}
        """
        with self.lock:
            if isinstance(deployment, dict):
                # 转换为array格式
                self.current_deployment = np.array([deployment.get(i, 0) 
                                                    for i in range(self.num_services)])
            else:
                self.current_deployment = deployment
    
    def get_state(self, device_list):
        """
        获取当前状态矩阵
        
        Args:
            device_list: 设备列表，按顺序对应矩阵的列
            
        Returns:
            state: (num_services+4, num_devices) numpy array
                   前num_services行是one-hot编码的部署情况
                   后4行分别是CPU负载、内存负载、带宽、时延
        """
        with self.lock:
            state = np.zeros((self.num_services + 4, self.num_devices), dtype=np.float32)
            
            # 前num_services行：one-hot编码的部署情况
            if self.current_deployment is not None:
                for service_id in range(self.num_services):
                    device_id = int(self.current_deployment[service_id])
                    if 0 <= device_id < self.num_devices:
                        state[service_id, device_id] = 1.0
            
            # 获取各设备的资源信息
            cpu_load_row = np.zeros(self.num_devices)
            memory_load_row = np.zeros(self.num_devices)
            bandwidth_row = np.zeros(self.num_devices)
            
            for idx, device in enumerate(device_list):
                if device in self.cpu_loads:
                    # 归一化：CPU和内存负载除以100
                    cpu_load_row[idx] = self.cpu_loads[device] / 100.0
                    memory_load_row[idx] = self.memory_loads[device] / 100.0
            
            # 带宽处理：找到第一个非0值，赋给所有设备
            bandwidth_value = 0.0
            for idx, device in enumerate(device_list):
                if device in self.bandwidths and self.bandwidths[device] > 0:
                    bandwidth_value = self.bandwidths[device]
                    break
            
            # 归一化：带宽除以10
            bandwidth_row[:] = bandwidth_value / 10.0
            
            # 时延处理：计算平均值，所有设备相同
            avg_delay = 0.0
            if len(self.delay_buffer) > 0:
                avg_delay = np.mean(list(self.delay_buffer))
            
            # 归一化：时延除以5
            delay_row = np.full(self.num_devices, avg_delay / 5.0)
            
            # 填充后4行
            state[self.num_services] = cpu_load_row
            state[self.num_services + 1] = memory_load_row
            state[self.num_services + 2] = bandwidth_row
            state[self.num_services + 3] = delay_row
            
            return state
    
    def get_average_delay(self):
        """获取近期平均时延"""
        with self.lock:
            if len(self.delay_buffer) > 0:
                return np.mean(list(self.delay_buffer))
            return 0.0
    
    def is_ready(self):
        """
        检查是否有足够的数据生成状态
        只需要有部署方案即可，资源信息可以使用默认值（0）
        """
        with self.lock:
            has_deployment = self.current_deployment is not None
            return has_deployment

