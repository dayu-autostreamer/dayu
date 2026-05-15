import abc
import random
from core.lib.common import ClassFactory, ClassType, KubeConfig, Context, ConfigLoader, TaskConstant, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('DynamicAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='dynamic')
class DynamicAgent(BaseAgent, abc.ABC):
    """
    Dynamic Agent that selects execution devices based on bandwidth and device load.
    - If bandwidth > threshold n, all stages execute on cloud
    - If bandwidth <= threshold n, dynamically select edge devices based on load (lower load = higher probability)
    """

    def __init__(self, system, agent_id: int, configuration=None, bandwidth_threshold=None):
        super().__init__(system, agent_id)

        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.system = system

        if configuration is None or isinstance(configuration, dict):
            self.default_configuration = configuration
        elif isinstance(configuration, str):
            self.default_configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        # 带宽阈值参数
        if bandwidth_threshold is None:
            self.bandwidth_threshold = 5.0  # 默认值
        elif isinstance(bandwidth_threshold, (int, float)):
            self.bandwidth_threshold = float(bandwidth_threshold)
        else:
            raise TypeError(f'Input "bandwidth_threshold" must be of type int or float, get type {type(bandwidth_threshold)}')

        self.latest_offloading_policy = {}  # 存储最新的offloading策略，供重部署策略使用
        self.overhead_estimator = OverheadEstimator('Dynamic', 'scheduler/dynamic', agent_id=self.agent_id)

    def get_bandwidth(self, source_device=None):
        """从resource_table获取带宽值"""
        resource_table = self.system.get_scheduler_resource()
        if not resource_table:
            return None
        
        # 优先从source设备获取带宽
        if source_device and source_device in resource_table:
            resource = resource_table[source_device]
            if isinstance(resource, dict) and 'available_bandwidth' in resource:
                bandwidth = resource['available_bandwidth']
                if bandwidth != -1 and bandwidth != 0:
                    return bandwidth
        
        # 尝试从所有设备获取带宽（找到第一个有效的）
        for device, resource in resource_table.items():
            if isinstance(resource, dict) and 'available_bandwidth' in resource:
                bandwidth = resource['available_bandwidth']
                if bandwidth != -1 and bandwidth != 0:
                    return bandwidth
        return None

    def get_edge_device_loads(self, all_edge_devices):
        """获取所有边缘设备的负载信息"""
        resource_table = self.system.get_scheduler_resource()
        device_loads = {}
        
        for device in all_edge_devices:
            if device not in resource_table:
                continue
            
            resource = resource_table[device]
            if not isinstance(resource, dict):
                continue
            
            cpu_usage = resource.get('cpu_usage', 0.5)  # 默认值0.5
            memory_usage = resource.get('memory_usage', 0.5)  # 默认值0.5
            
            # 确保是数值类型
            if not isinstance(cpu_usage, (int, float)):
                cpu_usage = 0.5
            if not isinstance(memory_usage, (int, float)):
                memory_usage = 0.5
            
            # 取CPU和内存负载的平均值
            avg_load = (cpu_usage + memory_usage) / 2.0
            device_loads[device] = avg_load
        
        return device_loads

    def select_device_by_load(self, all_edge_devices):
        """根据设备负载概率选择设备（负载越低，被选中的概率越高）"""
        device_loads = self.get_edge_device_loads(all_edge_devices)
        
        if not device_loads:
            # 如果没有负载信息，随机选择一个边缘设备
            return random.choice(all_edge_devices) if all_edge_devices else self.cloud_device
        
        # 计算反负载（1 - load），负载越低，反负载越高
        inverse_loads = {device: 1.0 - load for device, load in device_loads.items()}
        
        # 归一化概率（使用softmax-like归一化）
        total_inverse = sum(inverse_loads.values())
        if total_inverse == 0:
            # 如果所有设备负载都是1，则随机选择
            return random.choice(list(device_loads.keys())) if device_loads else self.cloud_device
        
        probabilities = {device: inv_load / total_inverse for device, inv_load in inverse_loads.items()}
        
        # 根据概率选择设备
        devices = list(probabilities.keys())
        probs = list(probabilities.values())
        selected_device = random.choices(devices, weights=probs, k=1)[0]
        
        LOGGER.debug(f'[Dynamic Agent] Device selection probabilities: {probabilities}, selected: {selected_device}')
        
        return selected_device

    def get_schedule_plan(self, info):
        print("-------------------------------------")
        if self.default_configuration is None:
            return None

        with self.overhead_estimator:
            configuration = self.default_configuration.copy()
            policy = {}
            policy.update(configuration)
            
            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            all_edge_devices = info['all_edge_devices']
            dag = info['dag']
            
            # 获取当前带宽
            bandwidth = self.get_bandwidth(source_device=source_edge_device)
            LOGGER.info(f'[Dynamic Agent] Current bandwidth: {bandwidth}, threshold: {self.bandwidth_threshold}')
            
            # 存储offloading策略
            offloading_policy = {}
            
            # 对每个服务进行判断
            for service_name in dag:
                # _end节点固定在cloud.kubeedge（即cloud_device）
                if service_name == '_end':
                    dag[service_name]['service']['execute_device'] = cloud_device
                    offloading_policy[service_name] = cloud_device
                    continue

                if service_name == TaskConstant.START.value:
                    # START节点固定在source设备
                    dag[service_name]['service']['execute_device'] = source_edge_device
                    offloading_policy[service_name] = source_edge_device
                    continue
                
                # 如果带宽大于阈值，所有阶段都交给云端执行
                if bandwidth is not None and bandwidth > self.bandwidth_threshold:
                    execute_device = cloud_device
                else:
                    # 根据设备负载动态选择边缘设备
                    if all_edge_devices:
                        execute_device = self.select_device_by_load(all_edge_devices)
                    else:
                        execute_device = cloud_device
                
                dag[service_name]['service']['execute_device'] = execute_device
                offloading_policy[service_name] = execute_device
            
            # 保存最新的offloading策略供重部署策略使用
            self.latest_offloading_policy = offloading_policy.copy()
            LOGGER.info(f'[Dynamic Agent] Latest offloading policy: {offloading_policy}')
            
            policy.update({'dag': dag})
        return policy

    def get_latest_offloading_policy(self):
        """获取最新的offloading策略，供重部署策略使用"""
        return self.latest_offloading_policy.copy()

    def run(self):
        pass

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()
