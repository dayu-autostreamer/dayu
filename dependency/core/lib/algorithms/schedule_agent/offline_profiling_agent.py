import abc
import random
from core.lib.common import ClassFactory, ClassType, KubeConfig, Context, ConfigLoader, TaskConstant, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('OfflineProfilingAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='offline_profiling')
class OfflineProfilingAgent(BaseAgent, abc.ABC):
    """
    Offline Profiling Agent that selects execution devices based on bandwidth and offline profiled latency data.
    - If bandwidth > threshold n, all stages execute on cloud
    - If bandwidth <= threshold n, probabilistically select edge devices using weighted latency
      (latency * service_importance_weight); lower effective latency -> higher probability
    - Resource `queue_length` per device (scalar or per-service dict): if > 5, that device's
      relative score is halved before probabilities are re-normalized.
    """

    _OFFLOAD_QUEUE_LEN_HALVE_GT = 5
    _OFFLOAD_QUEUE_LEN_HALVE_FACTOR = 0.5

    def __init__(self, system, agent_id: int, configuration=None, bandwidth_threshold=None, latency_profile=None,
                 service_importance_weights=None):
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

        # 离线测得的时延数据
        # 格式: {service_name: {device_name: latency}}
        if latency_profile is None:
            self.latency_profile = {}
        elif isinstance(latency_profile, dict):
            self.latency_profile = latency_profile
        elif isinstance(latency_profile, str):
            self.latency_profile = ConfigLoader.load(Context.get_file_path(latency_profile))
        else:
            raise TypeError(f'Input "latency_profile" must be of type str or dict, get type {type(latency_profile)}')

        # 服务重要性权重，与 profiled latency 相乘后参与卸载设备概率决策；未配置的服务默认为 1.0
        if service_importance_weights is None:
            self.service_importance_weights = {}
        elif isinstance(service_importance_weights, dict):
            self.service_importance_weights = {str(k): float(v) for k, v in service_importance_weights.items()}
        elif isinstance(service_importance_weights, str):
            self.service_importance_weights = {
                str(k): float(v) for k, v in ConfigLoader.load(Context.get_file_path(service_importance_weights)).items()
            }
        else:
            raise TypeError(
                f'Input "service_importance_weights" must be of type str or dict, get type {type(service_importance_weights)}'
            )

        self.latest_offloading_policy = {}  # 存储最新的offloading策略，供重部署策略使用
        self.overhead_estimator = OverheadEstimator('OfflineProfiling', 'scheduler/offline_profiling', agent_id=self.agent_id)

        LOGGER.info(f'[Offline Profiling Agent] Initialized with bandwidth threshold: {self.bandwidth_threshold}')
        LOGGER.info(f'[Offline Profiling Agent] Latency profile: {self.latency_profile}')
        LOGGER.info(f'[Offline Profiling Agent] Service importance weights: {self.service_importance_weights}')

    def _importance_weight(self, service_name: str) -> float:
        w = self.service_importance_weights.get(str(service_name), 1.0)
        return float(w) if w > 0 else 1.0

    @staticmethod
    def _resource_queue_length(resource, service_name: str) -> float:
        """从 scheduler resource 条目解析队列长度：标量或 {服务: 长度}；缺省为 0。"""
        if not isinstance(resource, dict):
            return 0.0
        ql = resource.get('queue_length')
        if ql is None:
            return 0.0
        if isinstance(ql, (int, float)):
            return float(ql)
        if isinstance(ql, dict):
            if service_name in ql:
                try:
                    return float(ql[service_name])
                except (TypeError, ValueError):
                    pass
            vals = []
            for v in ql.values():
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            return max(vals) if vals else 0.0
        return 0.0

    def _device_queue_length(self, device: str, service_name: str) -> float:
        resource_table = self.system.get_scheduler_resource()
        if not resource_table or device not in resource_table:
            return 0.0
        return self._resource_queue_length(resource_table[device], str(service_name))

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

    def get_current_deployment(self):
        """获取当前的服务部署情况"""
        # 从redeployment_policy获取当前的部署计划
        if hasattr(self.redeployment_policy, 'policy') and self.redeployment_policy.policy:
            return self.redeployment_policy.policy
        return {}

    def select_device_by_latency(self, service_name, deployed_devices):
        """
        根据加权执行代价 (latency * importance_weight) 并结合各边 queue_length 概率选择设备；
        某边 queue_length > 5 时该边得分先减半，再对所有候选边归一化。
        加权代价越小，被选中的相对概率越高。
        
        Args:
            service_name: 服务名称
            deployed_devices: 该服务当前部署的设备列表
            
        Returns:
            选中的设备
        """
        if not deployed_devices:
            LOGGER.warning(f'[Offline Profiling Agent] Service {service_name} has no deployed devices, using cloud')
            return self.cloud_device
        
        # 如果该服务没有latency profile，随机选择
        if service_name not in self.latency_profile:
            LOGGER.warning(f'[Offline Profiling Agent] Service {service_name} has no latency profile, random selection')
            return random.choice(deployed_devices)
        
        service_latency = self.latency_profile[service_name]
        
        # 获取已部署设备的时延
        device_latencies = {}
        for device in deployed_devices:
            if device in service_latency:
                device_latencies[device] = service_latency[device]
            else:
                LOGGER.warning(f'[Offline Profiling Agent] Device {device} has no latency data for service {service_name}')
        
        # 如果没有任何设备有latency数据，随机选择
        if not device_latencies:
            LOGGER.warning(f'[Offline Profiling Agent] No latency data for any deployed device, random selection')
            return random.choice(deployed_devices)
        
        weight = self._importance_weight(service_name)
        # 加权代价 = latency * weight，基础得分 ∝ 1/代价
        base_scores = {}
        for device, latency in device_latencies.items():
            cost = (latency * weight) if latency > 0 else 1e-9
            cost = max(cost, 1e-12)
            base_scores[device] = 1.0 / cost

        adjusted_scores = {}
        for device, base in base_scores.items():
            ql = self._device_queue_length(device, service_name)
            factor = (
                self._OFFLOAD_QUEUE_LEN_HALVE_FACTOR
                if ql > self._OFFLOAD_QUEUE_LEN_HALVE_GT
                else 1.0
            )
            adjusted_scores[device] = base * factor

        total_score = sum(adjusted_scores.values())
        if total_score == 0:
            return random.choice(list(device_latencies.keys()))

        probabilities = {device: s / total_score for device, s in adjusted_scores.items()}
        
        # 根据概率选择设备
        devices = list(probabilities.keys())
        probs = list(probabilities.values())
        selected_device = random.choices(devices, weights=probs, k=1)[0]
        
        LOGGER.debug(f'[Offline Profiling Agent] Service {service_name} selection probabilities: {probabilities}, selected: {selected_device}')
        
        return selected_device

    def get_schedule_plan(self, info):
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
            LOGGER.info(f'[Offline Profiling Agent] Current bandwidth: {bandwidth}, threshold: {self.bandwidth_threshold}')
            
            # 获取当前服务部署情况
            current_deployment = self.get_current_deployment()
            LOGGER.info(f'[Offline Profiling Agent] Current deployment: {current_deployment}')
            
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
                    # 根据当前部署情况和离线时延数据进行概率选择
                    deployed_devices = current_deployment.get(service_name, [])
                    
                    if deployed_devices:
                        execute_device = self.select_device_by_latency(service_name, deployed_devices)
                    else:
                        # 如果没有部署信息，则在所有边缘设备中选择
                        if all_edge_devices:
                            execute_device = self.select_device_by_latency(service_name, all_edge_devices)
                        else:
                            execute_device = cloud_device
                
                dag[service_name]['service']['execute_device'] = execute_device
                offloading_policy[service_name] = execute_device
            
            # 保存最新的offloading策略供重部署策略使用
            self.latest_offloading_policy = offloading_policy.copy()
            LOGGER.info(f'[Offline Profiling Agent] Latest offloading policy: {offloading_policy}')
            
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
        offloading_overhead = self.overhead_estimator.get_latest_overhead()
        redeployment_overhead = 0.0
        if hasattr(self.redeployment_policy, 'get_redeployment_overhead'):
            redeployment_overhead = self.redeployment_policy.get_redeployment_overhead()
        return offloading_overhead + redeployment_overhead

