import abc
import random
from core.lib.common import ClassFactory, ClassType, Context, ConfigLoader, LOGGER
from core.lib.estimation import OverheadEstimator
from core.lib.scheduling import materialize_offloading_plan, service_names, service_waiting_count
from core.lib.scheduling.live_state import (
    active_deployment_for_dag,
    active_targets,
    live_resources,
)

from .base_agent import BaseAgent

__all__ = ('OfflineProfilingAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='offline_profiling')
class OfflineProfilingAgent(BaseAgent, abc.ABC):
    """
    Offline Profiling Agent that selects execution devices based on bandwidth and offline profiled latency data.
    - If bandwidth > threshold n, all stages execute on cloud
    - If bandwidth <= threshold n, probabilistically select edge devices using weighted latency
      (latency * service_importance_weight); lower effective latency -> higher probability
    - Resource `queue_state[service].waiting_count` per device: if > 5, that device's
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
        """从 scheduler 的结构化 queue state 中读取服务等待数量。"""
        return service_waiting_count(resource, service_name)

    def _device_queue_length(self, device: str, service_name: str, resource_table=None) -> float:
        resource_table = resource_table or {}
        if not resource_table or device not in resource_table:
            return 0.0
        return self._resource_queue_length(resource_table[device], str(service_name))

    def get_bandwidth(self, source_device=None, resource_table=None):
        """从resource_table获取带宽值"""
        resource_table = resource_table or {}
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

    def select_device_by_latency(self, service_name, deployed_devices, resource_table=None):
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
            LOGGER.warning('[Offline Profiling Agent] No latency data for any deployed device, random selection')
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
            ql = self._device_queue_length(
                device,
                service_name,
                resource_table=resource_table,
            )
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
            raise ValueError('OfflineProfilingAgent requires a configuration mapping')

        with self.overhead_estimator:
            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            all_edge_devices = [str(device) for device in info['all_edge_devices']]
            dag = info['dag']
            snapshot, current_deployment = active_deployment_for_dag(
                self.system,
                dag,
            )
            resources = live_resources(snapshot)

            bandwidth = self.get_bandwidth(
                source_device=source_edge_device,
                resource_table=resources,
            )
            LOGGER.info(f'[Offline Profiling Agent] Current bandwidth: {bandwidth}, threshold: {self.bandwidth_threshold}')
            LOGGER.info(f'[Offline Profiling Agent] Current deployment: {current_deployment}')

            offloading_policy = {}
            for service_name in service_names(dag):
                active = active_targets(current_deployment, service_name)
                if bandwidth is not None and bandwidth > self.bandwidth_threshold:
                    if cloud_device not in active:
                        raise ValueError(
                            f'OfflineProfilingAgent selected cloud for {service_name!r}, '
                            'but the LIVE deployment has no cloud replica; enable '
                            'include_cloud in its deployment hooks'
                        )
                    execute_device = cloud_device
                else:
                    deployed_edges = active_targets(
                        current_deployment,
                        service_name,
                        candidates=all_edge_devices,
                    )
                    if deployed_edges:
                        execute_device = self.select_device_by_latency(
                            service_name,
                            deployed_edges,
                            resource_table=resources,
                        )
                    elif cloud_device in active:
                        execute_device = cloud_device
                    else:
                        execute_device = active[0]
                offloading_policy[service_name] = execute_device

            self.latest_offloading_policy = offloading_policy.copy()
            LOGGER.info(f'[Offline Profiling Agent] Latest offloading policy: {offloading_policy}')
            policy = materialize_offloading_plan(
                self.default_configuration,
                dag,
                offloading_policy,
                source_device=source_edge_device,
                cloud_device=cloud_device,
            )
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
