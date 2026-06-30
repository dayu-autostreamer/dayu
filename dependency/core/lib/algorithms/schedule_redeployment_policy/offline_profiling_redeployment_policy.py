import abc
import copy

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, KubeConfig, ConfigLoader, Context
from core.lib.estimation import OverheadEstimator

__all__ = ('OfflineProfilingRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='offline_profiling')
class OfflineProfilingRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    """
    Offline Profiling Redeployment Policy that deploys services greedily based on offline profiled latency data.
    - Task complexity: mean(latency * service_importance_weight) across devices
    - Device capability: mean(latency * service_importance_weight) over services on that device (lower is better)
    - Greedy deployment: harder (higher weighted-cost) tasks to more capable devices
    - Constraints:
      - Each device has a maximum number of services it can host
      - Each service can be deployed on a maximum number of devices
      - No duplicate service on the same device
    """

    def __init__(self, system, agent_id, latency_profile=None, device_service_limits=None, 
                 service_replica_count=None, default_service_limit=None, default_replica_count=None,
                 service_importance_weights=None, **kwargs):

        self.system = system
        self.agent_id = agent_id

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

        # 设备服务数量限制配置
        # 格式: {device_name: max_service_count}
        if device_service_limits is None:
            self.device_service_limits = {}
        elif isinstance(device_service_limits, dict):
            self.device_service_limits = {str(device): int(limit) for device, limit in device_service_limits.items()}
        else:
            raise TypeError(f'Input "device_service_limits" must be of type dict, get type {type(device_service_limits)}')

        # 服务可部署的设备数量限制
        # 格式: {service_name: max_device_count}
        if service_replica_count is None:
            self.service_replica_count = {}
        elif isinstance(service_replica_count, dict):
            self.service_replica_count = {str(service): int(count) for service, count in service_replica_count.items()}
        else:
            raise TypeError(f'Input "service_replica_count" must be of type dict, get type {type(service_replica_count)}')

        # 默认服务数量限制（每个设备）
        if default_service_limit is None:
            self.default_service_limit = 2
        elif isinstance(default_service_limit, (int, float)):
            self.default_service_limit = int(default_service_limit)
        else:
            raise TypeError(f'Input "default_service_limit" must be of type int or float, get type {type(default_service_limit)}')

        self.policy = None

        # 默认副本数量限制（每个服务）
        if default_replica_count is None:
            self.default_replica_count = 2
        elif isinstance(default_replica_count, (int, float)):
            self.default_replica_count = int(default_replica_count)
        else:
            raise TypeError(f'Input "default_replica_count" must be of type int or float, get type {type(default_replica_count)}')

        LOGGER.info(f'[Offline Profiling Redeployment] Initialized with latency profile: {self.latency_profile}')
        LOGGER.info(f'[Offline Profiling Redeployment] Service importance weights: {self.service_importance_weights}')
        LOGGER.info(f'[Offline Profiling Redeployment] Device service limits: {self.device_service_limits}, default: {self.default_service_limit}')
        LOGGER.info(f'[Offline Profiling Redeployment] Service replica count: {self.service_replica_count}, default: {self.default_replica_count}')
        self.overhead_estimator = OverheadEstimator(
            'OfflineProfilingRedeployment',
            'scheduler/offline_profiling',
            agent_id=self.agent_id,
        )

    def _importance_weight(self, service_name: str) -> float:
        w = self.service_importance_weights.get(str(service_name), 1.0)
        return float(w) if w > 0 else 1.0

    def calculate_task_complexity(self):
        """
        计算每个任务的复杂度：各设备上 (latency * importance_weight) 的平均值
        复杂度越高，值越大
        
        Returns:
            {service_name: complexity}
        """
        task_complexity = {}
        
        for service_name, device_latencies in self.latency_profile.items():
            if device_latencies:
                w = self._importance_weight(service_name)
                weighted = [lat * w for lat in device_latencies.values()]
                task_complexity[service_name] = sum(weighted) / len(weighted)
            else:
                task_complexity[service_name] = 0.0
        
        return task_complexity

    def calculate_device_capability(self, available_devices):
        """
        计算每个设备的计算能力：各任务在该设备上 (latency * importance_weight) 的平均值
        能力越强，值越小
        
        Args:
            available_devices: 可用的设备列表
            
        Returns:
            {device_name: capability_score}
        """
        device_capability = {}
        
        for device in available_devices:
            latencies = []
            for service_name, device_latencies in self.latency_profile.items():
                if device in device_latencies:
                    w = self._importance_weight(service_name)
                    latencies.append(device_latencies[device] * w)
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                device_capability[device] = avg_latency
            else:
                # 如果没有数据，设置一个较大的值（表示能力较弱）
                device_capability[device] = float('inf')
        
        return device_capability

    def get_device_service_limit(self, device_name):
        """获取指定设备的最大服务数量限制"""
        device_name = str(device_name)
        return self.device_service_limits.get(device_name, self.default_service_limit)

    def get_service_replica_count(self, service_name):
        """获取指定服务的最大副本数量限制"""
        service_name = str(service_name)
        return self.service_replica_count.get(service_name, self.default_replica_count)

    def greedy_deployment(self, dag, available_devices):
        """
        贪心部署策略：越难的任务部署到越强的设备上
        
        Args:
            dag: 任务DAG
            available_devices: 可用的边缘设备列表
            
        Returns:
            {service_name: [device1, device2, ...]}
        """
        # 计算任务复杂度和设备能力
        task_complexity = self.calculate_task_complexity()
        device_capability = self.calculate_device_capability(available_devices)
        
        # 按任务复杂度降序排序（复杂度高的优先）
        sorted_services = sorted(task_complexity.items(), key=lambda x: x[1], reverse=True)
        
        # 按设备能力升序排序（能力强的优先，即时延小的优先）
        sorted_devices = sorted(device_capability.items(), key=lambda x: x[1])
        
        LOGGER.info(f'[Offline Profiling Redeployment] Task complexity (descending): {sorted_services}')
        LOGGER.info(f'[Offline Profiling Redeployment] Device capability (ascending, lower is better): {sorted_devices}')
        
        # 初始化部署计划和设备当前服务数量
        deploy_plan = {}
        device_service_count = {device: 0 for device in available_devices}
        
        # 贪心部署
        for service_name, complexity in sorted_services:
            # 跳过不在dag中的服务
            if service_name not in dag:
                continue
            
            max_replicas = self.get_service_replica_count(service_name)
            deploy_plan[service_name] = []
            deployed_count = 0
            
            # 尝试将服务部署到能力最强的设备上
            for device_name, capability in sorted_devices:
                # 检查是否已达到该服务的最大副本数
                if deployed_count >= max_replicas:
                    break
                
                # 检查设备是否还有容量
                device_limit = self.get_device_service_limit(device_name)
                if device_service_count[device_name] >= device_limit:
                    continue
                
                # 检查该服务在该设备上是否有latency数据
                if service_name not in self.latency_profile or device_name not in self.latency_profile[service_name]:
                    LOGGER.debug(f'[Offline Profiling Redeployment] No latency data for service {service_name} on device {device_name}, skipping')
                    continue
                
                # 部署服务
                deploy_plan[service_name].append(device_name)
                device_service_count[device_name] += 1
                deployed_count += 1
                
                LOGGER.debug(f'[Offline Profiling Redeployment] Deployed service {service_name} to device {device_name} '
                           f'(complexity: {complexity:.4f}, capability: {capability:.4f}, '
                           f'device usage: {device_service_count[device_name]}/{device_limit})')
            
            # 如果该服务没有被部署到任何设备，至少部署到一个设备
            if not deploy_plan[service_name]:
                # 尝试找到第一个还有容量的设备
                for device_name, capability in sorted_devices:
                    device_limit = self.get_device_service_limit(device_name)
                    if device_service_count[device_name] < device_limit:
                        deploy_plan[service_name].append(device_name)
                        device_service_count[device_name] += 1
                        LOGGER.warning(f'[Offline Profiling Redeployment] Service {service_name} forced to device {device_name} '
                                     f'(no latency data but needed deployment)')
                        break
        
        LOGGER.info(f'[Offline Profiling Redeployment] Final device service count: {device_service_count}')
        
        return deploy_plan

    def __call__(self, info):
        """
        生成重部署计划
        基于离线profiling数据使用贪心策略进行部署
        """
        with self.overhead_estimator:
            source_id = info['source']['id']
            dag = info['dag']
            node_set = info['node_set']

            # 每次都重新计算部署计划，不读取和复用现有的policy
            # 过滤出边缘设备（排除cloud设备）
            available_devices = [node for node in node_set if node != 'cloud.kubeedge']

            if not available_devices:
                LOGGER.warning(f'[Offline Profiling Redeployment] (source {source_id}) No edge devices available')
                # 返回空的部署计划
                deploy_plan = {service_name: [] for service_name in dag if service_name not in ['_start', '_end']}
            else:
                # 执行贪心部署策略
                deploy_plan = self.greedy_deployment(dag, available_devices)

            LOGGER.info(f'[Offline Profiling Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')

            self.policy = deploy_plan

        return deploy_plan

    def get_redeployment_overhead(self):
        return self.overhead_estimator.get_latest_overhead()

