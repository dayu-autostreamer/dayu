import abc
import random
from core.lib.common import ClassFactory, ClassType, Context, ConfigLoader, LOGGER
from core.lib.estimation import OverheadEstimator
from core.lib.scheduling import materialize_offloading_plan, service_names
from core.lib.scheduling.live_state import (
    active_deployment_for_dag,
    active_targets,
    live_resources,
)

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

    def get_bandwidth(self, source_device=None, resource_table=None):
        """Read bandwidth from one revision-consistent resource snapshot."""
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

    @staticmethod
    def _normalize_usage(value, default=0.5):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return float(default)
        value = float(value)
        if value > 1.0:
            value /= 100.0
        return min(1.0, max(0.0, value))

    def get_edge_device_loads(self, all_edge_devices, resource_table=None):
        """Get normalized CPU/memory load from one LIVE snapshot."""
        resource_table = resource_table or {}
        device_loads = {}
        
        for device in all_edge_devices:
            if device not in resource_table:
                continue
            
            resource = resource_table[device]
            if not isinstance(resource, dict):
                continue
            
            cpu_usage = self._normalize_usage(resource.get('cpu_usage', 0.5))
            memory_usage = self._normalize_usage(resource.get('memory_usage', 0.5))
            avg_load = (cpu_usage + memory_usage) / 2.0
            device_loads[device] = avg_load
        
        return device_loads

    def select_device_by_load(self, all_edge_devices, resource_table=None):
        """根据设备负载概率选择设备（负载越低，被选中的概率越高）"""
        device_loads = self.get_edge_device_loads(
            all_edge_devices,
            resource_table=resource_table,
        )
        
        if not device_loads:
            # 如果没有负载信息，随机选择一个边缘设备
            return random.choice(all_edge_devices) if all_edge_devices else self.cloud_device
        
        # 计算反负载（1 - load），负载越低，反负载越高
        inverse_loads = {
            device: max(0.0, 1.0 - load)
            for device, load in device_loads.items()
        }
        
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
        if self.default_configuration is None:
            raise ValueError('DynamicAgent requires a configuration mapping')

        with self.overhead_estimator:
            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            all_edge_devices = [str(device) for device in info['all_edge_devices']]
            dag = info['dag']
            snapshot, deployment = active_deployment_for_dag(self.system, dag)
            resources = live_resources(snapshot)

            bandwidth = self.get_bandwidth(
                source_device=source_edge_device,
                resource_table=resources,
            )
            LOGGER.info(f'[Dynamic Agent] Current bandwidth: {bandwidth}, threshold: {self.bandwidth_threshold}')

            # ``proposed`` retains the algorithm's unconstrained decision for
            # the redeployment hook. ``served`` is the feasible projection on
            # the exact LIVE replicas used by this task.
            proposed = {}
            served = {}
            for service_name in service_names(dag):
                active = active_targets(deployment, service_name)
                if bandwidth is not None and bandwidth > self.bandwidth_threshold:
                    desired = cloud_device
                else:
                    desired = self.select_device_by_load(
                        all_edge_devices,
                        resource_table=resources,
                    ) if all_edge_devices else cloud_device
                proposed[service_name] = desired

                if desired in active:
                    served[service_name] = desired
                    continue
                active_edges = active_targets(
                    deployment,
                    service_name,
                    candidates=all_edge_devices,
                )
                if active_edges:
                    served[service_name] = self.select_device_by_load(
                        active_edges,
                        resource_table=resources,
                    )
                elif cloud_device in active:
                    served[service_name] = cloud_device
                else:
                    served[service_name] = active[0]
                LOGGER.info(
                    f'[Dynamic Agent] Proposed target {desired!r} for '
                    f'{service_name!r} is pending deployment; serving '
                    f'{served[service_name]!r} from LIVE revision '
                    f'{snapshot["runtime_directory_revision"]}'
                )

            self.latest_offloading_policy = proposed.copy()
            LOGGER.info(
                f'[Dynamic Agent] Proposed deployment targets: {proposed}; '
                f'LIVE task targets: {served}'
            )
            policy = materialize_offloading_plan(
                self.default_configuration,
                dag,
                served,
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
        return self.overhead_estimator.get_latest_overhead()
