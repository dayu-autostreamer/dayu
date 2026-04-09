import abc
import copy

import numpy as np

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager, ConfigLoader, Context, LOGGER, \
    KubeConfig, TaskConstant
from core.lib.content import Task
from core.lib.algorithms.shared.hedger import Hedger

from .base_agent import BaseAgent

__all__ = ('HedgerAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hedger')
class HedgerAgent(BaseAgent, abc.ABC):
    def __init__(self, system, agent_id: int, configuration=None, offloading=None):
        super().__init__(system, agent_id)

        self.system = system
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device

        self.default_configuration = None
        self.default_offloading = None
        self.load_default_policy(configuration, offloading)

        self.hedger = None
        self.register_hedger(hedger_id=f'hedger_{self.agent_id}')

    def register_hedger(self, hedger_id='hedger'):
        if self.hedger is None:
            network_params = self.system.network_params.copy()
            hyper_params = self.system.hyper_params.copy()
            agent_params = self.system.agent_params.copy()
            self.hedger = GlobalInstanceManager.get_instance(
                Hedger, hedger_id,
                network_params=network_params,
                hyper_params=hyper_params,
                agent_params=agent_params
            )
            self.hedger.register_offloading_agent()

    @staticmethod
    def _normalize_mapping(data):
        return copy.deepcopy(data) if isinstance(data, dict) else {}

    @staticmethod
    def _extract_task_complexity(service) -> float:
        """Aggregate the processor-produced `obj_num` signal into one complexity value."""
        scenario = service.get_scenario_data()
        if not isinstance(scenario, dict) or 'obj_num' not in scenario:
            return 0.0

        obj_num = scenario['obj_num']
        if obj_num is None:
            return 0.0
        if isinstance(obj_num, (int, float, np.number)):
            return float(obj_num)

        try:
            obj_num_array = np.asarray(obj_num, dtype=float)
        except (TypeError, ValueError):
            return 0.0

        if obj_num_array.size == 0:
            return 0.0
        return float(np.mean(obj_num_array.reshape(-1)))

    def get_schedule_plan(self, info):
        source_id = info['source']['id']
        source_edge_device = info['source_device']
        all_edge_devices = info['all_edge_devices']
        cloud_device = self.cloud_device
        all_devices = [*all_edge_devices, cloud_device]
        dag = info['dag']

        self.hedger.register_logical_topology(Task.extract_dag_from_dag_deployment(dag))
        self.hedger.register_physical_topology(all_edge_devices, source_edge_device)
        self.hedger.register_state_buffer()

        configuration = self._normalize_mapping(self.default_configuration)
        offloading = self._normalize_mapping(self.hedger.get_offloading_plan())
        if not offloading:
            offloading = self._normalize_mapping(self.default_offloading)
            LOGGER.warning('No offloading plan from Hedger, use default offloading policy.')

        policy = {}
        policy.update(configuration)
        service_info = KubeConfig.get_service_nodes_dict()

        for service_name in dag:
            if service_name in service_info and service_name in offloading \
                    and offloading[service_name] in all_devices:
                dag[service_name]['service']['execute_device'] = offloading[service_name]
            elif service_name == TaskConstant.START.value:
                dag[service_name]['service']['execute_device'] = source_edge_device
            else:
                dag[service_name]['service']['execute_device'] = cloud_device

        policy.update({'dag': dag})

        LOGGER.info(f'[Offloading] (source {source_id}) Schedule policy: {policy}')
        return policy

    def run(self):
        pass

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        if self.hedger.state_buffer is None:
            return
        if not isinstance(resource, dict):
            return

        bandwidth = resource.get('bandwidth')
        if isinstance(bandwidth, dict):
            self.hedger.state_buffer.add_bandwidths(bandwidth)

        gpu_flops = resource.get('gpu_flops')
        if gpu_flops is not None:
            self.hedger.state_buffer.add_gpu_flops(device, gpu_flops)

        memory_capacity = resource.get('memory_capacity')
        if memory_capacity is not None:
            self.hedger.state_buffer.add_memory_capacity(device, memory_capacity)

        gpu_usage = resource.get('gpu_usage')
        if gpu_usage is not None:
            self.hedger.state_buffer.add_gpu_utilization(device, gpu_usage)

        memory_usage = resource.get('memory_usage')
        if memory_usage is not None:
            self.hedger.state_buffer.add_memory_utilization(device, memory_usage)

        for service, flops in (resource.get('model_flops') or {}).items():
            self.hedger.state_buffer.add_model_flops(service, flops)
        for service, memory in (resource.get('model_memory') or {}).items():
            self.hedger.state_buffer.add_model_memory(service, memory)

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        if self.hedger.state_buffer is None:
            return
        for service_name in task.get_dag().nodes:
            if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                continue
            service = task.get_service(service_name)
            latency = service.get_execute_time()
            complexity = self._extract_task_complexity(service)
            self.hedger.state_buffer.add_task_complexity(service_name, complexity)
            self.hedger.state_buffer.add_task_latency(service_name, latency)

    def get_schedule_overhead(self):
        # TODO
        return 0

    def load_default_policy(self, configuration, offloading):
        if configuration is None or isinstance(configuration, dict):
            self.default_configuration = configuration
        elif isinstance(configuration, str):
            self.default_configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        if offloading is None or isinstance(offloading, dict):
            self.default_offloading = offloading
        elif isinstance(offloading, str):
            self.default_offloading = ConfigLoader.load(Context.get_file_path(offloading))
        else:
            raise TypeError(f'Input "offloading" must be of type str or dict, get type {type(offloading)}')
