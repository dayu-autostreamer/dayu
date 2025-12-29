import abc
import copy
from statistics import mean

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

        configuration = copy.deepcopy(self.default_configuration)
        offloading = self.hedger.get_offloading_plan()
        if not offloading:
            offloading = copy.deepcopy(self.default_offloading)
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
        if resource['bandwidth'] != -1:
            self.hedger.state_buffer.add_bandwidths(resource['bandwidth'])
        self.hedger.state_buffer.add_gpu_flops(device, resource['gpu_flops'])
        self.hedger.state_buffer.add_memory_capacity(device, resource['memory_capacity'])
        self.hedger.state_buffer.add_gpu_utilization(device, resource['gpu_usage'])
        self.hedger.state_buffer.add_memory_utilization(device, resource['memory_usage'])
        for service in resource['model_flops']:
            self.hedger.state_buffer.add_model_flops(service, resource['model_flops'][service])
        for service in resource['model_memory']:
            self.hedger.state_buffer.add_model_memory(service, resource['model_memory'][service])

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        for service_name in task.get_dag().nodes:
            if service_name in {TaskConstant.START.value, TaskConstant.END.value}:
                continue

            content = np.array(task.get_dag().get_node(service_name).service.get_content_data())
            if content.ndim ==2:
                complexity = mean([len(frame_content) for frame_content in content])
            elif content.ndim ==3:
                complexity = mean([len(frame_content[0]) for frame_content in content])
            else:
                raise TypeError(f'Dim of Input "content" must be 2 or 3, get dim {content.ndim}')
            latency = task.get_dag().get_node(service_name).service.get_execute_time()
            self.hedger.state_buffer.add_task_complexity(service_name, complexity)
            self.hedger.state_buffer.add_task_latency(service_name, latency)

    def get_schedule_overhead(self):
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
