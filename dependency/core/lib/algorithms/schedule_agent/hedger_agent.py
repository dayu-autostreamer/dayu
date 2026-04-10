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
        used_default_offloading = False
        if not offloading:
            offloading = self._normalize_mapping(self.default_offloading)
            used_default_offloading = True
            LOGGER.warning(
                f"[HedgerAgent][Schedule] source={source_id}, no Hedger offloading plan available; "
                f"fall back to default offloading policy."
            )

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

        service_names = [name for name in dag if name not in (TaskConstant.START.value, TaskConstant.END.value)]
        cloud_count = sum(
            1 for service_name in service_names
            if dag[service_name]['service'].get('execute_device') == cloud_device
        )
        assigned_devices = sorted({
            dag[service_name]['service'].get('execute_device')
            for service_name in service_names
        })
        sample_assignments = "; ".join(
            f"{service_name}->{dag[service_name]['service'].get('execute_device')}"
            for service_name in service_names[:3]
        ) or "[]"
        LOGGER.info(
            f"[HedgerAgent][Schedule] source={source_id}, services={len(service_names)}, "
            f"cloud={cloud_count}/{len(service_names) if service_names else 0}, "
            f"unique_devices={len(assigned_devices)}, used_default_offloading={used_default_offloading}, "
            f"sample={sample_assignments}"
        )
        LOGGER.debug(
            f"[HedgerAgent][Schedule] source={source_id}, assigned_devices={assigned_devices}, "
            f"full_policy={policy}"
        )
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

        bandwidth = resource.get('available_bandwidth')
        if bandwidth is not None and bandwidth != -1:
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

        model_flops_updates = resource.get('model_flops') or {}
        model_memory_updates = resource.get('model_memory') or {}
        updated_fields = []
        if bandwidth is not None and bandwidth != -1:
            updated_fields.append(f"wan={float(bandwidth):.2f}")
        if gpu_flops is not None:
            updated_fields.append(f"gpu_flops={float(gpu_flops):.2f}")
        if memory_capacity is not None:
            updated_fields.append(f"mem_capacity={float(memory_capacity):.2f}")
        if gpu_usage is not None:
            updated_fields.append(f"gpu_usage={float(gpu_usage):.4f}")
        if memory_usage is not None:
            updated_fields.append(f"mem_usage={float(memory_usage):.4f}")
        if model_flops_updates:
            updated_fields.append(f"model_flops={len(model_flops_updates)}")
        if model_memory_updates:
            updated_fields.append(f"model_memory={len(model_memory_updates)}")

        for service, flops in model_flops_updates.items():
            self.hedger.state_buffer.add_model_flops(service, flops)
        for service, memory in model_memory_updates.items():
            self.hedger.state_buffer.add_model_memory(service, memory)

        LOGGER.debug(
            f"[HedgerAgent][Resource] device={device}, updates={', '.join(updated_fields) if updated_fields else 'none'}"
        )

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        if self.hedger.state_buffer is None:
            return
        updated_services = 0
        complexity_values = []
        latency_values = []
        for service_name in task.get_dag().nodes:
            if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                continue
            service = task.get_service(service_name)
            latency = service.get_execute_time()
            complexity = self._extract_task_complexity(service)
            self.hedger.state_buffer.add_task_complexity(service_name, complexity)
            self.hedger.state_buffer.add_task_latency(service_name, latency)
            updated_services += 1
            complexity_values.append(float(complexity))
            latency_values.append(float(latency))

        if updated_services > 0:
            avg_complexity = float(np.mean(complexity_values)) if complexity_values else 0.0
            avg_latency = float(np.mean(latency_values)) if latency_values else 0.0
            LOGGER.debug(
                f"[HedgerAgent][Task] source={task.get_source_id()}, services={updated_services}, "
                f"avg_complexity={avg_complexity:.4f}, avg_latency={avg_latency:.4f}"
            )

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
