import abc
from core.lib.common import ClassFactory, ClassType, KubeConfig, Context, ConfigLoader
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('FixedAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='fixed')
class FixedAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, configuration=None, offloading=None):
        super().__init__()

        self.agent_id = agent_id
        self.cloud_device = system.cloud_device

        if configuration is None or isinstance(configuration, dict):
            self.fixed_configuration = configuration
        elif isinstance(configuration, str):
            self.fixed_configuration = ConfigLoader.load(Context.get_file_path(configuration))
        else:
            raise TypeError(f'Input "configuration" must be of type str or dict, get type {type(configuration)}')

        if offloading is None or isinstance(offloading, dict):
            self.fixed_offloading = offloading
        elif isinstance(offloading, str):
            self.fixed_offloading = ConfigLoader.load(Context.get_file_path(offloading))
        else:
            raise TypeError(f'Input "offloading" must be of type str or dict, get type {type(configuration)}')

        self.overhead_estimator = OverheadEstimator('Fixed', 'scheduler/fixed')

    def get_schedule_plan(self, info):
        if self.fixed_configuration is None or self.fixed_offloading is None:
            return None

        with self.overhead_estimator:
            configuration = self.fixed_configuration.copy()

            policy = {}
            policy.update(configuration)
            cloud_device = self.cloud_device
            source_edge_device = info['source_device']
            all_edge_devices = info['all_edge_devices']
            all_devices = [*all_edge_devices, cloud_device]
            service_info = KubeConfig.get_service_nodes_dict()

            dag = info['dag']

            for service_name in dag:
                if service_name in service_info and service_name in self.fixed_offloading \
                        and self.fixed_offloading[service_name] in all_devices:
                    dag[service_name]['service']['execute_device'] = self.fixed_offloading[service_name]
                elif service_name == 'start':
                    dag[service_name]['service']['execute_device'] = source_edge_device
                else:
                    dag[service_name]['service']['execute_device'] = cloud_device

            policy.update({'dag': dag})
        return policy

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
