import abc

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager, ConfigLoader, Context
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
        source_edge_device = info['source_device']
        all_edge_devices = info['all_edge_devices']
        dag = info['dag']

        self.hedger.register_logical_topology(Task.extract_dag_from_dag_deployment(dag))
        self.hedger.register_physical_topology(all_edge_devices, source_edge_device)

        return None

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
