import abc

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger')
class HedgerInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id):
        self.system = system
        self.agent_id = agent_id

        self.hedger = None

    def register_hedger(self, hedger_id='hedger'):
        if self.hedger is None:
            network_params = self.system.network_params.copy()
            self.hedger = GlobalInstanceManager.get_instance(Hedger, hedger_id,
                                                             network_params=network_params)
            self.hedger.register_deployment_agent()

    def __call__(self, info):
        dag = info['dag']
        node_set = info['node_set']

        self.register_hedger(hedger_id=f'hedger_{self.agent_id}')

