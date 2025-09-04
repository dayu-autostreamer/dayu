import abc

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager, Context, ConfigLoader
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger')
class HedgerRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id, deployment=None):
        self.system = system
        self.agent_id = agent_id

        self.default_deployment = None
        self.load_default_policy(deployment)

        self.hedger = self.register_hedger(f'hedger_{self.agent_id}')

    def load_default_policy(self, deployment):
        if deployment is None or isinstance(deployment, dict):
            self.default_deployment = deployment
        elif isinstance(deployment, str):
            self.default_deployment = ConfigLoader.load(Context.get_file_path(deployment))
        else:
            raise TypeError(f'Input "deployment" must be of type str or dict, get type {type(deployment)}')

    def register_hedger(self, hedger_id='hedger'):
        if self.hedger is None:
            network_params = self.system.network_params.copy()
            hyper_params = self.system.hyper_params.copy()
            agent_params = self.system.agent_params.copy()
            self.hedger = GlobalInstanceManager.get_instance(
                Hedger, hedger_id,
                network_params=network_params,
                hyper_params=hyper_params,
                agent_params=agent_params)
        self.hedger.register_deployment_agent()

    def __call__(self, info):
        dag = info['dag']
        node_set = info['node_set']

        self.hedger.get_redeployment_decision()
