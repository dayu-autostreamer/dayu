import abc
import copy

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager, ConfigLoader, Context, LOGGER
from core.lib.content import Task
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger')
class HedgerInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id, deployment=None):
        self.system = system
        self.agent_id = agent_id

        self.default_deployment = None
        self.load_default_policy(deployment)

        self.hedger = None
        self.register_hedger(hedger_id=f'hedger_{self.agent_id}')

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
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']
        source_device = info['source']['source_device']

        self.hedger.register_logical_topology(Task.extract_dag_from_dict(dag))
        self.hedger.register_physical_topology(list(node_set), source_device)
        self.hedger.register_initial_deployment(self.default_deployment)

        deploy_plan = self.hedger.get_initial_deployment_plan()

        if not deploy_plan:
            deploy_plan = copy.deepcopy(self.default_deployment)
            LOGGER.warning('No initial deployment plan, use default deployment policy.')

        all_services = list(dag.keys())
        for service in all_services:
            if service in deploy_plan:
                intersection_nodes = list(set(deploy_plan[service]) & set(node_set))
                deploy_plan[service] = intersection_nodes
            else:
                deploy_plan[service] = list(node_set)

        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')



