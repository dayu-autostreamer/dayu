import abc
import copy

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager, Context, ConfigLoader, LOGGER
from core.lib.content import Task
from core.lib.algorithms.shared.hedger import Hedger

__all__ = ('HedgerRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='hedger')
class HedgerRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id, deployment=None):
        self.system = system
        self.agent_id = agent_id

        self.default_deployment = None
        self.load_default_policy(deployment)

        self.hedger = None
        self.register_hedger(f'hedger_{self.agent_id}')

    def load_default_policy(self, deployment):
        if deployment is None or isinstance(deployment, dict):
            self.default_deployment = deployment
        elif isinstance(deployment, str):
            self.default_deployment = ConfigLoader.load(Context.get_file_path(deployment))
        else:
            raise TypeError(f'Input "deployment" must be of type str or dict, get type {type(deployment)}')

    def register_hedger(self, hedger_id='hedger'):
        if self.hedger is None:
            self.hedger = GlobalInstanceManager.get_instance(
                Hedger, hedger_id,
                config=copy.deepcopy(self.system.hedger_config))

    def __call__(self, info):
        LOGGER.debug(f'***info in redeployment: {info}')
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']
        source_device = info['source']['source_device']

        self.hedger.register_logical_topology(Task.extract_dag_from_dict(dag))
        self.hedger.register_physical_topology(list(node_set), source_device)
        self.hedger.register_state_buffer()

        deploy_plan = self.hedger.get_redeployment_plan()
        if deploy_plan is None:
            LOGGER.warning(
                f"[HedgerPolicy][Redeployment] source={source_id}, no Hedger redeployment plan available; "
                f"fall back to default deployment policy."
            )
            deploy_plan = copy.deepcopy(self.default_deployment) or {}

        all_services = list(dag.keys())
        for service in all_services:
            if service in deploy_plan:
                selected_nodes = deploy_plan[service]
                if isinstance(selected_nodes, (list, tuple, set)):
                    candidate_nodes = list(selected_nodes)
                else:
                    candidate_nodes = [selected_nodes]
                deploy_plan[service] = list(dict.fromkeys(node for node in candidate_nodes if node in node_set))
            else:
                deploy_plan[service] = list(node_set)

        total_replicas = sum(len(nodes) for nodes in deploy_plan.values())
        sample = "; ".join(
            f"{service}->{deploy_plan[service]}"
            for service in list(deploy_plan.keys())[:3]
        ) or "[]"
        LOGGER.info(
            f"[HedgerPolicy][Redeployment] source={source_id}, services={len(deploy_plan)}, "
            f"nodes={len(node_set)}, replicas={total_replicas}, sample={sample}"
        )
        LOGGER.debug(f"[HedgerPolicy][Redeployment] source={source_id}, full_plan={deploy_plan}")

        return deploy_plan
