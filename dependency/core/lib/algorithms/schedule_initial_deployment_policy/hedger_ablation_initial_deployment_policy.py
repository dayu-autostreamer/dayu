import abc
import copy

from core.lib.common import ClassFactory, ClassType, GlobalInstanceManager, ConfigLoader, Context, LOGGER
from core.lib.content import Task
from core.lib.algorithms.shared.hedger import (
    HedgerDeploymentAblation,
    HedgerFlatAblation,
    HedgerNoGraphEncoder,
    HedgerOffloadingAblation,
)

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

__all__ = (
    "HedgerFlatInitialDeploymentPolicy",
    "HedgerNoGraphEncoderInitialDeploymentPolicy",
    "HedgerDeploymentAblationInitialDeploymentPolicy",
    "HedgerOffloadingAblationInitialDeploymentPolicy",
)


class _HedgerAblationInitialDeploymentPolicyBase(BaseInitialDeploymentPolicy, abc.ABC):
    controller_cls = None
    controller_alias = "hedger_ablation"
    use_heuristic_deployment = False

    def __init__(self, system, agent_id, deployment=None):
        self.system = system
        self.agent_id = agent_id
        self.default_deployment = None
        self.load_default_policy(deployment)
        self.hedger = None
        self.register_hedger()

    def load_default_policy(self, deployment):
        if deployment is None or isinstance(deployment, dict):
            self.default_deployment = deployment
        elif isinstance(deployment, str):
            self.default_deployment = ConfigLoader.load(Context.get_file_path(deployment))
        else:
            raise TypeError(f'Input "deployment" must be of type str or dict, get type {type(deployment)}')

    def register_hedger(self):
        if self.hedger is None:
            self.hedger = GlobalInstanceManager.get_instance(
                self.controller_cls,
                f"{self.controller_alias}_{self.agent_id}",
                config=copy.deepcopy(self.system.hedger_config),
            )

    @staticmethod
    def _normalize_plan_for_backend(plan: dict, dag: dict, node_set) -> dict:
        plan = copy.deepcopy(plan) if isinstance(plan, dict) else {}
        normalized = {}
        for service in list(dag.keys()):
            selected_nodes = plan.get(service, list(node_set))
            if not isinstance(selected_nodes, (list, tuple, set)):
                selected_nodes = [selected_nodes]
            normalized[service] = list(dict.fromkeys(node for node in selected_nodes if node in node_set))
        return normalized

    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']
        source_device = info['source']['source_device']

        self.hedger.register_logical_topology(Task.extract_dag_from_dict(dag))
        self.hedger.register_physical_topology(list(node_set), source_device)
        self.hedger.register_state_buffer()
        self.hedger.register_initial_deployment(self.default_deployment)

        if self.use_heuristic_deployment:
            deploy_plan = self.hedger.set_heuristic_deployment_plan(
                info=info,
                default_deployment=self.default_deployment,
                mark_version=False,
            )
        else:
            deploy_plan = self.hedger.get_initial_deployment_plan()
            if not deploy_plan:
                deploy_plan = copy.deepcopy(self.default_deployment) or {}

        deploy_plan = self._normalize_plan_for_backend(deploy_plan, dag, node_set)
        total_replicas = sum(len(nodes) for nodes in deploy_plan.values())
        LOGGER.info(
            f"[HedgerAblation][InitialDeployment] alias={self.controller_alias}, source={source_id}, "
            f"services={len(deploy_plan)}, replicas={total_replicas}"
        )
        LOGGER.debug(f"[HedgerAblation][InitialDeployment] source={source_id}, full_plan={deploy_plan}")
        return deploy_plan


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-flat')
class HedgerFlatInitialDeploymentPolicy(_HedgerAblationInitialDeploymentPolicyBase):
    controller_cls = HedgerFlatAblation
    controller_alias = "hedger_flat"


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-no-graph-encoder')
class HedgerNoGraphEncoderInitialDeploymentPolicy(_HedgerAblationInitialDeploymentPolicyBase):
    controller_cls = HedgerNoGraphEncoder
    controller_alias = "hedger_no_graph_encoder"


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-deployment')
class HedgerDeploymentAblationInitialDeploymentPolicy(_HedgerAblationInitialDeploymentPolicyBase):
    controller_cls = HedgerDeploymentAblation
    controller_alias = "hedger_deployment"


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='hedger-offloading')
class HedgerOffloadingAblationInitialDeploymentPolicy(_HedgerAblationInitialDeploymentPolicyBase):
    controller_cls = HedgerOffloadingAblation
    controller_alias = "hedger_offloading"
    use_heuristic_deployment = True
