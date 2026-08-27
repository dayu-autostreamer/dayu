import abc
import copy

from core.lib.common import ConfigLoader, Context, GlobalInstanceManager, LOGGER
from core.lib.content import Task
from core.lib.scheduling.deployment_plan import cloud_replica_plan

from ..base_redeployment_policy import BaseRedeploymentPolicy

__all__ = ("HedgerAblationRedeploymentPolicyBase",)


class HedgerAblationRedeploymentPolicyBase(BaseRedeploymentPolicy, abc.ABC):
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
            hedger_config = copy.deepcopy(self.system.hedger_config)
            hedger_config.setdefault("agent_id", self.agent_id)
            self.hedger = GlobalInstanceManager.get_instance(
                self.controller_cls,
                f"{self.controller_alias}_{self.agent_id}",
                config=hedger_config,
            )

    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']
        source_device = info['source']['source_device']

        self.hedger.register_logical_topology(Task.extract_dag_from_dict(dag))
        self.hedger.register_physical_topology(list(node_set), source_device)
        self.hedger.register_state_buffer()

        cloud_device = str(self.system.cloud_device or "").strip()
        default_plan = cloud_replica_plan(
            copy.deepcopy(self.default_deployment),
            info,
            cloud_device,
            policy_name="Hedger",
        )

        if self.use_heuristic_deployment:
            deploy_plan = self.hedger.set_heuristic_deployment_plan(
                info=info,
                default_deployment=default_plan,
                mark_version=True,
            )
        else:
            deploy_plan = self.hedger.get_redeployment_plan()
            if deploy_plan is None:
                LOGGER.warning(
                    f"[HedgerAblation][Redeployment] alias={self.controller_alias}, source={source_id}, "
                    "no learned redeployment plan available; fall back to default deployment."
                )
                deploy_plan = copy.deepcopy(default_plan)

        deploy_plan = cloud_replica_plan(
            copy.deepcopy(deploy_plan),
            info,
            cloud_device,
            policy_name="Hedger",
        )
        deployment_version = int(self.hedger.get_active_deployment_version())
        plan_history = copy.deepcopy(
            getattr(self.hedger, "_deployment_plan_history", {}) or {}
        )
        plan_history[deployment_version] = copy.deepcopy(deploy_plan)
        for old_version in sorted(plan_history)[:-16]:
            plan_history.pop(old_version, None)
        self.hedger._deployment_plan_history = plan_history
        total_replicas = sum(len(nodes) for nodes in deploy_plan.values())
        LOGGER.info(
            f"[HedgerAblation][Redeployment] alias={self.controller_alias}, source={source_id}, "
            f"services={len(deploy_plan)}, replicas={total_replicas}"
        )
        LOGGER.debug(f"[HedgerAblation][Redeployment] source={source_id}, full_plan={deploy_plan}")
        return deploy_plan
