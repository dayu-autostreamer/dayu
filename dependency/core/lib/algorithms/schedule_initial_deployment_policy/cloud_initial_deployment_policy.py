from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.scheduling.deployment_plan import cloud_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ("CloudInitialDeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias="cloud")
class CloudInitialDeploymentPolicy(BaseInitialDeploymentPolicy):
    """Explicitly place every current DAG service on the injected cloud node."""

    def __init__(self, system, agent_id):
        self.system = system

    def __call__(self, info):
        plan = cloud_plan(self.system, info)
        LOGGER.info(
            f"[Initial Deployment] (source {info['source']['id']}) Cloud policy: {plan}"
        )
        return plan
