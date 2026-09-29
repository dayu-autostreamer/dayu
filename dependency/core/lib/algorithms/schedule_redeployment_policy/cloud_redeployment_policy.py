from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.scheduling.deployment_plan import cloud_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ("CloudRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias="cloud")
class CloudRedeploymentPolicy(BaseRedeploymentPolicy):
    """Keep every current DAG service explicitly placed on the cloud node."""

    def __init__(self, system, agent_id):
        self.system = system

    def __call__(self, info):
        plan = cloud_plan(self.system, info)
        LOGGER.info(
            f"[Redeployment] (source {info['source']['id']}) Cloud policy: {plan}"
        )
        return plan
