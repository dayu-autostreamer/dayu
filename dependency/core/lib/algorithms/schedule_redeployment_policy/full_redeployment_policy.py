from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.scheduling.deployment_plan import full_plan

__all__ = ("FullRedeploymentPolicy",)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias="full")
class FullRedeploymentPolicy(BaseRedeploymentPolicy):
    """Deploy every service to every selected edge processor and cloud node."""

    def __init__(self, system, agent_id):
        self.cloud_device = str(getattr(system, "cloud_device", "") or "").strip()

    def __call__(self, info):
        plan = full_plan(info, self.cloud_device)
        LOGGER.info(
            f"[Redeployment] (source {info['source']['id']}) Full policy: {plan}"
        )
        return plan
