from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.scheduling.deployment_plan import full_edge_plan

__all__ = ("FullEdgeRedeploymentPolicy",)


@ClassFactory.register(
    ClassType.SCH_REDEPLOYMENT_POLICY,
    alias="full-edge",
)
class FullEdgeRedeploymentPolicy(BaseRedeploymentPolicy):
    """Deploy every service to every selected edge processor node."""

    def __init__(self, system, agent_id):
        self.cloud_device = str(getattr(system, "cloud_device", "") or "").strip()

    def __call__(self, info):
        plan = full_edge_plan(info, self.cloud_device)
        LOGGER.info(
            f"[Redeployment] (source {info['source']['id']}) "
            f"Full-edge policy: {plan}"
        )
        return plan
