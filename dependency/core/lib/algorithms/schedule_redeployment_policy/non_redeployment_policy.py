from .base_redeployment_policy import BaseRedeploymentPolicy
from core.lib.scheduling.deployment_plan import fixed_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('NonRedeploymentPolicy',)

@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='non')
class NonRedeploymentPolicy(BaseRedeploymentPolicy):
    """No-operation redeployment policy"""

    def __init__(self, system, agent_id, policy=None):
        self.cloud_device = str(getattr(system, "cloud_device", "") or "")
        service_deployment = system.runtime_service_nodes()
        if service_deployment is None:
            raise RuntimeError("runtime directory deployment is not initialized")
        self.non_policy = service_deployment

    def __call__(self, info):
        plan = fixed_plan(self.non_policy, info, self.cloud_device)
        LOGGER.info(f"[Redeployment] Using NonRedeploymentPolicy, returning static plan: {plan}")
        return plan
