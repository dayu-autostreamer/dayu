from .base_redeployment_policy import BaseRedeploymentPolicy
from core.lib.common import ClassFactory, ClassType, KubeConfig, LOGGER

__all__ = ('NonRedeploymentPolicy',)

@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='non')
class NonRedeploymentPolicy(BaseRedeploymentPolicy):
    """No-operation redeployment policy"""

    def __init__(self, system, agent_id, policy=None):
        service_deployment = KubeConfig.get_service_nodes_dict()
        if service_deployment is None:
            raise RuntimeError("KubeConfig.get_service_nodes_dict() returned None")
        self.non_policy = service_deployment

    def __call__(self, info):
        LOGGER.info(f"[Redeployment] Using NonRedeploymentPolicy, returning static plan: {self.non_policy}")
        return self.non_policy 
