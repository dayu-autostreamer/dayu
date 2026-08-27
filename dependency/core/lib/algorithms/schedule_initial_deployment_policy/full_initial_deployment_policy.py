import abc

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.scheduling.deployment_plan import full_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('FullInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='full')
class FullInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    """Deploy every service to every selected edge processor and the cloud."""

    def __init__(self, system, agent_id):
        self.cloud_device = str(getattr(system, "cloud_device", "") or "").strip()

    def __call__(self, info):
        source_id = info['source']['id']
        deploy_plan = full_plan(info, self.cloud_device)

        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
