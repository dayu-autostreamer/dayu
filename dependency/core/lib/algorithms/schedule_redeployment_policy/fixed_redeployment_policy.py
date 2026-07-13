import abc
from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, ConfigLoader, Context
from core.lib.scheduling.deployment_plan import fixed_plan

__all__ = ('FixedRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='fixed')
class FixedRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id, policy=None):
        """
        Args:
            policy: {'service1':['node1', 'node2'], 'service2':['node2', 'node3']}
        """
        self.cloud_device = str(getattr(system, "cloud_device", "") or "")
        if policy is None:
            self.fixed_policy = {}
        elif isinstance(policy, dict):
            self.fixed_policy = policy
        elif isinstance(policy, str):
            self.fixed_policy = ConfigLoader.load(Context.get_file_path(policy))
        else:
            raise TypeError(f'Input "policy" must be of type str or dict, get type {type(policy)}')

    def __call__(self, info):
        source_id = info['source']['id']
        deploy_plan = fixed_plan(self.fixed_policy, info, self.cloud_device)

        LOGGER.info(f'[Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
