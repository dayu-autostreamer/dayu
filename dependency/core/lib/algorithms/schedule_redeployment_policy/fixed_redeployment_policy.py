import abc
from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, ConfigLoader, Context
from core.lib.scheduling.deployment_plan import fixed_plan, normalize_include_cloud

__all__ = ('FixedRedeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='fixed')
class FixedRedeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id, policy=None, include_cloud=False, **kwargs):
        """
        Args:
            policy: {'service1':['node1', '@cloud'], 'service2':['node2']}
            include_cloud: add the resolved cloud node to every service
        """
        self.cloud_device = str(getattr(system, "cloud_device", "") or "")
        self.include_cloud = normalize_include_cloud(include_cloud)
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
        deploy_plan = fixed_plan(
            self.fixed_policy,
            info,
            cloud_node=self.cloud_device,
            include_cloud=self.include_cloud,
        )

        LOGGER.info(f'[Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
