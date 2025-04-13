import abc

from .base_deployment_policy import BaseDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, ConfigLoader, Context

__all__ = ('FixedDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_DEPLOYMENT_POLICY, alias='fixed')
class FixedDeploymentPolicy(BaseDeploymentPolicy, abc.ABC):
    def __init__(self, policy):
        """
        Args:
            policy: {'service1':['node1', 'node2'], 'service2':['node2', 'node3']}
        """
        if policy is None or isinstance(policy, dict):
            self.fixed_policy = policy
        elif isinstance(policy, str):
            self.fixed_policy = ConfigLoader.load(Context.get_file_path(policy))
        else:
            raise TypeError(f'Input "policy" must be of type str or dict, get type {type(policy)}')

    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        all_services = list(dag.keys())
        for service in all_services:
            if service in self.fixed_policy:
                intersection_nodes = list(set(self.fixed_policy[service]) & node_set)
                self.fixed_policy[service] = intersection_nodes if intersection_nodes else list(node_set)
            else:
                self.fixed_policy[service] = list(node_set)

        deploy_plan = {}
        for node, services in self.fixed_policy.items():
            for service in services:
                deploy_plan.setdefault(service, []).append(node)

        LOGGER.info(f'[Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
