import abc
import copy

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, ConfigLoader, Context

__all__ = ('FixedInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='fixed')
class FixedInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
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

        deploy_plan = copy.deepcopy(self.fixed_policy)

        all_services = list(dag.keys())
        for service in all_services:
            if service in deploy_plan:
                intersection_nodes = list(set(deploy_plan[service]) & set(node_set))
                deploy_plan[service] = intersection_nodes
            else:
                deploy_plan[service] = list(node_set)


        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
