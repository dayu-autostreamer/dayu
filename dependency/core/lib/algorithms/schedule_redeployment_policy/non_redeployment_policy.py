import abc

from .base_redeployment_policy import BaseRedeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER, KubeConfig

__all__ = ('FixedInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_REDEPLOYMENT_POLICY, alias='non')
class FixedInitialDeploymentPolicy(BaseRedeploymentPolicy, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        deploy_plan = {}
        service_nodes_dict = KubeConfig.get_service_nodes_dict()
        all_services = list(dag.keys())
        for service in all_services:
            if service in service_nodes_dict:
                intersection_nodes = list(set(service_nodes_dict[service]) & set(node_set))
                deploy_plan[service] = intersection_nodes
            else:
                deploy_plan[service] = list(node_set)

        LOGGER.info(f'[Redeployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
