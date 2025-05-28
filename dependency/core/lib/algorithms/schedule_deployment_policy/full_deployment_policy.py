import abc

from .base_deployment_policy import BaseDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('FullDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_DEPLOYMENT_POLICY, alias='full')
class FullDeploymentPolicy(BaseDeploymentPolicy, abc.ABC):
    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        all_services = list(dag.keys())

        deploy_plan = {node: all_services.copy() for node in node_set}

        LOGGER.info(f'[Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
