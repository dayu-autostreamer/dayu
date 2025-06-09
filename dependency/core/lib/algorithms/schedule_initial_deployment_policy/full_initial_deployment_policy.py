import abc

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('FullInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='full')
class FullInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        all_services = list(dag.keys())

        deploy_plan = {node: all_services.copy() for node in node_set}

        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
