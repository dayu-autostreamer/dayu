import abc

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.scheduling.deployment_plan import dag_services, validate_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('FullInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='full')
class FullInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id):
        pass

    def __call__(self, info):
        source_id = info['source']['id']
        node_set = info['node_set']

        all_services = dag_services(info)

        # Canonical deployment contract: logical service -> target nodes.
        deploy_plan = {service: list(node_set) for service in all_services}

        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return validate_plan(deploy_plan, info)
