import abc
import random

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('RandomInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='random')
class RandomInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __init__(self, max_service_num=-1):
        self.max_service_num = max_service_num

    def __call__(self, info):
        source_id = info['source']['id']
        dag = info['dag']
        node_set = info['node_set']

        all_services = list(dag.keys())
        deploy_plan = {node: [] for node in node_set}

        for service in all_services:
            if self.max_service_num != -1:
                available_nodes = [n for n in node_set if len(deploy_plan[n]) < self.max_service_num]
                if not available_nodes:
                    LOGGER.warning(f"[Initial Deployment] (source {source_id}) Service '{service}' cannot be deployed，"
                                   f"please check max_service_num (current:{self.max_service_num}) "
                                   f"or add nodes (current: {node_set})")
                node = random.choice(available_nodes)
            else:
                node = random.choice(list(node_set))
            deploy_plan[node].append(service)

        for node in node_set:
            current_services = deploy_plan[node]
            candidates = list(set(all_services) - set(current_services))

            if self.max_service_num != -1:
                remaining = self.max_service_num - len(current_services)
                add_num = min(remaining, len(candidates))
            else:
                add_num = random.randint(0, len(candidates))

            if add_num > 0:
                deploy_plan[node].extend(random.sample(candidates, add_num))

        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return deploy_plan
