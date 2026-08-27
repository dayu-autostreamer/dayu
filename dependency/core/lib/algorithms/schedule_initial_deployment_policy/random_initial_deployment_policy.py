import abc
import random

from .base_initial_deployment_policy import BaseInitialDeploymentPolicy

from core.lib.scheduling.deployment_plan import dag_services, validate_plan
from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('RandomInitialDeploymentPolicy',)


@ClassFactory.register(ClassType.SCH_INITIAL_DEPLOYMENT_POLICY, alias='random')
class RandomInitialDeploymentPolicy(BaseInitialDeploymentPolicy, abc.ABC):
    def __init__(self, system, agent_id, max_service_num=-1):
        self.max_service_num = max_service_num

    def __call__(self, info):
        source_id = info['source']['id']
        node_set = info['node_set']

        all_services = dag_services(info)
        node_services = {node: [] for node in node_set}

        for service in all_services:
            if self.max_service_num != -1:
                available_nodes = [n for n in node_set if len(node_services[n]) < self.max_service_num]
                if not available_nodes:
                    LOGGER.warning(f"[Initial Deployment] (source {source_id}) Service '{service}' cannot be deployed，"
                                   f"please check max_service_num (current:{self.max_service_num}) "
                                   f"or add nodes (current: {node_set})")
                    available_nodes = list(node_set)
                node = random.choice(available_nodes)
            else:
                node = random.choice(list(node_set))
            node_services[node].append(service)

        for node in node_set:
            current_services = node_services[node]
            candidates = list(set(all_services) - set(current_services))

            if self.max_service_num != -1:
                remaining = self.max_service_num - len(current_services)
                add_num = min(remaining, len(candidates))
            else:
                add_num = random.randint(0, len(candidates))

            if add_num > 0:
                node_services[node].extend(random.sample(candidates, add_num))

        # Keep capacity accounting node-oriented internally, but expose the
        # same service -> [nodes] contract as every other deployment policy.
        deploy_plan = {
            service: [node for node in node_set if service in node_services[node]]
            for service in all_services
        }

        LOGGER.info(f'[Initial Deployment] (source {source_id}) Deploy policy: {deploy_plan}')

        return validate_plan(deploy_plan, info)
