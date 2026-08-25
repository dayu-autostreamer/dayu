import abc
import copy

from .base_operation import BaseASOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

__all__ = ('SimpleASOperation',)


@ClassFactory.register(ClassType.GEN_ASO, alias='simple')
class SimpleASOperation(BaseASOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, scheduler_response):
        if scheduler_response is None:
            return

        scheduler_policy = copy.deepcopy(scheduler_response.get('plan') or {})
        if not isinstance(scheduler_policy, dict):
            raise TypeError('scheduler response plan must be an object')

        deployment = scheduler_response.get('deployment')
        if isinstance(deployment, dict):
            system.service_deployment = copy.deepcopy(deployment)
        if 'deployment_version' in scheduler_response:
            deployment_version = scheduler_response.get('deployment_version')
            if deployment_version is not None:
                system.deployment_version = deployment_version

        dag_deployment = scheduler_policy.pop('dag', None)
        if dag_deployment is not None:
            if not isinstance(dag_deployment, dict):
                raise TypeError('scheduler plan dag must be an object')
            dag = Task.extract_dag_from_dag_deployment(dag_deployment)
            system.task_dag = copy.deepcopy(dag)

        # Every non-control plan field is a configuration decision. Missing
        # fields retain the current value, so algorithms may schedule any
        # subset without the host imposing video-specific defaults.
        system.meta_data.update(scheduler_policy)
