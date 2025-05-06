import abc

from .base_operation import BaseASOperation

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.content import Task

__all__ = ('SimpleASOperation',)


@ClassFactory.register(ClassType.GEN_ASO, alias='simple')
class SimpleASOperation(BaseASOperation, abc.ABC):
    def __init__(self):
        self.default_metadata = {
            'resolution': '1080p',
            'fps': 30,
            'encoding': 'mp4v',
            'buffer_size': 8
        }

    def __call__(self, system, scheduler_response):

        if scheduler_response is None:
            system.meta_data.update(self.default_metadata)
            default_execute_device = system.local_device
            system.task_dag = Task.set_execute_device(system.task_dag, default_execute_device)
        else:
            scheduler_policy = scheduler_response['plan']
            dag = scheduler_policy['dag']
            system.task_dag = Task.extract_dag_from_dag_deployment(dag)
            del scheduler_policy['dag']
            system.meta_data.update(scheduler_policy)
