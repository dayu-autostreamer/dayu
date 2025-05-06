import abc

from .base_operation import BaseASOperation

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.content import Task

__all__ = ('CASVAASOperation',)


@ClassFactory.register(ClassType.GEN_ASO, alias='casva')
class CASVAASOperation(BaseASOperation, abc.ABC):
    def __init__(self):
        self.default_metadata = {
            'resolution': '1080p',
            'fps': 5,
            'encoding': 'mp4v',
            'buffer_size': 10,
            'qp': 23
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

            if 'qp' not in system.meta_data:
                system.meta_data.update({'qp': self.default_metadata['qp']})
