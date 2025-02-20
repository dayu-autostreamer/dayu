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
            system.task_pipeline = Task.set_execute_device(system.task_pipeline, default_execute_device)
        else:
            scheduler_policy = scheduler_response['plan']
            pipeline = scheduler_policy['pipeline']
            system.task_pipeline = Task.extract_dag_from_dict(pipeline)
            del scheduler_policy['pipeline']
            system.meta_data.update(scheduler_policy)
