import abc

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
            # Remain the meta_data as before scheduling or raw_meta_data
            # Set execute device of all services as local device
            Task.set_execute_device(system.task_dag, system.local_device)
        else:
            scheduler_policy = scheduler_response['plan']
            dag = scheduler_policy['dag']
            system.task_dag = Task.extract_dag_from_dag_deployment(dag)
            del scheduler_policy['dag']
            system.meta_data.update(scheduler_policy)
