import abc

from .base_operation import BaseASOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

__all__ = ('CASVAASOperation',)


@ClassFactory.register(ClassType.GEN_ASO, alias='casva')
class CASVAASOperation(BaseASOperation, abc.ABC):
    def __init__(self):
        self.default_qp = 23

    def __call__(self, system, scheduler_response):

        if scheduler_response is None:
            # Remain the meta_data as before scheduling or raw_meta_data
            # Set execute device of all services as local device
            system.task_dag = Task.set_execute_device(system.task_dag, system.local_device)
        else:
            scheduler_policy = scheduler_response['plan']
            dag = scheduler_policy['dag']
            system.task_dag = Task.extract_dag_from_dag_deployment(dag)
            del scheduler_policy['dag']
            system.meta_data.update(scheduler_policy)

            if 'qp' not in system.meta_data:
                system.meta_data.update({'qp': self.default_qp})
