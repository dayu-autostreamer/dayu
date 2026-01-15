import abc
import copy

from .base_operation import BaseASOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task
from core.lib.network import NodeInfo

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
            system.service_deployment = scheduler_response.get('deployment', {})

            dag_deployment = scheduler_policy['dag']
            dag = Task.extract_dag_from_dag_deployment(dag_deployment)
            # Set execute device of start and end node
            dag.get_start_node().service.set_execute_device(system.local_device)
            dag.get_end_node().service.set_execute_device(NodeInfo.get_cloud_node())
            system.task_dag = copy.deepcopy(dag)
            del scheduler_policy['dag']
            system.meta_data.update(scheduler_policy)
