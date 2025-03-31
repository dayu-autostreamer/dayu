import abc

from .base_operation import BaseBSOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

__all__ = ('CASVABSOperation',)


@ClassFactory.register(ClassType.GEN_BSO, alias='casva')
class CASVABSOperation(BaseBSOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system):
        parameters = {'source_id': system.source_id,
                      'meta_data': system.raw_meta_data,
                      'source_device': system.local_device,
                      'all_edge_devices': system.all_edge_devices,
                      'dag': Task.extract_dag_deployment_from_dag(system.task_dag),
                      'skip_count': system.getter_filter.skip_count
                      }
        system.getter_filter.reset_filter()

        return parameters
