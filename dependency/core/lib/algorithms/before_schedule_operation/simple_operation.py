import abc

from .base_operation import BaseBSOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

__all__ = ('SimpleBSOperation',)


@ClassFactory.register(ClassType.GEN_BSO, alias='simple')
class SimpleBSOperation(BaseBSOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system):
        parameters = {'source_id': system.source_id,
                      'meta_data': system.raw_meta_data,
                      'device': system.local_device,
                      'pipeline': Task.extract_dicts_from_dag(system.task_pipeline)}

        return parameters
