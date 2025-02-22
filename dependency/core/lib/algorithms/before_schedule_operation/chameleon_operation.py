import abc

from .base_operation import BaseBSOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

__all__ = ('ChameleonBSOperation',)


@ClassFactory.register(ClassType.GEN_BSO, alias='chameleon')
class ChameleonBSOperation(BaseBSOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system):
        parameters = {'source_id': system.source_id,
                      'meta_data': system.raw_meta_data,
                      'device': system.local_device,
                      'pipeline': Task.extract_deployment_info_from_dag(system.task_pipeline),
                      'frame': system.temp_encoded_frame,
                      'hash_code':system.temp_hash_code
                      }

        return parameters
