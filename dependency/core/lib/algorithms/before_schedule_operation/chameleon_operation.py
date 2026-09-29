import abc

from .base_operation import BaseBSOperation

from core.lib.common import ClassFactory, ClassType

__all__ = ('ChameleonBSOperation',)


@ClassFactory.register(ClassType.GEN_BSO, alias='chameleon')
class ChameleonBSOperation(BaseBSOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system):
        parameters = {'frame': system.temp_encoded_frame,
                      'hash_code': system.temp_hash_code
                      }

        return parameters
