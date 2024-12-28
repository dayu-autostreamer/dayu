import abc

from .base_operation import BaseBSTOperation

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

__all__ = ('SimpleBSTOperation',)


@ClassFactory.register(ClassType.GEN_BSTO, alias='simple')
class SimpleBSTOperation(BaseBSTOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, compressed_file, hash_codes):
        pass
