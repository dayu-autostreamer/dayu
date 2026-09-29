import abc

from .base_operation import BaseBSOperation

from core.lib.common import ClassFactory, ClassType

__all__ = ('SimpleBSOperation',)


@ClassFactory.register(ClassType.GEN_BSO, alias='simple')
class SimpleBSOperation(BaseBSOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system):
        return {}
