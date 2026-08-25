import abc

from .base_operation import BaseBSOperation

from core.lib.common import ClassFactory, ClassType

__all__ = ('CASVABSOperation',)


@ClassFactory.register(ClassType.GEN_BSO, alias='casva')
class CASVABSOperation(BaseBSOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system):
        parameters = {'skip_count': system.getter_filter.skip_count}
        system.getter_filter.reset_filter()

        return parameters
