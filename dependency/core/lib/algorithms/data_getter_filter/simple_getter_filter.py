import abc


from .base_getter_filter import BaseDataGetterFilter

from core.lib.common import ClassFactory, ClassType

__all__ = ('SimpleDataGetterFilter',)


@ClassFactory.register(ClassType.GEN_GETTER_FILTER, alias='simple')
class SimpleDataGetterFilter(BaseDataGetterFilter, abc.ABC):

    def __init__(self):
        pass

    def __call__(self, system):
        return True
