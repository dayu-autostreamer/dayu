import abc


from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType

__all__ = ('MemoryCapacityMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='memory_capacity')
class MemoryCapacityMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'memory_capacity'

    def get_parameter_value(self):
        import psutil
        return psutil.virtual_memory().total / 1e9
