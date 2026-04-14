import abc


from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType

__all__ = ('MemoryUsageMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='memory_usage')
class MemoryUsageMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'memory_usage'

    def get_parameter_value(self):
        """Return memory utilization ratio in [0, 1]."""
        import psutil
        return psutil.virtual_memory().percent / 100.0
