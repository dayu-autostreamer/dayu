import abc


from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType

__all__ = ('CPUUsageMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='cpu_usage')
class CPUUsageMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'cpu_usage'

    def get_parameter_value(self):
        import psutil
        return psutil.cpu_percent()


