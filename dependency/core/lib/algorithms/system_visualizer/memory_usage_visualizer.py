import abc
from core.lib.common import ClassFactory, ClassType

from .curve_visualizer import CurveVisualizer

__all__ = ('MemoryUsageVisualizer',)


@ClassFactory.register(ClassType.SYSTEM_VISUALIZER, alias='memory_usage')
class MemoryUsageVisualizer(CurveVisualizer, abc.ABC):
    def __call__(self, resource=None, **_):
        if self.variables:
            if not resource:
                return {device: 0 for device in self.variables}
            return {device: resource[device]['memory_usage'] if device in resource else 0 for device in self.variables}

        else:
            if not resource:
                return {'no device': 0}
            else:
                return {device: resource[device]['memory_usage'] for device in resource}
