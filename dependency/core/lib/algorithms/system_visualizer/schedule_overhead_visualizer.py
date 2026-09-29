import abc
from core.lib.common import ClassFactory, ClassType

from .curve_visualizer import CurveVisualizer

__all__ = ('ScheduleOverheadVisualizer',)


@ClassFactory.register(ClassType.SYSTEM_VISUALIZER, alias='schedule_overhead')
class ScheduleOverheadVisualizer(CurveVisualizer, abc.ABC):
    def __call__(self, scheduling_overhead=None, **_):
        try:
            overhead_ms = float(scheduling_overhead) * 1000 if scheduling_overhead else 0
        except (TypeError, ValueError):
            overhead_ms = 0
        return {self.variables[0]: overhead_ms}
