import abc
from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

from .curve_visualizer import CurveVisualizer

__all__ = ('EndToEndDelayVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='e2e_delay')
class EndToEndDelayVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        return {self.variables[0]: task.calculate_total_time()}
