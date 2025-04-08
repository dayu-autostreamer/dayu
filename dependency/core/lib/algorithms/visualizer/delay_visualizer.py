import abc
from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

from .curve_visualizer import CurveVisualizer

__all__ = ('EndToEndDelayVisualizer',)


@ClassFactory.register(ClassType.VISUALIZER, alias='delay')
class EndToEndDelayVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, task: Task):
        return {'Delay':task.calculate_total_time(),
                "Real Delay": task.get_real_end_to_end_time()}
