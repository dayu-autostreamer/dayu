import abc
from core.lib.common import ClassFactory, ClassType
from lib.content import Task

from .curve_visualizer import CurveVisualizer

__all__ = ('EndToEndDelayVisualizer',)


@ClassFactory.register(ClassType.VISUALIZER, alias='e2e_delay')
class EndToEndDelayVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, task: Task):
        return {'delay':task.calculate_total_time()}
