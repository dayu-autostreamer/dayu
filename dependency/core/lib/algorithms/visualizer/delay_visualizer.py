import abc
import random
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
                "Other Delay": random.uniform(0.1, 0.4)}
