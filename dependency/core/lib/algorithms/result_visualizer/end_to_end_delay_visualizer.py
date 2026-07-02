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
        try:
            delay = task.get_real_end_to_end_time()
        except ValueError:
            delay = task.calculate_total_time()
        return {self.variables[0]: delay}
