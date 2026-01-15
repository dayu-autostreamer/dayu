import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

from .curve_visualizer import CurveVisualizer

__all__ = ('ObjectNumberVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='obj_num')
class ObjectNumberVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        import numpy as np
        result = float(np.mean(task.get_first_scenario_data()['obj_num']))
        return {self.variables[0]: result}
