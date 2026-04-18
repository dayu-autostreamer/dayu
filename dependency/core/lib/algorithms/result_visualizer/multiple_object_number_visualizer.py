import abc

import numpy as np

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

from .curve_visualizer import CurveVisualizer

__all__ = ('MultipleObjectNumberVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='multiple_obj_num')
class MultipleObjectNumberVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        return {
            variable: self._calculate_service_object_number(task, variable)
            for variable in self.variables
        }

    @classmethod
    def _calculate_service_object_number(cls, task: Task, service_name):
        try:
            scenario_data = task.get_scenario_data(service_name)
        except (AssertionError, KeyError):
            return 0.0
        return cls._calculate_object_number(scenario_data)

    @staticmethod
    def _calculate_object_number(scenario_data):
        if not isinstance(scenario_data, dict) or 'obj_num' not in scenario_data:
            return 0.0

        obj_num = scenario_data['obj_num']
        if obj_num is None:
            return 0.0
        if isinstance(obj_num, (int, float, np.number)):
            return float(obj_num)

        try:
            obj_num_array = np.asarray(obj_num, dtype=float)
        except (TypeError, ValueError):
            return 0.0

        if obj_num_array.size == 0:
            return 0.0
        return float(np.mean(obj_num_array.reshape(-1)))
