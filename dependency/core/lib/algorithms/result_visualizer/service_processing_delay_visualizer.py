import abc
from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task

from .curve_visualizer import CurveVisualizer

__all__ = ('ServiceProcessingDelayVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='service_processing_delay')
class ServiceProcessingDelayVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        result = {}
        for variable in self.variables:
            result.update({variable:task.get_dag().get_node(variable).service.get_execute_time()})

        return result

