import abc

from core.lib.content import Task

from .base_visualizer import BaseVisualizer


class CurveVisualizer(BaseVisualizer, abc.ABC):
    def __call__(self, task: Task):
        raise NotImplementedError
