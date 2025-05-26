import abc
from .base_visualizer import BaseVisualizer


class CurveVisualizer(BaseVisualizer, abc.ABC):
    def __call__(self):
        raise NotImplementedError
