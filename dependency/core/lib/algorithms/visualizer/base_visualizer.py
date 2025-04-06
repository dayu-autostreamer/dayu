import abc

from core.lib.content import Task


class BaseVisualizer(metaclass=abc.ABCMeta):
    def __call__(self, task: Task):
        raise NotImplementedError
