import abc

from core.lib.content import Task


class BaseVisualizer(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        self.variables = kwargs.get('variables', [])

    def __call__(self, task: Task):
        raise NotImplementedError
