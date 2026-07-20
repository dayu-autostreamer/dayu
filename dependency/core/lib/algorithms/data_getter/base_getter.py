import abc


class BaseDataGetter(metaclass=abc.ABCMeta):
    def __call__(self, system, task_identity=None):
        raise NotImplementedError
