import abc


class BaseBSTOperation(metaclass=abc.ABCMeta):
    def __call__(self, system, new_task):
        raise NotImplementedError
