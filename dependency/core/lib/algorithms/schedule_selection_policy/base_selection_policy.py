import abc


class BaseSelectionPolicy(metaclass=abc.ABCMeta):

    def __call__(self, info):
        raise NotImplementedError
