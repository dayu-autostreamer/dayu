import abc


class BasePolicyRetrieval(metaclass=abc.ABCMeta):
    def __call__(self, task):
        raise NotImplementedError


