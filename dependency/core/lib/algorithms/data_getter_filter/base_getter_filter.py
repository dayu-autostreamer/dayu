import abc


class BaseDataGetterFilter(metaclass=abc.ABCMeta):
    def __call__(self, system):
        raise NotImplementedError
