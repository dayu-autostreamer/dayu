import abc


class BaseScenarioRetrieval(metaclass=abc.ABCMeta):
    def __call__(self, task):
        raise NotImplementedError


