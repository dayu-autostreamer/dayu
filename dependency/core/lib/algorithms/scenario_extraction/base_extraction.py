import abc


class BaseExtraction(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def __call__(self, result, task):
        pass
