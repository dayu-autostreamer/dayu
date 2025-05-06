import abc


class BaseProcess(metaclass=abc.ABCMeta):
    def __call__(self, system, frame, source_resolution, target_resolution):
        raise NotImplementedError
