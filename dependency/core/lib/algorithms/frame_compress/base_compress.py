import abc


class BaseCompress(metaclass=abc.ABCMeta):
    def __call__(self, system, frame_buffer, file_name):
        raise NotImplementedError
