import abc


class BaseQueue(metaclass=abc.ABCMeta):
    def get(self):
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def put(self, task):
        raise NotImplementedError

    def empty(self):
        raise NotImplementedError

    def drain(self, max_count=None):
        raise NotImplementedError

    def get_all_without_drop(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError
