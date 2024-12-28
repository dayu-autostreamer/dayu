import abc


class BaseBSTOperation(metaclass=abc.ABCMeta):
    def __call__(self, system, compressed_file, hash_codes):
        raise NotImplementedError
