import abc


class BaseDeploymentPolicy(metaclass=abc.ABCMeta):

    def __call__(self, info):
        raise NotImplementedError
