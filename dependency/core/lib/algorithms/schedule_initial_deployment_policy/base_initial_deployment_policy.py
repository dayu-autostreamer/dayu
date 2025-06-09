import abc


class BaseInitialDeploymentPolicy(metaclass=abc.ABCMeta):

    def __call__(self, info):
        raise NotImplementedError
