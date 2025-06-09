import abc


class BaseRedeploymentPolicy(metaclass=abc.ABCMeta):

    def __call__(self, info):
        raise NotImplementedError
