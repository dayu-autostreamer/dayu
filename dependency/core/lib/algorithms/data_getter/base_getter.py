import abc
from enum import Enum

class DataGetterStatus(Enum):
    """Explicit non-task outcomes returned by a data getter."""

    EXHAUSTED = "exhausted"


class BaseDataGetter(metaclass=abc.ABCMeta):
    """Base contract implemented by every data-getter algorithm."""

    def __call__(self, system, task_identity=None):
        raise NotImplementedError
