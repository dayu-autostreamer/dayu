import abc

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType

__all__ = ('ObjectVelocityExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='obj_velocity')
class ObjectVelocityExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, result, task):
        pass
