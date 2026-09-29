import abc

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType

__all__ = ('StructuredProfileExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='structured_profile')
class StructuredProfileExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, result, task):
        if not isinstance(result, dict):
            return {}
        profile = result.get('profile')
        return profile if isinstance(profile, dict) else {}
