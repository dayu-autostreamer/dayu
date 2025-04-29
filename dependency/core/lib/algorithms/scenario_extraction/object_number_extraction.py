import abc

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType

__all__ = ('ObjectNumberExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='obj_num')
class ObjectNumberExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, result, task):
        obj_num = []

        for frame_result in result:
            bboxes = frame_result[0]
            boxes_num = len(bboxes)

            obj_num.append(boxes_num)

        return obj_num
