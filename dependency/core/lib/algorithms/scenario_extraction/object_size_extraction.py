import abc
import numpy as np

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType

__all__ = ('ObjectSizeExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='obj_size')
class ObjectSizeExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, result, task):
        obj_size = []
        frame_size = task.get_metadata()['resolution']
        for frame_result in result:
            bboxes = frame_result[0]
            boxes_size = 0 if len(bboxes) == 0 else \
                np.mean([((box[2] - box[0]) * (box[3] - box[1]))
                         / (frame_size[0] * frame_size[1]) for box in bboxes])

            obj_size.append(boxes_size)

        return obj_size
