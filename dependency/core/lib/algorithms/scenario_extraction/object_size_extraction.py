import abc
import numpy as np

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType, VideoOps

__all__ = ('ObjectSizeExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='obj_size')
class ObjectSizeExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, result, task):
        if not isinstance(result, dict):
            return []
        outputs = result.get('outputs')
        if not isinstance(outputs, dict):
            return []

        bboxes_by_frame = {}
        frame_size = VideoOps.text2resolution(task.get_metadata()['resolution'])
        frame_area = frame_size[0] * frame_size[1]
        for records in outputs.values():
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                frame_index = record.get('frame_index')
                bboxes_by_frame.setdefault(frame_index, []).extend([
                    item.get('bbox')
                    for item in record.get('items') or []
                    if isinstance(item, dict) and len(item.get('bbox') or []) == 4
                ])

        obj_size_by_frame = {}
        for frame_index, bboxes in bboxes_by_frame.items():
            boxes_size = 0 if not bboxes else np.mean([
                ((box[2] - box[0]) * (box[3] - box[1])) / frame_area
                for box in bboxes
            ])
            obj_size_by_frame[frame_index] = boxes_size

        return [obj_size_by_frame[index] for index in sorted(obj_size_by_frame, key=lambda value: -1 if value is None else value)]
