import abc

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType

__all__ = ('ObjectNumberExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='obj_num')
class ObjectNumberExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, result, task):
        if not isinstance(result, dict):
            return []
        outputs = result.get('outputs')
        if not isinstance(outputs, dict):
            return []

        frame_counts = {}
        for records in outputs.values():
            for record in records or []:
                if not isinstance(record, dict):
                    continue
                frame_index = record.get('frame_index')
                frame_counts[frame_index] = frame_counts.get(frame_index, 0) + len(record.get('items') or [])
        return [frame_counts[index] for index in sorted(frame_counts, key=lambda value: -1 if value is None else value)]
