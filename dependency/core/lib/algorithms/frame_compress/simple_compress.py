import abc

from core.lib.common import ClassFactory, ClassType, NameMaintainer
from .base_compress import BaseCompress

__all__ = ('SimpleCompress',)


@ClassFactory.register(ClassType.GEN_COMPRESS, alias='simple')
class SimpleCompress(BaseCompress, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, frame_buffer, file_name):
        assert frame_buffer, 'frame buffer is empty!'

        import cv2
        buffer_path = file_name

        fourcc = cv2.VideoWriter_fourcc(*system.meta_data['encoding'])
        height, width, _ = frame_buffer[0].shape
        out = cv2.VideoWriter(buffer_path, fourcc, 30, (width, height))
        for frame in frame_buffer:
            out.write(frame)
        out.release()
