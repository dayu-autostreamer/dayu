import abc
import os

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps
from .base_compress import BaseCompress

__all__ = ('CasvaCompress',)


@ClassFactory.register(ClassType.GEN_COMPRESS, alias='casva')
class CasvaCompress(BaseCompress, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, frame_buffer, file_name):
        import cv2

        assert frame_buffer, 'frame buffer is empty!'
        fourcc = cv2.VideoWriter_fourcc(*system.meta_data['encoding'])
        height, width, _ = frame_buffer[0].shape
        buffer_tmp_path = f'tmp_{file_name}'
        out = cv2.VideoWriter(buffer_tmp_path, fourcc, 30, (width, height))
        for frame in frame_buffer:
            out.write(frame)
        out.release()

        buffer_path = file_name
        if 'qp' in system.meta_data:
            qp = system.meta_data['qp']
            os.system(f'ffmpeg -i {buffer_tmp_path} -c:v libx264 -crf {qp} {buffer_path}')
            LOGGER.debug(f'[Generator Compress] compress {buffer_path} into qp of {qp}')

        FileOps.remove_file(buffer_tmp_path)
