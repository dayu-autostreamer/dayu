import abc
import os
import subprocess

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
        output_path = os.fspath(file_name)
        output_dir = os.path.dirname(os.path.abspath(output_path))
        output_name = os.path.basename(output_path)
        buffer_tmp_path = os.path.join(output_dir, f'.{output_name}.casva.tmp.mp4')
        out = cv2.VideoWriter(buffer_tmp_path, fourcc, 30, (width, height))
        for frame in frame_buffer:
            out.write(frame)
        out.release()

        try:
            if 'qp' in system.meta_data:
                qp = int(system.meta_data['qp'])
                subprocess.run(
                    [
                        'ffmpeg',
                        '-y',
                        '-i',
                        buffer_tmp_path,
                        '-c:v',
                        'libx264',
                        '-crf',
                        str(qp),
                        output_path,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                LOGGER.debug(
                    f'[Generator Compress] compress {output_path} into qp of {qp}'
                )
            else:
                os.replace(buffer_tmp_path, output_path)
            if not os.path.isfile(output_path):
                raise RuntimeError(
                    f'CASVA compressor did not create output {output_path!r}'
                )
            return output_path
        finally:
            FileOps.remove_file(buffer_tmp_path)
