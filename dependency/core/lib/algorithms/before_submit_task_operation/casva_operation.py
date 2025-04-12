import abc
import json
import os
import shutil
import subprocess

from .base_operation import BaseBSTOperation

from core.lib.common import ClassFactory, ClassType, LOGGER, VideoOps
from core.lib.content import Task

__all__ = ('CASVABSTOperation',)


@ClassFactory.register(ClassType.GEN_BSTO, alias='casva')
class CASVABSTOperation(BaseBSTOperation, abc.ABC):
    def __init__(self):
        # # in multiprocessing env, we should use disk file to transmit past task info
        # self.past_info_record_path = 'casva_info_record.json'
        self.frame_count = 0

    def modify_file_qp(self, meta_data, file_path):
        if 'qp' not in meta_data:
            LOGGER.warning(f"'qp' not found in system metadata for {file_path}. Skipping compression.")
            return

        qp = meta_data['qp']

        tmp_file_path = 'tmp.mp4'

        try:
            # Build ffmpeg command
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-c:v', 'libx264',
                '-crf', str(qp),
                '-y',  # Overwrite without asking
                str(tmp_file_path)
            ]

            LOGGER.debug(f"Executing ffmpeg command: {' '.join(cmd)}")

            # Execute the ffmpeg command
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                LOGGER.warning(f"ffmpeg failed for {file_path} with error: {result.stderr}")
                return

            # Replace the original file with the compressed file
            shutil.move(str(tmp_file_path), str(file_path))
            LOGGER.debug(f"[Generator Compress] Compressed {file_path} with qp={qp}")
        except Exception as e:
            LOGGER.exception(f"An error occurred while compressing {file_path}: {e}")

    def filter_frame(self, fps_raw, fps) -> bool:
        fps = int(min(fps, fps_raw))
        fps_mode, skip_frame_interval, remain_frame_interval = self.get_fps_adjust_mode(fps_raw, fps)

        self.frame_count += 1
        if fps_mode == 'skip' and self.frame_count % skip_frame_interval == 0:
            return False

        if fps_mode == 'remain' and self.frame_count % remain_frame_interval != 0:
            return False

        return True

    def get_fps_adjust_mode(self, fps_raw, fps):
        skip_frame_interval = 0
        remain_frame_interval = 0
        if fps >= fps_raw:
            fps_mode = 'same'
        elif fps < fps_raw // 2:
            fps_mode = 'remain'
            remain_frame_interval = fps_raw // fps
        else:
            fps_mode = 'skip'
            skip_frame_interval = fps_raw // (fps_raw - fps)

        return fps_mode, skip_frame_interval, remain_frame_interval

    def reprocess_data(self, compressed_file, meta_data, past_metadata):
        # import cv2
        # cap = cv2.VideoCapture(compressed_file)
        # frame_list = []
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #
        #     if self.filter_frame(meta_data['fps'], past_metadata['fps']):
        #         resolution = VideoOps.text2resolution(past_metadata['resolution'])
        #         frame = cv2.resize(frame, resolution)
        #         frame_list.append(frame)
        #
        # tmp_process_file = 'dynamic_tmp.mp4'
        #
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # height, width, _ = frame_list[0].shape
        # out = cv2.VideoWriter(tmp_process_file, fourcc, 30, (width, height))
        # for frame in frame_list:
        #     out.write(frame)
        # out.release()

        # self.modify_file_qp(past_metadata, tmp_process_file)

        # return os.path.getsize(tmp_process_file)

        raw_size = os.path.getsize(compressed_file)
        raw_resolution = VideoOps.text2resolution(meta_data['resolution'])
        resolution = VideoOps.text2resolution(past_metadata['resolution'])
        raw_fps = meta_data['fps']
        fps = past_metadata['fps']

        return raw_size * (resolution[0] / raw_resolution[0]) * (fps / raw_fps)

    def __call__(self, system, new_task:Task):
        task = system.current_task

        tmp_data = task.get_tmp_data()
        meta_data = task.get_metadata()

        # self.modify_file_qp(meta_data, compressed_file)

        compressed_file = new_task.get_file_path()
        file_size = os.path.getsize(compressed_file) / 1024 / 1024
        tmp_data['file_size'] = file_size

        if hasattr(system, 'past_metadata') and hasattr(system, 'past_file_size'):
            file_size_with_last_config = self.reprocess_data(compressed_file, meta_data,
                                                             system.past_metadata) / 1024 / 1024
            tmp_data['file_dynamics'] = (file_size_with_last_config - system.past_file_size) / system.past_file_size
        else:
            tmp_data['file_dynamics'] = 0

        system.past_metadata = meta_data
        system.past_file_size = file_size
        task.set_tmp_data(tmp_data)
