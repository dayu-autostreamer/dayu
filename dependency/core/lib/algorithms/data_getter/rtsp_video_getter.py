import abc
import threading
import copy
import os

from .base_getter import BaseDataGetter

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Counter, NameMaintainer

__all__ = ('RtspVideoGetter',)


@ClassFactory.register(ClassType.GEN_GETTER, alias='rtsp_video')
class RtspVideoGetter(BaseDataGetter, abc.ABC):
    """
    get video data from rtsp stream (in real time)
    simulate real video source, without accuracy information
    """

    def __init__(self):
        self.data_source_capture = None
        self.frame_buffer = []
        self.file_suffix = 'mp4'

    @staticmethod
    def filter_frame(system, frame):
        return system.frame_filter(system, frame)

    @staticmethod
    def process_frame(system, frame, source_resolution, target_resolution):
        return system.frame_process(system, frame, source_resolution, target_resolution)

    @staticmethod
    def compress_frames(system, frame_buffer, file_name):
        assert type(frame_buffer) is list and len(frame_buffer) > 0, 'Frame buffer is not list or is empty'
        return system.frame_compress(system, frame_buffer, file_name)

    def get_one_frame(self, system):
        import cv2
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000'
        if not self.data_source_capture:
            self.data_source_capture = cv2.VideoCapture(system.video_data_source)

        ret, frame = self.data_source_capture.read()
        first_no_signal = True

        # retry when no video signal
        while not ret:
            if first_no_signal:
                LOGGER.warning(f'No video signal from source {system.source_id}!')
                first_no_signal = False
            self.frame_buffer = []
            self.data_source_capture = cv2.VideoCapture(system.video_data_source, cv2.CAP_FFMPEG)
            ret, frame = self.data_source_capture.read()

        if not first_no_signal:
            LOGGER.info(f'Get video stream data from source {system.source_id}..')

        return frame

    def generate_and_send_new_task(self, system, frame_buffer, new_task_id, task_dag, meta_data, ):
        source_id = system.source_id

        LOGGER.debug(f'[Frame Buffer] (source {system.source_id} / task {new_task_id}) '
                     f'buffer size: {len(frame_buffer)}')

        frame_buffer = [
            self.process_frame(system, frame, system.raw_meta_data['resolution'], meta_data['resolution'])
            for frame in frame_buffer
        ]
        file_name = NameMaintainer.get_task_data_file_name(source_id, new_task_id, file_suffix=self.file_suffix)
        self.compress_frames(system, frame_buffer, file_name)

        new_task = system.generate_task(new_task_id, task_dag, meta_data, file_name, None)
        system.submit_task_to_controller(new_task)
        FileOps.remove_file(file_name)

    def __call__(self, system):
        while len(self.frame_buffer) < system.meta_data['buffer_size']:
            frame = self.get_one_frame(system)
            if self.filter_frame(system, frame):
                self.frame_buffer.append(frame)

        # generate tasks in parallel to avoid getting stuck with video compression
        new_task_id = Counter.get_count('task_id')
        threading.Thread(target=self.generate_and_send_new_task,
                         args=(system,
                               copy.deepcopy(self.frame_buffer),
                               new_task_id,
                               copy.deepcopy(system.task_dag),
                               copy.deepcopy(system.meta_data),)).start()

        self.frame_buffer = []
