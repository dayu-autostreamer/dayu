import abc
import multiprocessing
import copy
import os
import time

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
        # Keep a small internal buffer to reduce latency/jitter when pulling RTSP
        self._cap_buffer_size = 2
        # Backoff for reconnect attempts (seconds)
        self._reconnect_backoff = 0.5

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

    def _open_capture(self, url):
        import cv2
        # Prefer TCP to avoid UDP packet loss head-of-line artifacts on some networks.
        # Increase timeouts to avoid over-aggressive reconnect thrash under jitter.
        # Units are microseconds for ffmpeg options used by OpenCV backend.
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
            'rtsp_transport;tcp|'
            'stimeout;20000000|rw_timeout;20000000|'
            'max_delay;500000|'
            'probesize;32768|analyzeduration;0'
        )
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            # Reduce OpenCV internal queueing to limit latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cap_buffer_size)
        except Exception:
            pass
        return cap

    def get_one_frame(self, system):
        import cv2
        LOGGER.debug('[DEBUG] Start get a frame')
        if not self.data_source_capture or not self.data_source_capture.isOpened():
            # (Re)open with FFMPEG backend and low-latency options
            self.data_source_capture = self._open_capture(system.video_data_source)
        LOGGER.debug('[DEBUG] Check and open datasource')

        ret, frame = self.data_source_capture.read()
        first_no_signal = True
        LOGGER.debug('[DEBUG] Try to read a frame from datasource')

        # retry when no video signal
        while not ret:
            LOGGER.debug('[DEBUG] No frame signal from datasource')
            if first_no_signal:
                LOGGER.warning(f'No video signal from source {system.source_id}! Will retry...')
                first_no_signal = False
            LOGGER.debug('[DEBUG] Try to reopen datasource due to no signal')
            # Release and reopen to avoid stacking multiple sockets
            try:
                if self.data_source_capture:
                    self.data_source_capture.release()
            except Exception:
                pass
            self.data_source_capture = self._open_capture(system.video_data_source)
            # brief backoff to avoid tight reconnect spin
            LOGGER.debug('[DEBUG] Sleep for a while before reconnecting')
            time.sleep(self._reconnect_backoff)
            LOGGER.debug('[DEBUG] Try to read a frame from datasource')
            ret, frame = self.data_source_capture.read()

        LOGGER.debug('[DEBUG] Get a frame from datasource')

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
        LOGGER.debug(f'[DEBUG] Process frames in frame buffer')
        file_name = NameMaintainer.get_task_data_file_name(source_id, new_task_id, file_suffix=self.file_suffix)
        self.compress_frames(system, frame_buffer, file_name)
        LOGGER.debug(f'[DEBUG] Compress frames in frame buffer')

        new_task = system.generate_task(new_task_id, task_dag, meta_data, file_name, None)
        LOGGER.debug(f'[DEBUG] Generate new task {new_task_id}')
        system.submit_task_to_controller(new_task)
        LOGGER.debug(f'[DEBUG] Submit task {new_task_id}')
        FileOps.remove_file(file_name)

    def __call__(self, system):
        while len(self.frame_buffer) < system.meta_data['buffer_size']:
            frame = self.get_one_frame(system)
            if self.filter_frame(system, frame):
                self.frame_buffer.append(frame)
                LOGGER.debug(f'[DEBUG] Add a frame to frame buffer')

        # generate tasks in parallel to avoid getting stuck with video compression
        new_task_id = Counter.get_count('task_id')
        LOGGER.debug(f'[DEBUG] Ready to generate and send new task.')
        multiprocessing.Process(target=self.generate_and_send_new_task,
                                args=(system,
                                      copy.deepcopy(self.frame_buffer),
                                      new_task_id,
                                      copy.deepcopy(system.task_dag),
                                      copy.deepcopy(system.meta_data),)).start()

        self.frame_buffer = []
