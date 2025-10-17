import abc
import threading
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
        self._ffmpeg_opts_set = False
        self._ffmpeg_backend_available = None  # None=unknown, True/False after first probe

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

    def _ensure_ffmpeg_options(self, system):
        """Set OPENCV_FFMPEG_CAPTURE_OPTIONS only once with sane defaults or overrides from meta_data."""
        if self._ffmpeg_opts_set:
            return
        # Prefer values from system.meta_data if available
        meta = getattr(system, 'meta_data', {}) or {}
        rtsp_transport = meta.get('rtsp_transport', 'tcp')
        stimeout_ms = int(meta.get('rtsp_stimeout_ms', 5000000))  # socket open timeout
        rw_timeout_ms = int(meta.get('rtsp_rw_timeout_ms', 5000000))  # read/write timeout
        # Compose ffmpeg options; keep existing defaults while allowing overrides
        opts = [
            f'rtsp_transport;{rtsp_transport}',
            f'stimeout;{stimeout_ms}',
            f'rw_timeout;{rw_timeout_ms}',
        ]
        # Avoid overriding if the user already set it from environment
        if not os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = '|'.join(opts)
        self._ffmpeg_opts_set = True

    def _open_capture(self, system):
        """(Re)open VideoCapture; try FFMPEG first once, then fall back to default backend to reduce warnings."""
        import cv2
        # Clean up previous capture if any
        if self.data_source_capture is not None:
            try:
                self.data_source_capture.release()
            except Exception:
                pass
            self.data_source_capture = None

        opened = False
        # Prefer FFMPEG if available/unknown
        if self._ffmpeg_backend_available is not False:
            cap = cv2.VideoCapture(system.video_data_source, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                self.data_source_capture = cap
                opened = True
                if self._ffmpeg_backend_available is None:
                    self._ffmpeg_backend_available = True
            else:
                # Mark ffmpeg backend as not usable to avoid repeated warnings
                try:
                    cap.release()
                except Exception:
                    pass
                self._ffmpeg_backend_available = False

        # Fallback to any available backend
        if not opened:
            cap = cv2.VideoCapture(system.video_data_source)  # CAP_ANY
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            self.data_source_capture = cap
            opened = cap.isOpened()

        return opened

    def get_one_frame(self, system):
        import cv2
        LOGGER.debug('[DEBUG] Start get one frame')
        self._ensure_ffmpeg_options(system)
        LOGGER.debug('[DEBUG] Ensure ffmpeg options set')
        # Open capture if needed
        if not self.data_source_capture or not self.data_source_capture.isOpened():
            self._open_capture(system)
        LOGGER.debug('[DEBUG] Check and reopen datasource')

        # Retry when no video signal, but with bounded attempts and backoff to avoid tight loops
        attempts = 0
        max_attempts = int(getattr(system, 'meta_data', {}).get('rtsp_max_retry', 8))
        base_sleep = float(getattr(system, 'meta_data', {}).get('rtsp_retry_backoff_sec', 0.5))
        first_no_signal = True

        while True:
            ret, frame = (False, None)
            if self.data_source_capture and self.data_source_capture.isOpened():
                ret, frame = self.data_source_capture.read()

            if ret and frame is not None:
                if not first_no_signal:
                    LOGGER.info(f'Get video stream data from source {system.source_id}..')
                LOGGER.debug(f'[DEBUG] Get one frame from source {system.source_id}..')
                return frame

            LOGGER.debug('[DEBUG] Failed to get frame from source')
            # Not successful; prepare to retry or give up gracefully
            if first_no_signal:
                LOGGER.warning(f'No video signal from source {system.source_id}!')
                first_no_signal = False

            attempts += 1
            self.frame_buffer = []

            # Reopen the capture before next attempt
            self._open_capture(system)
            LOGGER.debug(f'[DEBUG] Reopen datasource')

            if attempts >= max_attempts:
                # Give up for this tick; avoid spamming logs and CPU
                LOGGER.warning(
                    f'RTSP read failed after {attempts} attempts (source {system.source_id}). '
                    f'Please verify the RTSP URL/path, credentials, and network reachability. '
                    f'Common cause: server returned 404 on DESCRIBE.'
                )
                return None

            sleep_s = min(8.0, base_sleep * (2 ** (attempts - 1)))
            time.sleep(sleep_s)

    def generate_and_send_new_task(self, system, frame_buffer, new_task_id, task_dag, meta_data, ):
        source_id = system.source_id

        LOGGER.debug(f'[Frame Buffer] (source {system.source_id} / task {new_task_id}) '
                     f'buffer size: {len(frame_buffer)}')

        frame_buffer = [
            self.process_frame(system, frame, system.raw_meta_data['resolution'], meta_data['resolution'])
            for frame in frame_buffer
        ]
        LOGGER.debug(f'[DEBUG] process frames done')
        file_name = NameMaintainer.get_task_data_file_name(source_id, new_task_id, file_suffix=self.file_suffix)
        self.compress_frames(system, frame_buffer, file_name)
        LOGGER.debug(f'[DEBUG] compress frames done')

        new_task = system.generate_task(new_task_id, task_dag, meta_data, file_name, None)
        LOGGER.debug(f'[DEBUG] generate new task {new_task_id} done')
        system.submit_task_to_controller(new_task)
        LOGGER.debug(f'[DEBUG] submit task {new_task_id} done')
        FileOps.remove_file(file_name)

    def __call__(self, system):
        while len(self.frame_buffer) < system.meta_data['buffer_size']:
            frame = self.get_one_frame(system)
            if frame is None:
                # Sleep briefly to avoid busy loop when source is unavailable
                time.sleep(0.2)
                continue
            if self.filter_frame(system, frame):
                self.frame_buffer.append(frame)
                LOGGER.debug('[DEBUG] Append one frame to frame buffer')

        # generate tasks in parallel to avoid getting stuck with video compression
        new_task_id = Counter.get_count('task_id')
        threading.Thread(target=self.generate_and_send_new_task,
                         args=(system,
                               copy.deepcopy(self.frame_buffer),
                               new_task_id,
                               copy.deepcopy(system.task_dag),
                               copy.deepcopy(system.meta_data),)).start()

        self.frame_buffer = []
