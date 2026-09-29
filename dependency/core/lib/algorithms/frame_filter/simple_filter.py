import abc
from core.lib.common import ClassFactory, ClassType
from .base_filter import BaseFilter

__all__ = ('SimpleFilter',)


@ClassFactory.register(ClassType.GEN_FILTER, alias='simple')
class SimpleFilter(BaseFilter, abc.ABC):
    def __init__(self):
        self._rate_key = None
        self._sampling_accumulator = 0

    def __call__(self, system, frame) -> bool:
        fps_raw = int(system.raw_meta_data['fps'])
        fps = min(int(system.meta_data['fps']), fps_raw)
        if fps_raw <= 0 or fps <= 0:
            return False
        if fps >= fps_raw:
            self._rate_key = (fps_raw, fps)
            self._sampling_accumulator = 0
            return True

        rate_key = (fps_raw, fps)
        if rate_key != self._rate_key:
            self._rate_key = rate_key
            # Keep the first frame after initialization/configuration changes,
            # then use a Bresenham-style accumulator.  Unlike integer skip
            # intervals, this preserves arbitrary ratios such as 30 -> 8 fps.
            self._sampling_accumulator = fps_raw - fps

        self._sampling_accumulator += fps
        if self._sampling_accumulator >= fps_raw:
            self._sampling_accumulator -= fps_raw
            return True
        return False

    @staticmethod
    def get_fps_adjust_mode(fps_raw, fps):
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
