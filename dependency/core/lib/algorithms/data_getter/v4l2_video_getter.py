import abc
import os
import re

from .rtsp_video_getter import RtspVideoGetter

from core.lib.common import ClassFactory, ClassType, LOGGER

__all__ = ('V4L2VideoGetter',)


@ClassFactory.register(ClassType.GEN_GETTER, alias='v4l2_video')
class V4L2VideoGetter(RtspVideoGetter, abc.ABC):
    """
    Get video data from a local V4L2 camera device mounted inside the container.
    """

    @staticmethod
    def _normalize_device_source(source):
        if isinstance(source, int):
            return source

        source = os.fspath(source)
        if source.isdigit():
            return int(source)

        match = re.fullmatch(r'/dev/video(\d+)', source)
        if match:
            return int(match.group(1))

        return source

    @classmethod
    def _get_open_candidates(cls, source):
        normalized = cls._normalize_device_source(source)
        raw_source = os.fspath(source) if not isinstance(source, int) else source

        if isinstance(raw_source, str) and raw_source.startswith('/dev/'):
            candidates = [raw_source]
            if normalized != raw_source:
                candidates.append(normalized)
            return candidates, raw_source

        candidates = [normalized]
        if raw_source != normalized:
            candidates.append(raw_source)
        return candidates, raw_source

    def _open_capture(self, source):
        import cv2

        candidates, raw_source = self._get_open_candidates(source)

        if isinstance(raw_source, str) and raw_source.startswith('/dev/'):
            if not os.path.exists(raw_source):
                LOGGER.warning(f'Camera device "{raw_source}" is not mounted inside the container.')
            elif not os.access(raw_source, os.R_OK):
                LOGGER.warning(f'Camera device "{raw_source}" exists but is not readable by the current process.')

        for candidate in candidates:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
            if cap and cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass

        return cv2.VideoCapture(candidates[0])
