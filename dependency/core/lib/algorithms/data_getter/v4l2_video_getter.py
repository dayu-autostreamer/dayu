import abc
import os
import re

from .rtsp_video_getter import RtspVideoGetter

from core.lib.common import ClassFactory, ClassType

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

    def _open_capture(self, source):
        import cv2

        candidates = []
        normalized = self._normalize_device_source(source)
        candidates.append(normalized)

        raw_source = os.fspath(source) if not isinstance(source, int) else source
        if raw_source != normalized:
            candidates.append(raw_source)

        for candidate in candidates:
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
            if cap and cap.isOpened():
                return cap
            try:
                cap.release()
            except Exception:
                pass

        return cv2.VideoCapture(normalized)
