import abc
import os

from .rtsp_video_getter import RtspVideoGetter

from core.lib.common import ClassFactory, ClassType

__all__ = ('V4L2VideoGetter',)


@ClassFactory.register(ClassType.GEN_GETTER, alias='v4l2_video')
class V4L2VideoGetter(RtspVideoGetter, abc.ABC):
    """
    Get video data from a local V4L2 camera device mounted inside the container.

    The same as RtspVideoGetter, except for the mode of open_source
    """

    @staticmethod
    def _normalize_device_source(source):
        if isinstance(source, int):
            return source

        source = os.fspath(source)
        return int(source) if source.isdigit() else source

    def _open_capture(self, source):
        import cv2

        return cv2.VideoCapture(self._normalize_device_source(source), cv2.CAP_V4L2)
