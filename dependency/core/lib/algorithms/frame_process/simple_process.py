import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.common import VideoOps

from .base_process import BaseProcess

__all__ = ('SimpleProcess',)


@ClassFactory.register(ClassType.GEN_PROCESS, alias='simple')
class SimpleProcess(BaseProcess, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, frame, source_resolution, target_resolution):
        import cv2
        if source_resolution == target_resolution:
            return frame
        else:
            return cv2.resize(frame, VideoOps.text2resolution(target_resolution))
