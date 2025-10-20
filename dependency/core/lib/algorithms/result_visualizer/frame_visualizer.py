import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('FrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='frame')
class FrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        file_path = task.get_file_path()

        try:
            image = self.get_first_frame_from_video(file_path)
            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            import cv2
            base64_data = EncodeOps.encode_image(
                cv2.imread(self.default_visualization_image)
            )
            LOGGER.warning(f'Video visualization fetch failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}
