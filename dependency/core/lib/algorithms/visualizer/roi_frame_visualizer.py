import abc
import cv2

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('ROIFrameVisualizer',)


@ClassFactory.register(ClassType.VISUALIZER, alias='roi_frame')
class ROIFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, bbox_node: str = None):
        super().__init__()

        self.bbox_node = bbox_node

    def __call__(self, task: Task):
        try:
            if self.bbox_node:
                content = task.get_dag().get_node(self.bbox_node).service.get_content_data()
            else:
                content = task.get_first_content()
        except Exception as e:
            content = task.get_first_content()
        file_path = task.get_file_path()

        try:
            image = self.get_first_frame_from_video(file_path)
            image = self.draw_bboxes(image, content[0][0])

            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            base64_data = EncodeOps.encode_image(
                cv2.imread(self.default_visualization_image)
            )
            LOGGER.warning(f'Video visualization fetch failed: {str(e)}')
            LOGGER.exception(e)

        return {'image': base64_data}
