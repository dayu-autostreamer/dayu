import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('ROILabelFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='roi_label_frame')
class ROILabelFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.roi_service = kwargs.get('roi_service', None)
        self.label_service = kwargs.get('label_service', None)

    def __call__(self, task: Task):
        try:
            if self.roi_service:
                roi_content = task.get_dag().get_node(self.roi_service).service.get_content_data()
            else:
                roi_content = task.get_first_content()
        except Exception as e:
            roi_content = task.get_first_content()

        try:
            if self.label_service:
                label_content = task.get_dag().get_node(self.label_service).service.get_content_data()
            else:
                label_content = task.get_last_content()
        except Exception as e:
            label_content = task.get_last_content()

        file_path = task.get_file_path()

        try:
            image = self.get_first_frame_from_video(file_path)
            image = self.draw_bboxes_and_labels(image, roi_content[0][0], label_content[0][0])

            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            import cv2
            base64_data = EncodeOps.encode_image(
                cv2.imread(self.default_visualization_image)
            )
            LOGGER.warning(f'Video visualization fetch failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}
