import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('BBoxFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='bbox_frame')
class BBoxFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'bbox')
        self.frame_index = kwargs.get('frame_index', 0)
        self.label_fields = kwargs.get('label_fields')
        self.label_template = kwargs.get('label_template')
        self.color = tuple(kwargs.get('color', (0, 255, 0)))

    def __call__(self, task: Task):
        try:
            image = self.get_frame_from_video(task.get_file_path(), self.frame_index)
            content = self._content(task)
            items = self.extract_items(content, self.output, self.frame_index)
            if not items:
                image = self.draw_no_output(image, self.service or 'last-content', self.output)
            else:
                image = self._draw_items(image, items)
            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            base64_data = self._fallback_image()
            LOGGER.warning(f'BBox frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _content(self, task):
        if self.service:
            return task.get_service(self.service).get_content_data()
        return task.get_last_content()

    def _draw_items(self, image, items):
        loose_labels = []
        drawn = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            label = self.item_label(item, fields=self.label_fields, template=self.label_template)
            if self.clip_bbox(image, item.get('bbox')):
                self.draw_safe_bbox(image, item.get('bbox'), label, color=self.color)
                drawn += 1
            elif label:
                loose_labels.append(label)
        if drawn == 0:
            self.draw_panel(image, self.service or 'bbox frame', loose_labels or ['bbox missing'], color=self.color)
        return image

    def _fallback_image(self):
        import cv2
        import numpy as np

        image = cv2.imread(self.default_visualization_image)
        if image is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        return EncodeOps.encode_image(image)
