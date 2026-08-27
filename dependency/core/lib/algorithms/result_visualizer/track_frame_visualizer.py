import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('TrackFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='track_frame')
class TrackFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'track')
        self.frame_index = kwargs.get('frame_index', 0)
        self.label_fields = kwargs.get('label_fields', ['track_id', 'direction', 'speed_px_per_s'])
        self.color = tuple(kwargs.get('color', (255, 180, 0)))

    def __call__(self, task: Task):
        try:
            image = self.get_frame_from_video(task.get_file_path(), self.frame_index)
            content = task.get_service(self.service).get_content_data() if self.service else task.get_last_content()
            items = self.extract_items(content, self.output, self.frame_index)
            if not items:
                image = self.draw_no_output(image, self.service or 'last-content', self.output)
            else:
                image = self._draw_items(image, items)
            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            base64_data = self._fallback_image()
            LOGGER.warning(f'Track frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _draw_items(self, image, items):
        drawn = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            bboxes = item.get('bboxes') or []
            centers = [self._bbox_center(bbox) for bbox in bboxes]
            centers = [center for center in centers if center]
            if centers:
                self.draw_polyline(image, centers, color=self.color, thickness=3)
                drawn += 1
            if bboxes:
                label = self.item_label(item, fields=self.label_fields)
                self.draw_safe_bbox(image, bboxes[-1], label, color=self.color)
                drawn += 1
        if drawn == 0:
            self.draw_panel(image, self.service or 'track', ['track history missing'], color=self.color)
        return image

    @staticmethod
    def _bbox_center(bbox):
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            return None
        try:
            return [(float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0]
        except (TypeError, ValueError):
            return None

    def _fallback_image(self):
        import cv2
        import numpy as np

        image = cv2.imread(self.default_visualization_image)
        if image is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        return EncodeOps.encode_image(image)
