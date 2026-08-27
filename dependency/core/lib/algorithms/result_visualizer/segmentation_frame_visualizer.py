import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('SegmentationFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='segmentation_frame')
class SegmentationFrameVisualizer(ImageVisualizer, abc.ABC):
    default_colors = {
        'lane_polyline': (255, 255, 0),
        'road_boundary': (0, 255, 255),
        'drivable_area': (0, 160, 255),
        'crosswalk_region': (255, 0, 255),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'segmentation')
        self.frame_index = kwargs.get('frame_index', 0)
        self.alpha = float(kwargs.get('alpha', 0.28))

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
            LOGGER.warning(f'Segmentation frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _draw_items(self, image, items):
        drawn = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = item.get('type', 'segmentation')
            color = self.default_colors.get(item_type, (0, 180, 255))
            polygon = item.get('polygon')
            polyline = item.get('polyline') or item.get('points')
            if polygon:
                self.draw_polygon(image, polygon, color=color, alpha=self.alpha)
                self._label_shape(image, item_type, polygon, color)
                drawn += 1
            elif polyline:
                self.draw_polyline(image, polyline, color=color, thickness=3)
                self._label_shape(image, item_type, polyline, color)
                drawn += 1
        if drawn == 0:
            self.draw_panel(image, self.service or 'segmentation', ['no drawable shape'], color=(80, 80, 80))
        return image

    def _label_shape(self, image, label, points, color):
        if not points:
            return
        point = self._point(points[0])
        if point:
            self.draw_text(image, label, point, background=color)

    def _fallback_image(self):
        import cv2
        import numpy as np

        image = cv2.imread(self.default_visualization_image)
        if image is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        return EncodeOps.encode_image(image)
