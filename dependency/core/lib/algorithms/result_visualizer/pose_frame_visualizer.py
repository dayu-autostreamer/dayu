import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('PoseFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='pose_frame')
class PoseFrameVisualizer(ImageVisualizer, abc.ABC):
    fallback_links = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    coco_links = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
        (0, 2), (1, 3), (2, 4),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'pose')
        self.frame_index = kwargs.get('frame_index', 0)
        self.color = tuple(kwargs.get('color', (255, 0, 255)))

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
            LOGGER.warning(f'Pose frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _draw_items(self, image, items):
        drawn = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            keypoints = item.get('keypoints') or []
            links = self.coco_links if len(keypoints) >= 17 else self.fallback_links
            self.draw_keypoints(image, keypoints, color=self.color, links=links)
            if keypoints:
                drawn += 1
            label = self.item_label(item, fields=['person_id', 'orientation'])
            if self.clip_bbox(image, item.get('bbox')):
                self.draw_safe_bbox(image, item.get('bbox'), label, color=self.color)
                drawn += 1
            elif label:
                anchor = self._first_keypoint(keypoints)
                if anchor:
                    self.draw_text(image, label, anchor, background=self.color)
        if drawn == 0:
            self.draw_panel(image, self.service or 'pose', ['keypoints missing'], color=self.color)
        return image

    def _first_keypoint(self, keypoints):
        for keypoint in keypoints or []:
            point = self._point(keypoint)
            if point:
                return point
        return None

    def _fallback_image(self):
        import cv2
        import numpy as np

        image = cv2.imread(self.default_visualization_image)
        if image is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        return EncodeOps.encode_image(image)
