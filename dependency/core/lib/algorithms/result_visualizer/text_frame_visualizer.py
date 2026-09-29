import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('TextFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='text_frame')
class TextFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'text')
        self.frame_index = kwargs.get('frame_index', 0)
        self.label_fields = kwargs.get('label_fields')
        self.label_template = kwargs.get('label_template')
        self.anchor_service = kwargs.get('anchor_service')
        self.anchor_output = kwargs.get('anchor_output', 'pose')
        self.anchor_key = kwargs.get('anchor_key', 'person_id')
        self.color = tuple(kwargs.get('color', (40, 180, 80)))

    def __call__(self, task: Task):
        try:
            image = self.get_frame_from_video(task.get_file_path(), self.frame_index)
            content = task.get_service(self.service).get_content_data() if self.service else task.get_last_content()
            items = self.extract_items(content, self.output, self.frame_index)
            anchors = self._anchors(task)
            if not items:
                image = self.draw_no_output(image, self.service or 'last-content', self.output)
            else:
                image = self._draw_items(image, items, anchors)
            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            base64_data = self._fallback_image()
            LOGGER.warning(f'Text frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _anchors(self, task):
        if not self.anchor_service:
            return {}
        try:
            content = task.get_service(self.anchor_service).get_content_data()
        except Exception:
            return {}
        anchors = {}
        for item in self.extract_items(content, self.anchor_output, self.frame_index):
            if isinstance(item, dict) and item.get(self.anchor_key):
                anchors[item.get(self.anchor_key)] = item
        return anchors

    def _draw_items(self, image, items, anchors):
        panel_lines = []
        drawn = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            label = self.item_label(
                item,
                fields=self.label_fields,
                template=self.label_template,
                fallback_fields=['text', 'intent', 'state', 'action', 'confidence'],
            )
            if self.clip_bbox(image, item.get('bbox')):
                self.draw_safe_bbox(image, item.get('bbox'), label, color=self.color)
                drawn += 1
                continue

            anchor = anchors.get(item.get(self.anchor_key))
            if anchor:
                origin = self._origin_from_anchor(anchor)
                if self.clip_bbox(image, anchor.get('bbox')):
                    self.draw_safe_bbox(image, anchor.get('bbox'), '', color=self.color)
                if origin:
                    self.draw_text(image, label, origin, background=self.color)
                    drawn += 1
                    continue

            if label:
                panel_lines.append(label)

        if panel_lines:
            self.draw_panel(image, self.service or 'text', panel_lines, color=self.color)
            drawn += 1
        if drawn == 0:
            self.draw_panel(image, self.service or 'text', ['text annotation missing'], color=self.color)
        return image

    def _origin_from_anchor(self, anchor):
        bbox = anchor.get('bbox') or []
        if len(bbox) == 4:
            try:
                return int(float(bbox[0])), int(float(bbox[1])) - 8
            except (TypeError, ValueError):
                return None
        for keypoint in anchor.get('keypoints') or []:
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
