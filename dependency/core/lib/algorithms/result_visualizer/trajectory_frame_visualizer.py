import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('TrajectoryFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='trajectory_frame')
class TrajectoryFrameVisualizer(ImageVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'trajectory')
        self.frame_index = kwargs.get('frame_index', 0)
        self.track_service = kwargs.get('track_service')
        self.track_output = kwargs.get('track_output', 'track')
        self.color = tuple(kwargs.get('color', (0, 120, 255)))

    def __call__(self, task: Task):
        try:
            image = self.get_frame_from_video(task.get_file_path(), self.frame_index)
            content = task.get_service(self.service).get_content_data() if self.service else task.get_last_content()
            items = self.extract_items(content, self.output, self.frame_index)
            tracks = self._tracks(task)
            if not items:
                image = self.draw_no_output(image, self.service or 'last-content', self.output)
            else:
                image = self._draw_items(image, items, tracks)
            base64_data = EncodeOps.encode_image(image)
        except Exception as e:
            base64_data = self._fallback_image()
            LOGGER.warning(f'Trajectory frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _tracks(self, task):
        if not self.track_service:
            return {}
        try:
            content = task.get_service(self.track_service).get_content_data()
        except Exception:
            return {}
        tracks = {}
        for item in self.extract_items(content, self.track_output, self.frame_index):
            if isinstance(item, dict) and item.get('track_id'):
                tracks[item.get('track_id')] = item
        return tracks

    def _draw_items(self, image, items, tracks):
        import cv2

        drawn = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            track = tracks.get(item.get('track_id'), {})
            bboxes = track.get('bboxes') or []
            if bboxes:
                self.draw_safe_bbox(image, bboxes[-1], str(item.get('track_id', '')), color=self.color)
            label = self._label(item)
            for trajectory in item.get('future_trajectories') or []:
                points = trajectory.get('points') or []
                valid_points = [self._point(point) for point in points]
                valid_points = [point for point in valid_points if point]
                if len(valid_points) >= 2:
                    self.draw_polyline(image, valid_points, color=self.color, thickness=3)
                    for point in valid_points:
                        cv2.circle(image, point, 3, self.color, -1, cv2.LINE_AA)
                    self.draw_text(image, label, valid_points[-1], background=self.color)
                    drawn += 1
        if drawn == 0:
            self.draw_panel(image, self.service or 'trajectory', ['trajectory points missing'], color=self.color)
        return image

    @staticmethod
    def _label(item):
        label_parts = []
        for field in ('track_id', 'vehicle_type'):
            value = item.get(field)
            if value:
                label_parts.append(str(value))
        if isinstance(item.get('abnormal_stop_prob'), (int, float)):
            label_parts.append(f"stop:{float(item.get('abnormal_stop_prob')):.2f}")
        return ' '.join(label_parts)

    def _fallback_image(self):
        import cv2
        import numpy as np

        image = cv2.imread(self.default_visualization_image)
        if image is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        return EncodeOps.encode_image(image)
