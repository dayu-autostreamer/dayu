import abc

from core.lib.common import ClassFactory, ClassType, EncodeOps, LOGGER
from core.lib.content import Task

from .image_visualizer import ImageVisualizer

__all__ = ('EventFrameVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='event_frame')
class EventFrameVisualizer(ImageVisualizer, abc.ABC):
    level_colors = {
        'low': (30, 160, 80),
        'medium': (0, 170, 255),
        'high': (40, 40, 220),
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.service = kwargs.get('service')
        self.output = kwargs.get('output', 'graph')
        self.frame_index = kwargs.get('frame_index', 0)
        self.max_events = int(kwargs.get('max_events', 4))

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
            LOGGER.warning(f'Event frame visualization failed: {str(e)}')
            LOGGER.exception(e)

        return {self.variables[0]: base64_data}

    def _draw_items(self, image, items):
        graph = next((item for item in items if isinstance(item, dict)), {})
        if not graph:
            self.draw_panel(image, self.service or 'events', ['graph item missing'], color=(80, 80, 80))
            return image

        risk_level = str(graph.get('risk_level') or self.get_nested_value(graph, 'summary.risk_level') or 'unknown')
        risk_confidence = graph.get('risk_confidence')
        if risk_confidence is None:
            risk_confidence = self.get_nested_value(graph, 'summary.risk_confidence')
        color = self.level_colors.get(risk_level, (80, 80, 80))

        summary = graph.get('summary') or {}
        lines = [
            self._format_risk(risk_level, risk_confidence),
            f"nodes:{len(graph.get('nodes') or [])} edges:{len(graph.get('edges') or [])}",
        ]
        for key in ('entity_count', 'relation_count', 'signal_count'):
            if key in summary:
                lines.append(f'{key}:{summary.get(key)}')
        for event in (graph.get('events') or [])[:self.max_events]:
            if not isinstance(event, dict):
                continue
            event_type = event.get('type', 'event')
            risk_score = event.get('risk_score')
            if isinstance(risk_score, (int, float)):
                lines.append(f'{event_type}:{float(risk_score):.2f}')
            else:
                lines.append(str(event_type))

        self.draw_panel(image, self.service or 'events', lines, color=color)
        return image

    @staticmethod
    def _format_risk(risk_level, risk_confidence):
        if isinstance(risk_confidence, (int, float)):
            return f'risk:{risk_level} {float(risk_confidence):.2f}'
        return f'risk:{risk_level}'

    def _fallback_image(self):
        import cv2
        import numpy as np

        image = cv2.imread(self.default_visualization_image)
        if image is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        return EncodeOps.encode_image(image)
