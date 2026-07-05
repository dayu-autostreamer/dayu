import json
import os

from core.lib.common import Context


class TrafficObjectDetection:
    service_name = 'traffic-object-detection'
    default_model_name = 'traffic-object-detector'

    def __init__(self, model_name=None, model_weight='', model_variant='prototype',
                 device=0, synthetic_complexity=1):
        self.model_name = model_name or self.default_model_name
        self.model_weight = Context.get_file_path(model_weight) if model_weight else ''
        self.model_variant = model_variant
        self.device = device
        self.synthetic_complexity = max(1, int(synthetic_complexity))
        self.flops = 0

    def __call__(self, payload):
        width, height = self._frame_shape(payload)
        width = max(width, 640)
        height = max(height, 360)
        detections = []

        templates = [
            ('car', 0.08, 0.48, 0.28, 0.74, 0.92),
            ('bus', 0.40, 0.44, 0.70, 0.78, 0.88),
            ('pedestrian', 0.72, 0.45, 0.80, 0.86, 0.90),
            ('cyclist', 0.56, 0.50, 0.66, 0.84, 0.84),
            ('traffic_light', 0.46, 0.08, 0.50, 0.22, 0.91),
            ('traffic_sign', 0.82, 0.18, 0.90, 0.30, 0.86),
        ]

        for frame_id in self._frame_ids(payload):
            shift = (frame_id % 5) * 3
            for index, (category, x1, y1, x2, y2, score) in enumerate(templates):
                bbox = [
                    int(x1 * width + shift),
                    int(y1 * height),
                    int(x2 * width + shift),
                    int(y2 * height),
                ]
                detections.append({
                    'frame_id': frame_id,
                    'object_id': f'{category}-{index}',
                    'category': category,
                    'bbox': bbox,
                    'score': score,
                })

        outputs = {
            'detections': detections,
            'object_counts': self._detection_counts(detections),
        }
        return self._wrap_result(payload, outputs, num_objects=len(detections))

    def _wrap_result(self, payload, outputs, num_objects=0):
        profile = {
            'num_objects': int(num_objects),
            'input_bytes': self._input_bytes(payload),
            'output_bytes': self._output_bytes(outputs),
            'frame_count': len(payload.get('frames') or []),
            'model_name': self.model_name,
            'model_variant': self.model_variant,
            'model_weight': os.path.basename(self.model_weight) if self.model_weight else '',
            'synthetic_complexity': self.synthetic_complexity,
        }
        return {
            'service': self.service_name,
            'outputs': outputs,
            'profile': profile,
        }

    @staticmethod
    def _frame_shape(payload):
        frames = payload.get('frames') or []
        if not frames:
            return 0, 0
        height, width = frames[0].shape[:2]
        return int(width), int(height)

    @staticmethod
    def _frame_ids(payload):
        hashes = (payload.get('task') or {}).get('hash_data') or []
        if hashes:
            return list(range(len(hashes)))
        return list(range(len(payload.get('frames') or [])))

    @staticmethod
    def _detection_counts(detections):
        counts = {}
        for detection in detections:
            category = detection.get('category', 'unknown')
            counts[category] = counts.get(category, 0) + 1
        return counts

    @staticmethod
    def _input_bytes(payload):
        file_path = (payload.get('task') or {}).get('file_path')
        try:
            return os.path.getsize(file_path) if file_path else 0
        except OSError:
            return 0

    @staticmethod
    def _output_bytes(outputs):
        try:
            return len(json.dumps(outputs, default=str).encode('utf-8'))
        except TypeError:
            return 0
