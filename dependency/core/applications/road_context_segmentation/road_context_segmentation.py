import json
import os

from core.lib.common import Context


class RoadContextSegmentation:
    service_name = 'road-context-segmentation'
    default_model_name = 'road-context-segmenter'

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

        lane_polylines = [
            [[int(width * 0.34), height], [int(width * 0.42), int(height * 0.58)], [int(width * 0.48), int(height * 0.35)]],
            [[int(width * 0.66), height], [int(width * 0.58), int(height * 0.58)], [int(width * 0.52), int(height * 0.35)]],
        ]
        drivable_area = [
            [0, height],
            [int(width * 0.40), int(height * 0.35)],
            [int(width * 0.60), int(height * 0.35)],
            [width, height],
        ]
        crosswalk_regions = [
            [
                [int(width * 0.20), int(height * 0.62)],
                [int(width * 0.80), int(height * 0.62)],
                [int(width * 0.86), int(height * 0.74)],
                [int(width * 0.14), int(height * 0.74)],
            ]
        ]

        outputs = {
            'lane_polylines': lane_polylines,
            'drivable_area': drivable_area,
            'crosswalk_regions': crosswalk_regions,
            'road_boundary': [[0, height], [width, height]],
        }
        return self._wrap_result(payload, outputs, num_objects=len(lane_polylines) + len(crosswalk_regions))

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
