import json
import os

from core.lib.common import Context


class VehicleReidentificationTracking:
    service_name = 'vehicle-reidentification-tracking'
    default_model_name = 'vehicle-reidentification-tracker'

    def __init__(self, model_name=None, model_weight='', model_variant='prototype',
                 device=0, synthetic_complexity=1):
        self.model_name = model_name or self.default_model_name
        self.model_weight = Context.get_file_path(model_weight) if model_weight else ''
        self.model_variant = model_variant
        self.device = device
        self.synthetic_complexity = max(1, int(synthetic_complexity))
        self.flops = 0

    def __call__(self, payload):
        detections = self._first_output(payload.get('inputs'), 'detections', default=[])
        vehicles = self._filter_detections(detections, {'car', 'bus', 'truck', 'motorcycle'})
        grouped = {}
        for detection in vehicles:
            grouped.setdefault(detection.get('object_id'), []).append(detection)

        tracklets = []
        for index, (object_id, object_detections) in enumerate(sorted(grouped.items())):
            object_detections = sorted(object_detections, key=lambda item: item.get('frame_id', 0))
            bboxes = [item.get('bbox', []) for item in object_detections]
            frames = [item.get('frame_id', 0) for item in object_detections]
            first_box = bboxes[0] if bboxes else [0, 0, 0, 0]
            last_box = bboxes[-1] if bboxes else first_box
            dx = ((last_box[0] + last_box[2]) - (first_box[0] + first_box[2])) / 2
            dy = ((last_box[1] + last_box[3]) - (first_box[1] + first_box[3])) / 2
            tracklets.append({
                'track_id': f'vehicle-{index}',
                'source_object_id': object_id,
                'category': object_detections[0].get('category', 'vehicle') if object_detections else 'vehicle',
                'frames': frames,
                'bboxes': bboxes,
                'embedding': [round((index + 1) * 0.01 * dim, 4) for dim in range(1, 9)],
                'speed_px_per_s': round((dx * dx + dy * dy) ** 0.5, 3),
                'direction': 'eastbound' if dx >= 0 else 'westbound',
            })

        outputs = {'vehicle_tracklets': tracklets}
        return self._wrap_result(payload, outputs, num_objects=len(tracklets))

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
    def _first_output(inputs, key, default=None):
        for content in (inputs or {}).values():
            if not isinstance(content, dict):
                continue
            outputs = content.get('outputs')
            if isinstance(outputs, dict) and key in outputs:
                return outputs[key]
        return default

    @staticmethod
    def _filter_detections(detections, categories):
        category_set = set(categories)
        return [detection for detection in detections if detection.get('category') in category_set]

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
