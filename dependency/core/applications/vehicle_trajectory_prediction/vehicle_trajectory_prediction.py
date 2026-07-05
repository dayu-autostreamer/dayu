import json
import os

from core.lib.common import Context


class VehicleTrajectoryPrediction:
    service_name = 'vehicle-trajectory-prediction'
    default_model_name = 'vehicle-trajectory-predictor'

    def __init__(self, model_name=None, model_weight='', model_variant='prototype',
                 device=0, synthetic_complexity=1):
        self.model_name = model_name or self.default_model_name
        self.model_weight = Context.get_file_path(model_weight) if model_weight else ''
        self.model_variant = model_variant
        self.device = device
        self.synthetic_complexity = max(1, int(synthetic_complexity))
        self.flops = 0

    def __call__(self, payload):
        inputs = payload.get('inputs')
        tracklets = self._all_outputs(inputs, 'vehicle_tracklets')
        attributes = self._all_outputs(inputs, 'vehicle_attributes')
        road_context = self._first_output(inputs, 'drivable_area', default=[])
        attribute_by_object = {item.get('object_id'): item for item in attributes}

        predictions = []
        for tracklet in tracklets:
            bboxes = tracklet.get('bboxes') or [[0, 0, 0, 0]]
            last_box = bboxes[-1]
            center_x = (last_box[0] + last_box[2]) / 2
            center_y = (last_box[1] + last_box[3]) / 2
            speed = float(tracklet.get('speed_px_per_s', 0))
            attr = attribute_by_object.get(tracklet.get('source_object_id'), {})
            future_points = [
                [round(center_x + speed * step * 0.2, 2), round(center_y, 2), round(step * 0.5, 2)]
                for step in range(1, 5)
            ]
            predictions.append({
                'track_id': tracklet.get('track_id'),
                'vehicle_type': attr.get('type', tracklet.get('category', 'vehicle')),
                'future_trajectories': [{'prob': 0.72, 'points': future_points}],
                'abnormal_stop_prob': 0.18 if speed < 1 else 0.05,
                'road_context_available': bool(road_context),
            })

        outputs = {'trajectory_predictions': predictions}
        return self._wrap_result(payload, outputs, num_objects=len(predictions))

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
    def _all_outputs(inputs, key):
        values = []
        for content in (inputs or {}).values():
            if not isinstance(content, dict):
                continue
            outputs = content.get('outputs')
            if isinstance(outputs, dict) and key in outputs:
                value = outputs[key]
                if isinstance(value, list):
                    values.extend(value)
                else:
                    values.append(value)
        return values

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
