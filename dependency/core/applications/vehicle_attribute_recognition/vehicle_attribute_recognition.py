import json
import os

from core.lib.common import Context


class VehicleAttributeRecognition:
    service_name = 'vehicle-attribute-recognition'
    default_model_name = 'vehicle-attribute-classifier'

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
        colors = ['white', 'black', 'silver', 'blue']
        attributes = []

        seen_objects = []
        for detection in vehicles:
            object_id = detection.get('object_id')
            if object_id in seen_objects:
                continue
            seen_objects.append(object_id)
            index = len(attributes)
            attributes.append({
                'object_id': object_id,
                'type': detection.get('category', 'vehicle'),
                'color': colors[index % len(colors)],
                'orientation': 'front-left' if index % 2 == 0 else 'rear-right',
                'confidence': round(float(detection.get('score', 0.8)), 3),
            })

        outputs = {'vehicle_attributes': attributes}
        return self._wrap_result(payload, outputs, num_objects=len(attributes))

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
