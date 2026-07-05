import json
import os

from core.lib.common import Context


class PedestrianCyclistIntentRecognition:
    service_name = 'pedestrian-cyclist-intent-recognition'
    default_model_name = 'pedestrian-cyclist-intent-classifier'

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
        skeletons = self._all_outputs(inputs, 'skeletons')
        crosswalk_regions = self._first_output(inputs, 'crosswalk_regions', default=[])
        intents = []

        for skeleton in skeletons:
            category = skeleton.get('category', 'pedestrian')
            action = 'crossing' if category == 'pedestrian' else 'riding'
            confidence = 0.82 if crosswalk_regions else 0.68
            intents.append({
                'person_id': skeleton.get('person_id'),
                'category': category,
                'action': action,
                'intent': 'likely_to_cross' if action == 'crossing' else 'likely_to_enter_lane',
                'confidence': confidence,
                'time_window': [0.0, 2.0],
                'road_context_available': bool(crosswalk_regions),
            })

        outputs = {'pedestrian_cyclist_intents': intents}
        return self._wrap_result(payload, outputs, num_objects=len(intents))

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
