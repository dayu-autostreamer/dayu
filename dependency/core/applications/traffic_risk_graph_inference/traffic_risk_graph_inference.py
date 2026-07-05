import json
import os

from core.lib.common import Context


class TrafficRiskGraphInference:
    service_name = 'traffic-risk-graph-inference'
    default_model_name = 'traffic-risk-graph-model'

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
        trajectories = self._all_outputs(inputs, 'trajectory_predictions')
        intents = self._all_outputs(inputs, 'pedestrian_cyclist_intents')
        signals = self._all_outputs(inputs, 'signals')
        red_signal = any(signal.get('state') == 'red' for signal in signals)

        events = []
        if trajectories and intents:
            events.append({
                'type': 'near_miss',
                'start_time': 0.5,
                'end_time': 2.0,
                'entities': [
                    trajectories[0].get('track_id'),
                    intents[0].get('person_id'),
                ],
                'risk_score': 0.87,
                'explanation': 'predicted vehicle trajectory intersects pedestrian or cyclist intent region',
            })
        if red_signal and trajectories:
            events.append({
                'type': 'red_light_violation_risk',
                'start_time': 0.0,
                'end_time': 1.5,
                'entities': [trajectories[0].get('track_id')],
                'risk_score': 0.74,
                'explanation': 'vehicle trajectory remains active while traffic signal is red',
            })

        graph_summary = {
            'entity_count': len(trajectories) + len(intents),
            'relation_count': len(trajectories) * max(len(intents), 1),
            'signal_count': len(signals),
        }
        outputs = {
            'events': events,
            'graph_summary': graph_summary,
        }
        return self._wrap_result(payload, outputs, num_objects=graph_summary['entity_count'])

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
