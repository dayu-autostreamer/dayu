import json
import os

class TrafficRiskGraphInference:
    service_name = 'traffic-risk-graph-inference'
    default_model_name = 'traffic-risk-graph-model'

    def __init__(self, weights='', device=0):
        self.model_name = self.default_model_name
        self.weights = weights
        self.device = device
        self.model = self._load_model(self.weights)
        self.flops = 0

    def _load_model(self, weight_path):
        if not weight_path:
            return None
        model = {
            'weight_path': weight_path,
            'exists': os.path.exists(weight_path),
            'loaded': False,
            'backend': 'rule-risk-graph',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch
            from torch import nn

            class RiskMLP(nn.Module):
                def __init__(self, input_dim, hidden_dim=96, num_classes=3):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, num_classes),
                    )

                def forward(self, x):
                    return self.net(x)

            checkpoint = self._torch_load(torch, weight_path)
            architecture = checkpoint.get('architecture', {})
            labels = architecture.get('labels') or ['low', 'medium', 'high']
            network = RiskMLP(
                input_dim=int(architecture.get('input_dim', 29)),
                hidden_dim=int(architecture.get('hidden_dim', 96)),
                num_classes=len(labels),
            )
            network.load_state_dict(checkpoint['model_state_dict'])
            device = self._torch_device(torch)
            network.to(device)
            network.eval()
            model.update({
                'loaded': True,
                'backend': 'dota-risk-mlp',
                'model': network,
                'torch': torch,
                'device': device,
                'architecture': architecture,
                'labels': list(labels),
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        inputs = payload.get('inputs')
        trajectories = self._all_outputs(inputs, 'trajectory_predictions')
        intents = self._all_outputs(inputs, 'pedestrian_cyclist_intents')
        signals = self._all_outputs(inputs, 'signals')
        red_signal = any(signal.get('state') == 'red' for signal in signals)
        events = self._rule_events(trajectories, intents, red_signal)
        risk_level, risk_confidence = self._infer_risk(payload, trajectories, intents, signals, events)
        events = self._score_events(events, risk_level, risk_confidence)

        graph_summary = {
            'entity_count': len(trajectories) + len(intents),
            'relation_count': len(trajectories) * max(len(intents), 1),
            'signal_count': len(signals),
            'risk_level': risk_level,
            'risk_confidence': risk_confidence,
        }
        outputs = {
            'events': events,
            'graph_summary': graph_summary,
        }
        return self._wrap_result(payload, outputs, num_objects=graph_summary['entity_count'],
                                 inference_backend=self._model_backend())

    @staticmethod
    def _rule_events(trajectories, intents, red_signal):
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
        return events

    def _infer_risk(self, payload, trajectories, intents, signals, events):
        if not (self.model and self.model.get('loaded')):
            return self._fallback_risk_level(events)
        torch = self.model['torch']
        try:
            features = self._risk_features(payload, trajectories, intents, signals, events)
            with torch.no_grad():
                tensor = torch.tensor([features], dtype=torch.float32, device=self.model['device'])
                logits = self.model['model'](tensor)
                probs = torch.softmax(logits, dim=1)[0]
                score, index = torch.max(probs, dim=0)
            label = self.model.get('labels', ['low', 'medium', 'high'])[int(index.item())]
            return label, round(float(score.item()), 4)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
            self.model['error'] = str(exc)
            return self._fallback_risk_level(events)

    def _risk_features(self, payload, trajectories, intents, signals, events):
        architecture = self.model.get('architecture') or {}
        input_dim = int(architecture.get('input_dim', 29))
        anomaly_classes = architecture.get('anomaly_classes') or []
        frame_count = max(1, len(payload.get('frames') or []))
        duration = frame_count / 30.0
        event_count = len(events)
        anomaly_ratio = min(1.0, event_count / 3.0)
        numeric = [
            duration / 300.0,
            0.0,
            min(1.0, max((event.get('end_time', 0.0) for event in events), default=0.0) / max(duration, 1.0)),
            anomaly_ratio,
            min(1.0, (len(trajectories) + len(intents) + len(signals)) / 20.0),
        ]
        event_text = ' '.join(event.get('type', '') for event in events).lower()
        prefix_features = [
            0.0,
            1.0 if trajectories else 0.0,
            1.0 if any(intent.get('category') == 'pedestrian' for intent in intents) else 0.0,
            1.0 if any(intent.get('category') == 'cyclist' for intent in intents) else 0.0,
        ]
        selected_class = self._select_anomaly_class(event_text, anomaly_classes)
        one_hot = [1.0 if name == selected_class else 0.0 for name in anomaly_classes]
        features = numeric + prefix_features + one_hot
        if len(features) < input_dim:
            features.extend([0.0] * (input_dim - len(features)))
        return features[:input_dim]

    @staticmethod
    def _select_anomaly_class(event_text, anomaly_classes):
        if not anomaly_classes:
            return None
        terms = ['collision', 'hit', 'crash'] if 'near_miss' in event_text else ['red', 'traffic', 'vehicle']
        for anomaly_class in anomaly_classes:
            lowered = anomaly_class.lower()
            if any(term in lowered for term in terms):
                return anomaly_class
        return anomaly_classes[0]

    @staticmethod
    def _fallback_risk_level(events):
        if any(event.get('type') == 'near_miss' for event in events):
            return 'high', 0.87
        if events:
            return 'medium', 0.74
        return 'low', 0.66

    @staticmethod
    def _score_events(events, risk_level, risk_confidence):
        if not events:
            return events
        level_floor = {'low': 0.35, 'medium': 0.62, 'high': 0.82}.get(risk_level, 0.5)
        score = max(level_floor, float(risk_confidence))
        updated = []
        for event in events:
            item = dict(event)
            item['risk_score'] = round(score, 4)
            item['risk_level'] = risk_level
            updated.append(item)
        return updated

    def _wrap_result(self, payload, outputs, num_objects=0, inference_backend='rule-risk-graph'):
        profile = {
            'num_objects': int(num_objects),
            'input_bytes': self._input_bytes(payload),
            'output_bytes': self._output_bytes(outputs),
            'frame_count': len(payload.get('frames') or []),
            'model_name': self.model_name,
            'model_weight': os.path.basename(self.weights) if self.weights else '',
            'model_weight_exists': bool(self.model and self.model.get('exists')),
            'model_loaded': bool(self.model and self.model.get('loaded')),
            'inference_backend': inference_backend,
            'model_error': (self.model or {}).get('error', ''),
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

    def _model_backend(self):
        return (self.model or {}).get('backend', 'rule-risk-graph')

    def _torch_device(self, torch):
        if isinstance(self.device, str):
            return torch.device(self.device if self.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            return torch.device(f'cuda:{int(self.device)}')
        return torch.device('cpu')

    @staticmethod
    def _torch_load(torch, weight_path):
        try:
            return torch.load(weight_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(weight_path, map_location='cpu')

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
