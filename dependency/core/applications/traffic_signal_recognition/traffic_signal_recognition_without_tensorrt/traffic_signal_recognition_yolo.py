import json
import os

class TrafficSignalRecognition:
    service_name = 'traffic-signal-recognition'
    default_model_name = 'traffic-signal-classifier'

    def __init__(self, weights='', device=0, confidence_threshold=0.25):
        self.model_name = self.default_model_name
        self.weights = weights
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.model = self._load_model(self.weights)
        self.flops = 0

    def _load_model(self, weight_path):
        if not weight_path:
            return None
        model = {
            'weight_path': weight_path,
            'exists': os.path.exists(weight_path),
            'loaded': False,
            'backend': 'upstream-bbox-rule',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            from ultralytics import YOLO

            detector = YOLO(weight_path)
            model.update({
                'loaded': True,
                'backend': 'ultralytics-yolo',
                'model': detector,
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        signals = self._infer_with_model(payload)
        inference_backend = self._model_backend()
        if signals is None:
            signals = self._fallback_signals(payload)
            inference_backend = 'upstream-bbox-rule'

        outputs = {'signals': signals}
        return self._wrap_result(payload, outputs, num_objects=len(signals),
                                 inference_backend=inference_backend)

    def _infer_with_model(self, payload):
        if not (self.model and self.model.get('loaded')):
            return None
        frames = payload.get('frames') or []
        if not frames:
            return []
        detector = self.model.get('model')
        signals = []
        for frame_id, frame in zip(self._frame_ids(payload), frames):
            try:
                results = detector.predict(
                    source=frame,
                    verbose=False,
                    conf=self.confidence_threshold,
                    device=self._ultralytics_device(),
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
                self.model['error'] = str(exc)
                return None
            if not results:
                continue
            result = results[0]
            names = result.names if hasattr(result, 'names') else getattr(detector, 'names', {})
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            xyxy = self._tensor_to_list(boxes.xyxy)
            scores = self._tensor_to_list(boxes.conf)
            classes = self._tensor_to_list(boxes.cls)
            for index, (box, score, class_index) in enumerate(zip(xyxy, scores, classes)):
                state = names.get(int(class_index), str(int(class_index))) if isinstance(names, dict) else str(int(class_index))
                signals.append({
                    'frame_id': int(frame_id),
                    'signal_id': f'traffic-signal-{int(frame_id)}-{index}',
                    'bbox': [int(round(value)) for value in box[:4]],
                    'type': 'traffic_light',
                    'state': state.replace('-', '_'),
                    'score': round(float(score), 4),
                    'model_label': state,
                })
        return signals

    def _fallback_signals(self, payload):
        detections = self._first_output(payload.get('inputs'), 'detections', default=[])
        signal_detections = self._filter_detections(detections, {'traffic_light', 'traffic_sign'})
        signals = []

        if signal_detections:
            for detection in signal_detections:
                category = detection.get('category')
                state = 'red' if category == 'traffic_light' else 'stop'
                signals.append({
                    'frame_id': detection.get('frame_id', 0),
                    'bbox': detection.get('bbox', []),
                    'type': category,
                    'state': state,
                    'score': round(float(detection.get('score', 0.8)), 3),
                })
        else:
            for frame_id in self._frame_ids(payload)[:1]:
                signals.append({
                    'frame_id': frame_id,
                    'bbox': [],
                    'type': 'traffic_light',
                    'state': 'red',
                    'score': 0.75,
                })
        return signals

    def _wrap_result(self, payload, outputs, num_objects=0, inference_backend='upstream-bbox-rule'):
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
    def _frame_ids(payload):
        hashes = (payload.get('task') or {}).get('hash_data') or []
        if hashes:
            return list(range(len(hashes)))
        return list(range(len(payload.get('frames') or [])))

    def _model_backend(self):
        return (self.model or {}).get('backend', 'upstream-bbox-rule')

    def _ultralytics_device(self):
        if self.device is None:
            return None
        if isinstance(self.device, str):
            return self.device
        try:
            import torch

            if not torch.cuda.is_available():
                return 'cpu'
        except Exception:
            return 'cpu'
        return int(self.device)

    @staticmethod
    def _tensor_to_list(value):
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'tolist'):
            return value.tolist()
        return list(value)

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
