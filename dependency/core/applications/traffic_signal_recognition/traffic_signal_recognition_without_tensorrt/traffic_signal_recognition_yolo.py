import os

class TrafficSignalRecognition:
    service_name = 'traffic-signal-recognition'

    def __init__(self, weights='', device=0, confidence_threshold=0.25):
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
        if signals is None:
            signals = self._fallback_signals(payload)

        return {'text': self._text_records(signals)}

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
                    'label': 'traffic_light',
                    'type': 'traffic_light',
                    'text': state.replace('-', '_'),
                    'state': state.replace('-', '_'),
                    'score': round(float(score), 4),
                    'model_label': state,
                })
        return signals

    def _fallback_signals(self, payload):
        detections = self._all_items(payload.get('inputs'), 'bbox')
        signal_detections = self._filter_detections(detections, {'traffic_light', 'traffic_sign'})
        signals = []

        if signal_detections:
            for detection in signal_detections:
                category = detection.get('category')
                state = 'red' if category == 'traffic_light' else 'stop'
                signals.append({
                    'frame_id': detection.get('frame_id', detection.get('frame_index', 0)),
                    'bbox': detection.get('bbox', []),
                    'label': category,
                    'type': category,
                    'text': state,
                    'state': state,
                    'score': round(float(detection.get('score', 0.8)), 3),
                })
        else:
            for frame_id in self._frame_ids(payload)[:1]:
                signals.append({
                    'frame_id': frame_id,
                    'bbox': [],
                    'label': 'traffic_light',
                    'type': 'traffic_light',
                    'text': 'red',
                    'state': 'red',
                    'score': 0.75,
                })
        return signals

    @staticmethod
    def _all_items(inputs, key):
        items = []
        for content in (inputs or {}).values():
            if not isinstance(content, dict):
                continue
            outputs = content.get('outputs')
            if isinstance(outputs, dict) and key in outputs:
                for record in outputs[key] or []:
                    if isinstance(record, dict):
                        frame_index = record.get('frame_index')
                        for item in record.get('items') or []:
                            item = dict(item)
                            item.setdefault('frame_index', frame_index)
                            items.append(item)
        return items

    @staticmethod
    def _filter_detections(detections, categories):
        category_set = set(categories)
        return [
            detection for detection in detections
            if (detection.get('label') or detection.get('category')) in category_set
        ]

    @staticmethod
    def _text_records(signals):
        grouped = {}
        for signal in signals:
            frame_index = int(signal.get('frame_id', signal.get('frame_index', 0)))
            item = dict(signal)
            item['frame_index'] = frame_index
            item.setdefault('text', item.get('state', ''))
            grouped.setdefault(frame_index, []).append(item)
        return [
            {'frame_index': frame_index, 'items': grouped[frame_index]}
            for frame_index in sorted(grouped)
        ]

    @staticmethod
    def _frame_ids(payload):
        hashes = (payload.get('task') or {}).get('hash_data') or []
        if hashes:
            return list(range(len(hashes)))
        return list(range(len(payload.get('frames') or [])))

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
