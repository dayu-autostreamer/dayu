import os

class TrafficSignalRecognition:
    service_name = 'traffic-signal-recognition'
    signal_categories = {'traffic_light'}

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
            'backend': 'upstream-traffic-light-crop',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            from ultralytics import YOLO

            detector = YOLO(weight_path)
            model.update({
                'loaded': True,
                'backend': 'ultralytics-yolo-crop',
                'model': detector,
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        signals = self._infer_with_model(payload)
        return {'text': self._text_records(signals)}

    def _infer_with_model(self, payload):
        if not (self.model and self.model.get('loaded')):
            return []
        frames = payload.get('frames') or []
        if not frames:
            return []
        detections = self._filter_detections(self._all_items(payload.get('inputs'), 'bbox'), self.signal_categories)
        if not detections:
            return []
        detector = self.model.get('model')
        signals = []
        for index, detection in enumerate(detections):
            crop = self._crop_for_detection(frames, detection)
            if crop is None:
                continue
            try:
                results = detector.predict(
                    source=crop,
                    verbose=False,
                    conf=self.confidence_threshold,
                    device=self._ultralytics_device(),
                )
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
                self.model['error'] = str(exc)
                continue
            state, score, model_label = self._best_state(results, detector)
            if not state:
                continue
            frame_id = detection.get('frame_id', detection.get('frame_index', 0))
            signals.append({
                'frame_id': frame_id,
                'signal_id': f'traffic-signal-{int(frame_id)}-{index}',
                'source_object_id': detection.get('object_id'),
                'bbox': detection.get('bbox', []),
                'label': 'traffic_light',
                'type': 'traffic_light',
                'text': state,
                'state': state,
                'score': round(float(score), 4),
                'source_score': round(float(detection.get('score', 0.0)), 4),
                'model_label': model_label,
            })
        return signals

    def _crop_for_detection(self, frames, detection):
        frame_id = int(detection.get('frame_id', detection.get('frame_index', 0)))
        if frame_id < 0 or frame_id >= len(frames):
            return None
        bbox = detection.get('bbox') or []
        if len(bbox) != 4:
            return None
        frame = frames[frame_id]
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _best_state(self, results, detector):
        if not results:
            return '', 0.0, ''
        result = results[0]
        names = result.names if hasattr(result, 'names') else getattr(detector, 'names', {})
        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return '', 0.0, ''
        scores = self._tensor_to_list(boxes.conf)
        classes = self._tensor_to_list(boxes.cls)
        if not scores or not classes:
            return '', 0.0, ''
        best_index = max(range(len(scores)), key=lambda index: float(scores[index]))
        class_index = int(classes[best_index])
        model_label = names.get(class_index, str(class_index)) if isinstance(names, dict) else str(class_index)
        state = model_label.replace('-', '_')
        return state, float(scores[best_index]), model_label

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
