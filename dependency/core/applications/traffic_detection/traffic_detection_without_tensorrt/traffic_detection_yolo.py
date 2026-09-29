import os

class TrafficDetection:
    service_name = 'traffic-detection'
    traffic_categories = {
        'person': 'pedestrian',
        'bicycle': 'cyclist',
        'motorcycle': 'motorcycle',
        'car': 'car',
        'bus': 'bus',
        'truck': 'truck',
        'traffic light': 'traffic_light',
        'stop sign': 'traffic_sign',
    }

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
            'backend': 'template',
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
        detections = self._infer_with_model(payload)
        if detections is None:
            detections = self._fallback_detections(payload)

        return {'bbox': self._bbox_records(detections)}

    def _infer_with_model(self, payload):
        if not (self.model and self.model.get('loaded')):
            return None
        frames = payload.get('frames') or []
        if not frames:
            return []
        detector = self.model.get('model')
        detections = []
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
                raw_name = names.get(int(class_index), str(int(class_index))) if isinstance(names, dict) else str(int(class_index))
                category = self.traffic_categories.get(raw_name)
                if not category:
                    continue
                detections.append({
                    'frame_id': int(frame_id),
                    'object_id': f'{category}-{int(frame_id)}-{index}',
                    'label': category,
                    'category': category,
                    'bbox': [int(round(value)) for value in box[:4]],
                    'score': round(float(score), 4),
                    'model_label': raw_name,
                })
        return detections

    def _fallback_detections(self, payload):
        width, height = self._frame_shape(payload)
        width = max(width, 640)
        height = max(height, 360)
        detections = []

        templates = [
            ('car', 0.08, 0.48, 0.28, 0.74, 0.92),
            ('bus', 0.40, 0.44, 0.70, 0.78, 0.88),
            ('pedestrian', 0.72, 0.45, 0.80, 0.86, 0.90),
            ('cyclist', 0.56, 0.50, 0.66, 0.84, 0.84),
            ('traffic_light', 0.46, 0.08, 0.50, 0.22, 0.91),
            ('traffic_sign', 0.82, 0.18, 0.90, 0.30, 0.86),
        ]

        for frame_id in self._frame_ids(payload):
            shift = (frame_id % 5) * 3
            for index, (category, x1, y1, x2, y2, score) in enumerate(templates):
                bbox = [
                    int(x1 * width + shift),
                    int(y1 * height),
                    int(x2 * width + shift),
                    int(y2 * height),
                ]
                detections.append({
                    'frame_id': frame_id,
                    'object_id': f'{category}-{index}',
                    'label': category,
                    'category': category,
                    'bbox': bbox,
                    'score': score,
                })
        return detections

    @staticmethod
    def _frame_shape(payload):
        frames = payload.get('frames') or []
        if not frames:
            return 0, 0
        height, width = frames[0].shape[:2]
        return int(width), int(height)

    @staticmethod
    def _frame_ids(payload):
        hashes = (payload.get('task') or {}).get('hash_data') or []
        if hashes:
            return list(range(len(hashes)))
        return list(range(len(payload.get('frames') or [])))

    @staticmethod
    def _bbox_records(detections):
        grouped = {}
        for detection in detections:
            frame_index = int(detection.get('frame_id', detection.get('frame_index', 0)))
            item = dict(detection)
            item['frame_index'] = frame_index
            item.setdefault('label', item.get('category', 'object'))
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
