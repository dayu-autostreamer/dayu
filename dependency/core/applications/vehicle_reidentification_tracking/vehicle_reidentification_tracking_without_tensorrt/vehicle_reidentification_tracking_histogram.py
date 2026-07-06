import json
import os

class VehicleReidentificationTracking:
    service_name = 'vehicle-reidentification-tracking'
    default_model_name = 'vehicle-reidentification-tracker'

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
            'checkpoint_loaded': False,
            'backend': 'histogram-iou-tracker',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch

            checkpoint = VehicleReidentificationTracking._torch_load(torch, weight_path)
            model['checkpoint_loaded'] = True
            if isinstance(checkpoint, dict):
                model['checkpoint_keys'] = sorted(str(key) for key in checkpoint.keys())[:8]
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        detections = self._first_output(payload.get('inputs'), 'detections', default=[])
        vehicles = self._filter_detections(detections, {'car', 'bus', 'truck', 'motorcycle'})
        tracks = self._associate_detections(payload, vehicles)

        tracklets = []
        for index, track in enumerate(tracks):
            object_detections = sorted(track['detections'], key=lambda item: item.get('frame_id', 0))
            bboxes = [item.get('bbox', []) for item in object_detections]
            frames = [item.get('frame_id', 0) for item in object_detections]
            first_box = bboxes[0] if bboxes else [0, 0, 0, 0]
            last_box = bboxes[-1] if bboxes else first_box
            dx = ((last_box[0] + last_box[2]) - (first_box[0] + first_box[2])) / 2
            dy = ((last_box[1] + last_box[3]) - (first_box[1] + first_box[3])) / 2
            tracklets.append({
                'track_id': f'vehicle-{index}',
                'source_object_id': object_detections[0].get('object_id') if object_detections else '',
                'category': object_detections[0].get('category', 'vehicle') if object_detections else 'vehicle',
                'frames': frames,
                'bboxes': bboxes,
                'embedding': [round(float(value), 4) for value in track.get('embedding', [])],
                'speed_px_per_s': round((dx * dx + dy * dy) ** 0.5, 3),
                'direction': 'eastbound' if dx >= 0 else 'westbound',
            })

        outputs = {'vehicle_tracklets': tracklets}
        return self._wrap_result(payload, outputs, num_objects=len(tracklets),
                                 inference_backend=self._model_backend())

    def _associate_detections(self, payload, detections):
        tracks = []
        for detection in sorted(detections, key=lambda item: (item.get('frame_id', 0), item.get('object_id', ''))):
            embedding = self._embedding_for_detection(payload, detection)
            best_track = None
            best_score = 0.0
            for track in tracks:
                if detection.get('category') != track.get('category'):
                    continue
                frame_gap = int(detection.get('frame_id', 0)) - int(track.get('last_frame', 0))
                if frame_gap < 0 or frame_gap > 8:
                    continue
                iou = self._bbox_iou(detection.get('bbox', []), track.get('last_bbox', []))
                similarity = self._cosine_similarity(embedding, track.get('embedding', []))
                score = 0.7 * iou + 0.3 * similarity
                if score > best_score:
                    best_score = score
                    best_track = track
            if best_track is None or best_score < 0.25:
                tracks.append({
                    'category': detection.get('category', 'vehicle'),
                    'detections': [detection],
                    'last_bbox': detection.get('bbox', []),
                    'last_frame': detection.get('frame_id', 0),
                    'embedding': embedding,
                })
            else:
                count = len(best_track['detections'])
                best_track['detections'].append(detection)
                best_track['last_bbox'] = detection.get('bbox', [])
                best_track['last_frame'] = detection.get('frame_id', 0)
                best_track['embedding'] = [
                    (old * count + new) / float(count + 1)
                    for old, new in zip(best_track.get('embedding', embedding), embedding)
                ]
        return tracks

    def _wrap_result(self, payload, outputs, num_objects=0, inference_backend='histogram-iou-tracker'):
        profile = {
            'num_objects': int(num_objects),
            'input_bytes': self._input_bytes(payload),
            'output_bytes': self._output_bytes(outputs),
            'frame_count': len(payload.get('frames') or []),
            'model_name': self.model_name,
            'model_weight': os.path.basename(self.weights) if self.weights else '',
            'model_weight_exists': bool(self.model and self.model.get('exists')),
            'model_loaded': bool(self.model and self.model.get('loaded')),
            'checkpoint_loaded': bool(self.model and self.model.get('checkpoint_loaded')),
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

    def _embedding_for_detection(self, payload, detection):
        crop = self._crop_for_detection(payload, detection)
        if crop is None:
            bbox = detection.get('bbox') or [0, 0, 0, 0]
            values = [float(value) for value in bbox[:4]]
            total = sum(abs(value) for value in values) or 1.0
            return [(value / total) for value in values] * 2
        try:
            import cv2
            import numpy as np

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [4, 4], [0, 180, 0, 256]).flatten()
            norm = np.linalg.norm(hist) or 1.0
            return (hist / norm).astype('float32').tolist()
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            if self.model is not None:
                self.model['error'] = str(exc)
            bbox = detection.get('bbox') or [0, 0, 0, 0]
            values = [float(value) for value in bbox[:4]]
            total = sum(abs(value) for value in values) or 1.0
            return [(value / total) for value in values] * 2

    @staticmethod
    def _bbox_iou(first, second):
        if len(first) != 4 or len(second) != 4:
            return 0.0
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_first = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
        area_second = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
        union = area_first + area_second - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _cosine_similarity(first, second):
        if not first or not second or len(first) != len(second):
            return 0.0
        numerator = sum(float(a) * float(b) for a, b in zip(first, second))
        norm_first = sum(float(a) * float(a) for a in first) ** 0.5
        norm_second = sum(float(b) * float(b) for b in second) ** 0.5
        denominator = norm_first * norm_second
        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def _crop_for_detection(payload, detection):
        frames = payload.get('frames') or []
        if not frames:
            return None
        frame_id = int(detection.get('frame_id', 0))
        frame_id = max(0, min(frame_id, len(frames) - 1))
        frame = frames[frame_id]
        bbox = detection.get('bbox') or []
        if len(bbox) != 4:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(value)) for value in bbox]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _model_backend(self):
        return (self.model or {}).get('backend', 'histogram-iou-tracker')

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
