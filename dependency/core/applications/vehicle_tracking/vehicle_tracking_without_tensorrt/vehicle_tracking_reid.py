import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class _Track:
    track_id: str
    label: str
    category: str
    source_object_id: str
    state: np.ndarray
    covariance: np.ndarray
    embedding: List[float]
    detections: List[dict] = field(default_factory=list)
    hits: int = 0
    missed: int = 0
    last_frame: int = 0
    score: float = 0.0


class VehicleTracking:
    service_name = 'vehicle-tracking'
    vehicle_categories = {'car', 'bus', 'truck', 'motorcycle'}

    def __init__(self, weights='', device=0, high_score_threshold=0.35, low_score_threshold=0.10,
                 new_track_threshold=0.35, match_score_threshold=0.35, secondary_iou_threshold=0.25,
                 max_age=8, embedding_weight=0.45, iou_weight=0.45, motion_weight=0.10):
        self.weights = weights
        self.device = device
        self.high_score_threshold = float(high_score_threshold)
        self.low_score_threshold = float(low_score_threshold)
        self.new_track_threshold = float(new_track_threshold)
        self.match_score_threshold = float(match_score_threshold)
        self.secondary_iou_threshold = float(secondary_iou_threshold)
        self.max_age = int(max_age)
        self.embedding_weight = float(embedding_weight)
        self.iou_weight = float(iou_weight)
        self.motion_weight = float(motion_weight)
        self._next_track_index = 0
        self.model = self._load_model(self.weights)
        self.flops = 0.301 if self.model and self.model.get('loaded') else 0

    def _load_model(self, weight_path):
        if not weight_path:
            return None
        model = {
            'weight_path': weight_path,
            'exists': os.path.exists(weight_path),
            'loaded': False,
            'checkpoint_loaded': False,
            'backend': 'mobilenetv2-reid-kalman',
            'architecture': 'mobilenet_v2',
            'feature_dim': 1280,
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch
            from torchvision.models import mobilenet_v2

            checkpoint = VehicleTracking._torch_load(torch, weight_path)
            model['checkpoint_loaded'] = True
            state_dict = self._state_dict_from_checkpoint(checkpoint)
            network = mobilenet_v2(weights=None)
            network.load_state_dict(state_dict, strict=False)
            device = self._torch_device(torch)
            network.to(device)
            network.eval()
            model.update({
                'loaded': True,
                'network': network,
                'torch': torch,
                'device': device,
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        detections = self._prepare_detections(payload)
        tracks = self._build_tracks(payload, detections)
        tracklets = [self._track_to_item(track) for track in tracks if track.detections]
        return {'track': [{'frame_index': None, 'items': tracklets}]}

    def _prepare_detections(self, payload):
        detections = self._all_items(payload.get('inputs'), 'bbox')
        vehicles = self._filter_detections(detections, self.vehicle_categories)
        prepared = []
        for detection in vehicles:
            item = dict(detection)
            frame_index = int(item.get('frame_id', item.get('frame_index', 0)))
            item['frame_index'] = frame_index
            item.setdefault('frame_id', frame_index)
            item['score'] = float(item.get('score', 1.0))
            item['embedding'] = self._embedding_for_detection(payload, item)
            prepared.append(item)
        return sorted(prepared, key=lambda item: (item.get('frame_index', 0), item.get('object_id', '')))

    def _build_tracks(self, payload, detections):
        detections_by_frame = self._group_by_frame(detections)
        frame_ids = self._frame_ids(payload, detections_by_frame)
        active_tracks = []
        finished_tracks = []

        for frame_index in frame_ids:
            for track in active_tracks:
                self._predict_track(track)

            frame_detections = detections_by_frame.get(frame_index, [])
            high_detections = [
                detection for detection in frame_detections
                if detection.get('score', 0.0) >= self.high_score_threshold
            ]
            low_detections = [
                detection for detection in frame_detections
                if self.low_score_threshold <= detection.get('score', 0.0) < self.high_score_threshold
            ]

            matched_track_ids = set()
            matched_high_ids = set()
            for track, detection_index in self._match(active_tracks, high_detections, use_appearance=True):
                self._update_track(track, high_detections[detection_index], frame_index)
                matched_track_ids.add(id(track))
                matched_high_ids.add(detection_index)

            remaining_tracks = [track for track in active_tracks if id(track) not in matched_track_ids]
            for track, detection_index in self._match(remaining_tracks, low_detections, use_appearance=False):
                self._update_track(track, low_detections[detection_index], frame_index)
                matched_track_ids.add(id(track))

            next_active_tracks = []
            for track in active_tracks:
                if id(track) not in matched_track_ids:
                    track.missed += 1
                if track.missed <= self.max_age:
                    next_active_tracks.append(track)
                else:
                    finished_tracks.append(track)
            active_tracks = next_active_tracks

            for detection_index, detection in enumerate(high_detections):
                if detection_index not in matched_high_ids and detection.get('score', 0.0) >= self.new_track_threshold:
                    active_tracks.append(self._new_track(detection, frame_index))

        return finished_tracks + active_tracks

    def _match(self, tracks, detections, use_appearance=True):
        if not tracks or not detections:
            return []

        scores = np.full((len(tracks), len(detections)), -1.0, dtype=np.float32)
        for track_index, track in enumerate(tracks):
            for detection_index, detection in enumerate(detections):
                scores[track_index, detection_index] = self._association_score(
                    track, detection, use_appearance=use_appearance
                )

        threshold = self.match_score_threshold if use_appearance else self.secondary_iou_threshold
        matches = []
        try:
            from scipy.optimize import linear_sum_assignment

            row_indexes, column_indexes = linear_sum_assignment(1.0 - scores)
            candidate_pairs = zip(row_indexes.tolist(), column_indexes.tolist())
        except Exception:  # pragma: no cover - scipy is optional in local lightweight test envs.
            candidate_pairs = [
                (row, column)
                for row in range(scores.shape[0])
                for column in range(scores.shape[1])
            ]
            candidate_pairs = sorted(candidate_pairs, key=lambda pair: scores[pair[0], pair[1]], reverse=True)

        used_tracks = set()
        used_detections = set()
        for track_index, detection_index in candidate_pairs:
            if track_index in used_tracks or detection_index in used_detections:
                continue
            if scores[track_index, detection_index] < threshold:
                continue
            matches.append((tracks[track_index], detection_index))
            used_tracks.add(track_index)
            used_detections.add(detection_index)
        return matches

    def _association_score(self, track, detection, use_appearance=True):
        if self._label(detection) != track.label:
            return -1.0
        predicted_bbox = self._state_to_bbox(track.state)
        detection_bbox = detection.get('bbox') or []
        iou = self._bbox_iou(predicted_bbox, detection_bbox)
        if not use_appearance:
            return iou

        appearance = max(0.0, self._cosine_similarity(track.embedding, detection.get('embedding', [])))
        motion = self._motion_score(predicted_bbox, detection_bbox)
        return (
            self.iou_weight * iou
            + self.embedding_weight * appearance
            + self.motion_weight * motion
        )

    def _new_track(self, detection, frame_index):
        state = self._bbox_to_state(detection.get('bbox') or [0, 0, 0, 0])
        covariance = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0]).astype(np.float32)
        track = _Track(
            track_id=f'vehicle-{self._next_track_index}',
            label=self._label(detection),
            category=detection.get('category', self._label(detection)),
            source_object_id=detection.get('object_id', ''),
            state=state,
            covariance=covariance,
            embedding=list(detection.get('embedding') or []),
            last_frame=frame_index,
            score=float(detection.get('score', 0.0)),
        )
        self._next_track_index += 1
        self._update_track(track, detection, frame_index, update_filter=False)
        return track

    def _predict_track(self, track):
        transition = np.eye(8, dtype=np.float32)
        transition[0, 4] = 1.0
        transition[1, 5] = 1.0
        transition[2, 6] = 1.0
        transition[3, 7] = 1.0
        process_noise = np.diag([1.0, 1.0, 0.4, 0.4, 2.0, 2.0, 0.8, 0.8]).astype(np.float32)
        track.state = transition @ track.state
        track.covariance = transition @ track.covariance @ transition.T + process_noise

    def _update_track(self, track, detection, frame_index, update_filter=True):
        if update_filter:
            self._update_kalman(track, detection.get('bbox') or [0, 0, 0, 0])
        count = max(track.hits, 0)
        embedding = detection.get('embedding') or track.embedding
        if track.embedding and embedding and len(track.embedding) == len(embedding):
            track.embedding = [
                (old * count + new) / float(count + 1)
                for old, new in zip(track.embedding, embedding)
            ]
        else:
            track.embedding = list(embedding or [])
        stored_detection = dict(detection)
        stored_detection.pop('embedding', None)
        stored_detection['frame_index'] = frame_index
        track.detections.append(stored_detection)
        track.hits += 1
        track.missed = 0
        track.last_frame = frame_index
        track.score = float(detection.get('score', track.score))

    def _update_kalman(self, track, bbox):
        measurement = self._bbox_to_state(bbox)[:4]
        observation = np.zeros((4, 8), dtype=np.float32)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 3] = 1.0
        measurement_noise = np.diag([8.0, 8.0, 6.0, 6.0]).astype(np.float32)
        projected = observation @ track.state
        innovation = measurement - projected
        covariance = observation @ track.covariance @ observation.T + measurement_noise
        try:
            kalman_gain = track.covariance @ observation.T @ np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            kalman_gain = track.covariance @ observation.T @ np.linalg.pinv(covariance)
        track.state = track.state + kalman_gain @ innovation
        identity = np.eye(8, dtype=np.float32)
        track.covariance = (identity - kalman_gain @ observation) @ track.covariance

    def _embedding_for_detection(self, payload, detection):
        crop = self._crop_for_detection(payload, detection)
        if crop is None:
            return self._bbox_embedding(detection)
        if self.model and self.model.get('loaded'):
            embedding = self._model_embedding(crop)
            if embedding:
                return embedding
        return self._histogram_embedding(crop, detection)

    def _model_embedding(self, crop):
        try:
            torch = self.model['torch']
            network = self.model['network']
            device = self.model['device']
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (rgb - mean) / std
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                features = network.features(tensor)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                features = torch.flatten(features, 1)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            return features.squeeze(0).detach().cpu().float().tolist()
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
            if self.model is not None:
                self.model['error'] = str(exc)
            return []

    @staticmethod
    def _histogram_embedding(crop, detection):
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [4, 4], [0, 180, 0, 256]).flatten()
            norm = np.linalg.norm(hist) or 1.0
            return (hist / norm).astype('float32').tolist()
        except Exception:
            return VehicleTracking._bbox_embedding(detection)

    @staticmethod
    def _bbox_embedding(detection):
        bbox = detection.get('bbox') or [0, 0, 0, 0]
        values = [float(value) for value in bbox[:4]]
        total = sum(abs(value) for value in values) or 1.0
        return [(value / total) for value in values] * 2

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
        return [detection for detection in detections if VehicleTracking._label(detection) in category_set]

    @staticmethod
    def _group_by_frame(detections):
        grouped = {}
        for detection in detections:
            grouped.setdefault(int(detection.get('frame_index', 0)), []).append(detection)
        return grouped

    @staticmethod
    def _frame_ids(payload, detections_by_frame):
        frame_count = len(payload.get('frames') or [])
        hashes = (payload.get('task') or {}).get('hash_data') or []
        frame_count = max(frame_count, len(hashes), 0)
        frame_ids = set(range(frame_count))
        frame_ids.update(detections_by_frame.keys())
        return sorted(frame_ids)

    @staticmethod
    def _label(item):
        return item.get('label') or item.get('category') or ''

    @staticmethod
    def _crop_for_detection(payload, detection):
        frames = payload.get('frames') or []
        if not frames:
            return None
        frame_id = int(detection.get('frame_id', detection.get('frame_index', 0)))
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
    def _motion_score(predicted_bbox, detection_bbox):
        if len(predicted_bbox) != 4 or len(detection_bbox) != 4:
            return 0.0
        px, py = VehicleTracking._bbox_center(predicted_bbox)
        dx, dy = VehicleTracking._bbox_center(detection_bbox)
        distance = ((px - dx) ** 2 + (py - dy) ** 2) ** 0.5
        width = max(abs(float(predicted_bbox[2]) - float(predicted_bbox[0])), 1.0)
        height = max(abs(float(predicted_bbox[3]) - float(predicted_bbox[1])), 1.0)
        scale = (width ** 2 + height ** 2) ** 0.5
        return max(0.0, 1.0 - min(distance / scale, 1.0))

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
    def _bbox_to_state(bbox):
        if len(bbox) != 4:
            bbox = [0, 0, 0, 0]
        x1, y1, x2, y2 = [float(value) for value in bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        return np.asarray([center_x, center_y, width, height, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _state_to_bbox(state):
        center_x, center_y, width, height = [float(value) for value in state[:4]]
        width = max(1.0, width)
        height = max(1.0, height)
        return [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ]

    @staticmethod
    def _bbox_center(bbox):
        return (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )

    def _track_to_item(self, track):
        object_detections = sorted(track.detections, key=lambda item: item.get('frame_index', 0))
        bboxes = [item.get('bbox', []) for item in object_detections]
        frames = [item.get('frame_index', 0) for item in object_detections]
        first_box = bboxes[0] if bboxes else [0, 0, 0, 0]
        last_box = bboxes[-1] if bboxes else first_box
        dx = ((last_box[0] + last_box[2]) - (first_box[0] + first_box[2])) / 2
        dy = ((last_box[1] + last_box[3]) - (first_box[1] + first_box[3])) / 2
        frame_span = max((frames[-1] - frames[0]) if len(frames) >= 2 else 1, 1)
        speed = ((dx * dx + dy * dy) ** 0.5) / frame_span
        return {
            'track_id': track.track_id,
            'source_object_id': track.source_object_id,
            'label': track.label,
            'category': track.category,
            'frames': frames,
            'bboxes': bboxes,
            'score': round(float(track.score), 4),
            'hit_count': int(track.hits),
            'missed_count': int(track.missed),
            'embedding': [round(float(value), 4) for value in track.embedding[:16]],
            'speed_px_per_s': round(float(speed), 3),
            'direction': 'eastbound' if dx >= 0 else 'westbound',
        }

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint):
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get('state_dict')
                or checkpoint.get('model_state_dict')
                or checkpoint.get('model_state')
                or checkpoint
            )
        if not isinstance(state_dict, dict):
            raise ValueError('vehicle tracking checkpoint must be a state_dict-like dictionary')
        cleaned = {}
        for key, value in state_dict.items():
            key = str(key)
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned[key] = value
        return cleaned

    def _torch_device(self, torch):
        if isinstance(self.device, str):
            if self.device == 'cpu':
                return torch.device('cpu')
            if self.device.startswith('cuda') and torch.cuda.is_available():
                return torch.device(self.device)
        try:
            index = int(self.device)
            if torch.cuda.is_available():
                return torch.device(f'cuda:{index}')
        except Exception:
            pass
        return torch.device('cpu')

    @staticmethod
    def _torch_load(torch, weight_path):
        try:
            return torch.load(weight_path, map_location='cpu', weights_only=False)
        except TypeError:
            return torch.load(weight_path, map_location='cpu')
