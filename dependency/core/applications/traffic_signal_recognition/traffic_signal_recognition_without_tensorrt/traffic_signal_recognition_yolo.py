import os

import cv2

from core.lib.common import LOGGER


class TrafficSignalRecognition:
    service_name = 'traffic-signal-recognition'
    signal_categories = {'traffic_light'}

    def __init__(self, weights='', device=0, confidence_threshold=0.25, temporal_reuse=True,
                 reuse_iou_threshold=0.40, reuse_hist_correlation=0.95, reuse_value_delta=0.08,
                 inference_batch_size=1, inference_imgsz=320):
        self.weights = weights
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.temporal_reuse = bool(temporal_reuse)
        self.reuse_iou_threshold = float(reuse_iou_threshold)
        self.reuse_hist_correlation = float(reuse_hist_correlation)
        self.reuse_value_delta = float(reuse_value_delta)
        self.inference_batch_size = max(1, int(inference_batch_size))
        self.inference_imgsz = max(32, int(inference_imgsz))
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

        detections_by_frame = {}
        for original_index, detection in enumerate(detections):
            frame_index = self._frame_index(detection)
            if frame_index is not None:
                detections_by_frame.setdefault(frame_index, []).append((original_index, detection))

        signals = []
        previous_frame_index = None
        previous_inferences = []
        inference_count = 0
        reuse_count = 0

        for frame_index in sorted(detections_by_frame):
            current_entries = []
            for original_index, detection in detections_by_frame[frame_index]:
                crop = self._crop_for_detection(frames, detection)
                if crop is None:
                    continue
                current_entries.append({
                    'original_index': original_index,
                    'detection': detection,
                    'crop': crop,
                    'signature': self._crop_signature(crop) if self.temporal_reuse else None,
                })

            if previous_frame_index is None or frame_index != previous_frame_index + 1:
                previous_inferences = []
            matches = self._match_previous_frame(previous_inferences, current_entries)
            current_inferences = []

            pending_indices = [
                current_index
                for current_index in range(len(current_entries))
                if current_index not in matches
            ]
            inferred = {}
            for offset in range(0, len(pending_indices), self.inference_batch_size):
                batch_indices = pending_indices[offset:offset + self.inference_batch_size]
                batch_crops = [current_entries[index]['crop'] for index in batch_indices]
                predictions = self._predict_crops(batch_crops)
                inferred.update(zip(batch_indices, predictions))
            inference_count += len(pending_indices)

            for current_index, entry in enumerate(current_entries):
                prediction = matches.get(current_index)
                if prediction is None:
                    prediction = inferred.get(current_index)
                    if prediction is not None and entry['signature'] is not None:
                        current_inferences.append({
                            'bbox': entry['detection'].get('bbox', []),
                            'signature': entry['signature'],
                            'prediction': prediction,
                        })
                else:
                    reuse_count += 1

                if prediction is None or not prediction[0]:
                    continue
                signals.append(self._signal_item(
                    entry['detection'], entry['original_index'], frame_index, prediction,
                ))

            previous_frame_index = frame_index
            previous_inferences = current_inferences

        processed_count = inference_count + reuse_count
        reuse_ratio = reuse_count / processed_count if processed_count else 0.0
        LOGGER.debug(
            f'Traffic signal recognition candidates={len(detections)} '
            f'inferences={inference_count} reused={reuse_count} reuse_ratio={reuse_ratio:.4f}'
        )
        signals.sort(key=lambda signal: signal['_original_index'])
        for signal in signals:
            signal.pop('_original_index')
        return signals

    def _predict_crop(self, crop):
        return self._predict_crops([crop])[0]

    def _predict_crops(self, crops):
        if not crops:
            return []
        detector = self.model.get('model')
        try:
            results = detector.predict(
                source=crops[0] if len(crops) == 1 else crops,
                verbose=False,
                conf=self.confidence_threshold,
                device=self._ultralytics_device(),
                imgsz=self.inference_imgsz,
            )
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
            self.model['error'] = str(exc)
            return [None] * len(crops)

        results = list(results or [])
        predictions = [
            self._best_state([result], detector)
            for result in results[:len(crops)]
        ]
        if len(predictions) < len(crops):
            predictions.extend([None] * (len(crops) - len(predictions)))
        return predictions

    @staticmethod
    def _signal_item(detection, original_index, frame_index, prediction):
        state, score, model_label = prediction
        output_frame_id = detection.get('frame_id')
        if output_frame_id is None:
            output_frame_id = detection.get('frame_index', frame_index)
        return {
            '_original_index': original_index,
            'frame_id': output_frame_id,
            'signal_id': f'traffic-signal-{frame_index}-{original_index}',
            'source_object_id': detection.get('object_id'),
            'bbox': detection.get('bbox', []),
            'label': 'traffic_light',
            'type': 'traffic_light',
            'text': state,
            'state': state,
            'score': round(float(score), 4),
            'source_score': round(float(detection.get('score', 0.0)), 4),
            'model_label': model_label,
        }

    def _match_previous_frame(self, previous_entries, current_entries):
        if not self.temporal_reuse or not previous_entries or not current_entries:
            return {}

        candidates = []
        for previous_index, previous in enumerate(previous_entries):
            for current_index, current in enumerate(current_entries):
                iou = self._bbox_iou(previous['bbox'], current['detection'].get('bbox', []))
                if iou < self.reuse_iou_threshold:
                    continue
                if not self._signatures_match(previous['signature'], current['signature']):
                    continue
                candidates.append((iou, previous_index, current_index))

        matches = {}
        used_previous = set()
        used_current = set()
        ordered_candidates = sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
        for _, previous_index, current_index in ordered_candidates:
            if previous_index in used_previous or current_index in used_current:
                continue
            matches[current_index] = previous_entries[previous_index]['prediction']
            used_previous.add(previous_index)
            used_current.add(current_index)
        return matches

    def _signatures_match(self, previous, current):
        if previous is None or current is None:
            return False
        histogram_correlation = cv2.compareHist(previous[0], current[0], cv2.HISTCMP_CORREL)
        value_delta = abs(previous[1] - current[1])
        return (
            histogram_correlation >= self.reuse_hist_correlation
            and value_delta <= self.reuse_value_delta
        )

    @staticmethod
    def _crop_signature(crop):
        try:
            resized = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            histogram = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).flatten()
            histogram_norm = cv2.norm(histogram, cv2.NORM_L1)
            if histogram_norm <= 0:
                return None
            histogram = (histogram / histogram_norm).astype('float32')
            mean_value = float(hsv[:, :, 2].mean()) / 255.0
            return histogram, mean_value
        except (cv2.error, ValueError, TypeError):
            return None

    @staticmethod
    def _bbox_iou(first_bbox, second_bbox):
        if len(first_bbox) != 4 or len(second_bbox) != 4:
            return 0.0
        first_x1, first_y1, first_x2, first_y2 = [float(value) for value in first_bbox]
        second_x1, second_y1, second_x2, second_y2 = [float(value) for value in second_bbox]
        intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
        intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
        intersection = intersection_width * intersection_height
        first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
        second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _frame_index(detection):
        frame_index = detection.get('frame_id')
        if frame_index is None:
            frame_index = detection.get('frame_index', 0)
        try:
            return int(frame_index)
        except (TypeError, ValueError):
            return None

    def _crop_for_detection(self, frames, detection):
        frame_id = self._frame_index(detection)
        if frame_id is None:
            return None
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
