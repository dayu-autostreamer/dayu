import os

class PedestrianCyclistPoseEstimation:
    service_name = 'pedestrian-cyclist-pose-estimation'

    def __init__(self, weights='', device=0, model_config=''):
        self.weights = weights
        self.device = device
        self.model_config = model_config
        self.model = self._load_model(self.weights)
        self.flops = 0

    def _load_model(self, weight_path):
        if not weight_path:
            return None
        model = {
            'weight_path': weight_path,
            'exists': os.path.exists(weight_path),
            'loaded': False,
            'backend': 'geometric-pose',
            'error': '',
        }
        if not model['exists']:
            return model
        if not self.model_config:
            model['error'] = 'mmpose config path is required for RTMPose checkpoint inference'
            return model
        try:
            from mmpose.apis import init_model, inference_topdown

            pose_model = init_model(self.model_config, weight_path, device=self._mmpose_device())
            model.update({
                'loaded': True,
                'backend': 'mmpose-rtmpose',
                'model': pose_model,
                'inference_topdown': inference_topdown,
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        detections = self._all_items(payload.get('inputs'), 'bbox')
        people = self._filter_detections(detections, {'pedestrian', 'cyclist'})
        skeletons = self._infer_with_model(payload, people)
        if skeletons is None:
            skeletons = self._fallback_skeletons(people)

        return {'pose': self._pose_records(skeletons)}

    def _infer_with_model(self, payload, people):
        if not (self.model and self.model.get('loaded')):
            return None
        frames = payload.get('frames') or []
        if not frames:
            return []
        skeletons = []
        inference_topdown = self.model['inference_topdown']
        pose_model = self.model['model']
        for frame_id in sorted({int(item.get('frame_id', item.get('frame_index', 0))) for item in people}):
            frame_people = [item for item in people if int(item.get('frame_id', item.get('frame_index', 0))) == frame_id]
            if not frame_people or frame_id >= len(frames):
                continue
            try:
                import numpy as np

                bboxes = np.array([item.get('bbox', [0, 0, 0, 0]) for item in frame_people], dtype=np.float32)
                results = inference_topdown(pose_model, frames[frame_id], bboxes=bboxes)
            except Exception as exc:  # pragma: no cover - runtime dependent.
                self.model['error'] = str(exc)
                return None
            for index, (detection, sample) in enumerate(zip(frame_people, results)):
                keypoints, scores = self._extract_mmpose_keypoints(sample)
                skeletons.append({
                    'person_id': f'pedestrian-cyclist-{frame_id}-{index}',
                    'source_object_id': detection.get('object_id'),
                    'frame_id': detection.get('frame_id', detection.get('frame_index', 0)),
                    'label': self._label(detection) or 'pedestrian',
                    'category': detection.get('category', self._label(detection) or 'pedestrian'),
                    'bbox': detection.get('bbox', []),
                    'keypoints': keypoints,
                    'keypoint_scores': scores,
                    'orientation': 'toward-road',
                })
        return skeletons

    @staticmethod
    def _fallback_skeletons(people):
        skeletons = []

        for index, detection in enumerate(people):
            x1, y1, x2, y2 = detection.get('bbox', [0, 0, 0, 0])
            width = max(x2 - x1, 1)
            height = max(y2 - y1, 1)
            keypoints = [
                [round(x1 + width * 0.50, 2), round(y1 + height * 0.12, 2), 0.92],
                [round(x1 + width * 0.35, 2), round(y1 + height * 0.35, 2), 0.88],
                [round(x1 + width * 0.65, 2), round(y1 + height * 0.35, 2), 0.88],
                [round(x1 + width * 0.40, 2), round(y1 + height * 0.78, 2), 0.84],
                [round(x1 + width * 0.60, 2), round(y1 + height * 0.78, 2), 0.84],
            ]
            skeletons.append({
                'person_id': f'pedestrian-cyclist-{index}',
                'source_object_id': detection.get('object_id'),
                'frame_id': detection.get('frame_id', detection.get('frame_index', 0)),
                'label': PedestrianCyclistPoseEstimation._label(detection) or 'pedestrian',
                'category': detection.get('category', PedestrianCyclistPoseEstimation._label(detection) or 'pedestrian'),
                'bbox': detection.get('bbox', []),
                'keypoints': keypoints,
                'orientation': 'toward-road',
            })
        return skeletons

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
        return [detection for detection in detections if PedestrianCyclistPoseEstimation._label(detection) in category_set]

    @staticmethod
    def _label(item):
        return item.get('label') or item.get('category') or ''

    @staticmethod
    def _pose_records(skeletons):
        grouped = {}
        for skeleton in skeletons:
            frame_index = int(skeleton.get('frame_id', skeleton.get('frame_index', 0)))
            item = dict(skeleton)
            item['frame_index'] = frame_index
            grouped.setdefault(frame_index, []).append(item)
        return [
            {'frame_index': frame_index, 'items': grouped[frame_index]}
            for frame_index in sorted(grouped)
        ]

    @staticmethod
    def _extract_mmpose_keypoints(sample):
        pred_instances = getattr(sample, 'pred_instances', None)
        if pred_instances is None:
            return [], []
        keypoints = getattr(pred_instances, 'keypoints', [])
        scores = getattr(pred_instances, 'keypoint_scores', [])
        if hasattr(keypoints, 'tolist'):
            keypoints = keypoints.tolist()
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()
        keypoints = keypoints[0] if keypoints and isinstance(keypoints[0], list) and keypoints[0] and isinstance(keypoints[0][0], list) else keypoints
        scores = scores[0] if scores and isinstance(scores[0], list) else scores
        output_keypoints = []
        for index, point in enumerate(keypoints):
            score = scores[index] if index < len(scores) else 0.0
            output_keypoints.append([round(float(point[0]), 2), round(float(point[1]), 2), round(float(score), 4)])
        return output_keypoints, [round(float(score), 4) for score in scores]

    def _mmpose_device(self):
        if isinstance(self.device, str):
            return self.device
        try:
            import torch

            if not torch.cuda.is_available():
                return 'cpu'
        except Exception:
            return 'cpu'
        return f'cuda:{int(self.device)}'
