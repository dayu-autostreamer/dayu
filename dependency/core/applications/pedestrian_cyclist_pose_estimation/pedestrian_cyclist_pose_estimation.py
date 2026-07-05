import json
import os

from core.lib.common import Context


class PedestrianCyclistPoseEstimation:
    service_name = 'pedestrian-cyclist-pose-estimation'
    default_model_name = 'pedestrian-cyclist-pose-estimator'

    def __init__(self, model_name=None, model_weight='', model_variant='prototype',
                 device=0, synthetic_complexity=1):
        self.model_name = model_name or self.default_model_name
        self.model_weight = Context.get_file_path(model_weight) if model_weight else ''
        self.model_variant = model_variant
        self.device = device
        self.synthetic_complexity = max(1, int(synthetic_complexity))
        self.flops = 0

    def __call__(self, payload):
        detections = self._first_output(payload.get('inputs'), 'detections', default=[])
        people = self._filter_detections(detections, {'pedestrian', 'cyclist'})
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
                'frame_id': detection.get('frame_id', 0),
                'category': detection.get('category', 'pedestrian'),
                'bbox': detection.get('bbox', []),
                'keypoints': keypoints,
                'orientation': 'toward-road',
            })

        outputs = {'skeletons': skeletons}
        return self._wrap_result(payload, outputs, num_objects=len(skeletons))

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
