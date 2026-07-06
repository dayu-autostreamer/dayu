import json
import os

class RoadContextSegmentation:
    service_name = 'road-context-segmentation'
    default_model_name = 'road-context-segmenter'

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
            'backend': 'opencv-road-context',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch

            checkpoint = RoadContextSegmentation._torch_load(torch, weight_path)
            model['checkpoint_loaded'] = True
            if isinstance(checkpoint, dict):
                model['checkpoint_keys'] = sorted(str(key) for key in checkpoint.keys())[:8]
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        outputs = self._infer_context(payload)
        return self._wrap_result(payload, outputs,
                                 num_objects=len(outputs.get('lane_polylines', [])) +
                                             len(outputs.get('crosswalk_regions', [])),
                                 inference_backend=self._model_backend())

    def _infer_context(self, payload):
        frames = payload.get('frames') or []
        if not frames:
            return self._fallback_context(payload)
        frame = frames[0]
        try:
            import cv2
            import numpy as np
        except Exception as exc:  # pragma: no cover - base image provides opencv/numpy.
            if self.model is not None:
                self.model['error'] = str(exc)
            return self._fallback_context(payload)

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        lower_mask = np.zeros_like(gray)
        lower_mask[int(height * 0.35):, :] = gray[int(height * 0.35):, :]
        edges = cv2.Canny(lower_mask, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45,
                                minLineLength=max(24, width // 18), maxLineGap=max(12, width // 60))
        lane_polylines = self._lane_polylines_from_lines(lines, width, height)
        if not lane_polylines:
            lane_polylines = self._fallback_context(payload)['lane_polylines']

        drivable_area = [
            [0, height],
            [int(width * 0.40), int(height * 0.38)],
            [int(width * 0.60), int(height * 0.38)],
            [width, height],
        ]
        crosswalk_regions = self._detect_crosswalk_regions(gray, width, height)
        outputs = {
            'lane_polylines': lane_polylines,
            'drivable_area': drivable_area,
            'crosswalk_regions': crosswalk_regions,
            'road_boundary': [[0, height], [width, height]],
        }
        return outputs

    def _fallback_context(self, payload):
        width, height = self._frame_shape(payload)
        width = max(width, 640)
        height = max(height, 360)

        lane_polylines = [
            [[int(width * 0.34), height], [int(width * 0.42), int(height * 0.58)], [int(width * 0.48), int(height * 0.35)]],
            [[int(width * 0.66), height], [int(width * 0.58), int(height * 0.58)], [int(width * 0.52), int(height * 0.35)]],
        ]
        drivable_area = [
            [0, height],
            [int(width * 0.40), int(height * 0.35)],
            [int(width * 0.60), int(height * 0.35)],
            [width, height],
        ]
        crosswalk_regions = [
            [
                [int(width * 0.20), int(height * 0.62)],
                [int(width * 0.80), int(height * 0.62)],
                [int(width * 0.86), int(height * 0.74)],
                [int(width * 0.14), int(height * 0.74)],
            ]
        ]

        outputs = {
            'lane_polylines': lane_polylines,
            'drivable_area': drivable_area,
            'crosswalk_regions': crosswalk_regions,
            'road_boundary': [[0, height], [width, height]],
        }
        return outputs

    @staticmethod
    def _lane_polylines_from_lines(lines, width, height):
        if lines is None:
            return []
        left_lines = []
        right_lines = []
        for line in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [int(value) for value in line]
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.35:
                continue
            target = left_lines if slope < 0 else right_lines
            target.append((x1, y1, x2, y2))

        def average_line(candidates):
            if not candidates:
                return None
            xs = []
            ys = []
            for x1, y1, x2, y2 in candidates:
                xs.extend([x1, x2])
                ys.extend([y1, y2])
            top = min(ys)
            bottom = height
            if max(ys) == min(ys):
                center_x = int(sum(xs) / max(len(xs), 1))
                return [[center_x, bottom], [center_x, top]]
            slope, intercept = RoadContextSegmentation._fit_line(xs, ys)
            if abs(slope) < 1e-6:
                return None
            x_bottom = int((bottom - intercept) / slope)
            x_top = int((top - intercept) / slope)
            x_bottom = max(0, min(width, x_bottom))
            x_top = max(0, min(width, x_top))
            return [[x_bottom, bottom], [x_top, int(top)]]

        polylines = []
        for candidates in (left_lines, right_lines):
            line = average_line(candidates)
            if line:
                polylines.append(line)
        return polylines

    @staticmethod
    def _fit_line(xs, ys):
        count = float(len(xs))
        mean_x = sum(xs) / count
        mean_y = sum(ys) / count
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        denominator = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        return slope, intercept

    @staticmethod
    def _detect_crosswalk_regions(gray, width, height):
        try:
            import cv2
        except Exception:  # pragma: no cover - base image provides opencv.
            return []
        region = gray[int(height * 0.45):int(height * 0.85), :]
        if region.size == 0:
            return []
        _, binary = cv2.threshold(region, 210, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        y_offset = int(height * 0.45)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > width * 0.12 and h < height * 0.08:
                boxes.append([x, y + y_offset, x + w, y + y_offset + h])
        if len(boxes) < 2:
            return []
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)
        return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]

    def _wrap_result(self, payload, outputs, num_objects=0, inference_backend='opencv-road-context'):
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
    def _frame_shape(payload):
        frames = payload.get('frames') or []
        if not frames:
            return 0, 0
        height, width = frames[0].shape[:2]
        return int(width), int(height)

    def _model_backend(self):
        return (self.model or {}).get('backend', 'opencv-road-context')

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
