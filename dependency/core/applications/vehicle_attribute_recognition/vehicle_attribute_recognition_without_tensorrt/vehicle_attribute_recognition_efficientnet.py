import json
import os

class VehicleAttributeRecognition:
    service_name = 'vehicle-attribute-recognition'
    default_model_name = 'vehicle-attribute-classifier'

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
            'backend': 'rule-attribute',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch
            from torch import nn
            from torchvision import models

            checkpoint = self._torch_load(torch, weight_path)
            classes = checkpoint.get('classes') or checkpoint.get('attribute_heads', {}).get('vehicle_type') or []
            network = models.efficientnet_b0(weights=None)
            in_features = network.classifier[1].in_features
            network.classifier[1] = nn.Linear(in_features, len(classes))
            network.load_state_dict(checkpoint['model_state'])
            device = self._torch_device(torch)
            network.to(device)
            network.eval()
            model.update({
                'loaded': True,
                'backend': 'efficientnet-b0',
                'model': network,
                'torch': torch,
                'device': device,
                'classes': list(classes),
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        detections = self._first_output(payload.get('inputs'), 'detections', default=[])
        vehicles = self._filter_detections(detections, {'car', 'bus', 'truck', 'motorcycle'})
        attributes = []

        seen_objects = []
        for detection in vehicles:
            object_id = detection.get('object_id')
            if object_id in seen_objects:
                continue
            seen_objects.append(object_id)
            index = len(attributes)
            vehicle_type, confidence = self._infer_vehicle_type(payload, detection)
            if not vehicle_type:
                vehicle_type = detection.get('category', 'vehicle')
                confidence = round(float(detection.get('score', 0.8)), 3)
            crop = self._crop_for_detection(payload, detection)
            attributes.append({
                'object_id': object_id,
                'type': vehicle_type,
                'color': self._estimate_color(crop, index),
                'orientation': self._estimate_orientation(detection, index),
                'confidence': round(float(confidence), 4),
                'source_category': detection.get('category', 'vehicle'),
            })

        outputs = {'vehicle_attributes': attributes}
        return self._wrap_result(payload, outputs, num_objects=len(attributes),
                                 inference_backend=self._model_backend())

    def _infer_vehicle_type(self, payload, detection):
        if not (self.model and self.model.get('loaded')):
            return None, 0.0
        crop = self._crop_for_detection(payload, detection)
        tensor = self._preprocess_crop(crop)
        if tensor is None:
            return None, 0.0
        torch = self.model['torch']
        network = self.model['model']
        try:
            with torch.no_grad():
                logits = network(tensor.to(self.model['device']))
                probs = torch.softmax(logits, dim=1)[0]
                score, index = torch.max(probs, dim=0)
            classes = self.model.get('classes') or []
            if not classes:
                return None, 0.0
            return classes[int(index.item())], float(score.item())
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
            self.model['error'] = str(exc)
            return None, 0.0

    def _wrap_result(self, payload, outputs, num_objects=0, inference_backend='rule-attribute'):
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

    def _crop_for_detection(self, payload, detection):
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

    def _preprocess_crop(self, crop):
        if crop is None:
            return None
        try:
            import cv2
            import numpy as np

            image = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            image = image[:, :, :3][:, :, ::-1].astype('float32') / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype='float32')
            std = np.array([0.229, 0.224, 0.225], dtype='float32')
            image = (image - mean) / std
            tensor = self.model['torch'].from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            return tensor
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            if self.model is not None:
                self.model['error'] = str(exc)
            return None

    @staticmethod
    def _estimate_color(crop, index):
        if crop is None:
            return ['white', 'black', 'silver', 'blue'][index % 4]
        try:
            mean_bgr = crop.reshape(-1, crop.shape[-1]).mean(axis=0)
            blue, green, red = [float(value) for value in mean_bgr[:3]]
        except Exception:
            return ['white', 'black', 'silver', 'blue'][index % 4]
        brightness = (red + green + blue) / 3.0
        if brightness < 55:
            return 'black'
        if brightness > 205:
            return 'white'
        if abs(red - green) < 18 and abs(green - blue) < 18:
            return 'silver'
        if red >= green and red >= blue:
            return 'red'
        if blue >= red and blue >= green:
            return 'blue'
        return 'green'

    @staticmethod
    def _estimate_orientation(detection, index):
        bbox = detection.get('bbox') or [0, 0, 0, 0]
        width = max(float(bbox[2] - bbox[0]), 1.0) if len(bbox) == 4 else 1.0
        height = max(float(bbox[3] - bbox[1]), 1.0) if len(bbox) == 4 else 1.0
        if width / height > 1.6:
            return 'side'
        return 'front-left' if index % 2 == 0 else 'rear-right'

    def _model_backend(self):
        return (self.model or {}).get('backend', 'rule-attribute')

    def _torch_device(self, torch):
        if isinstance(self.device, str):
            return torch.device(self.device if self.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            return torch.device(f'cuda:{int(self.device)}')
        return torch.device('cpu')

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
