import os

class VehicleTrajectoryPrediction:
    service_name = 'vehicle-trajectory-prediction'

    def __init__(self, weights='', device=0):
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
            'backend': 'constant-velocity',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch
            from torch import nn

            class TrajectoryGRU(nn.Module):
                def __init__(self, input_dim=8, hidden_dim=96, horizon=8):
                    super().__init__()
                    self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
                    self.head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, horizon * 2),
                    )
                    self.horizon = horizon

                def forward(self, x):
                    _, hidden = self.encoder(x)
                    return self.head(hidden[-1]).view(x.shape[0], self.horizon, 2)

            checkpoint = self._torch_load(torch, weight_path)
            architecture = checkpoint.get('architecture', {})
            network = TrajectoryGRU(
                input_dim=int(architecture.get('input_dim', 8)),
                hidden_dim=int(architecture.get('hidden_dim', 96)),
                horizon=int(architecture.get('horizon', 8)),
            )
            network.load_state_dict(checkpoint['model_state_dict'])
            device = self._torch_device(torch)
            network.to(device)
            network.eval()
            model.update({
                'loaded': True,
                'backend': 'pie-trajectory-gru',
                'model': network,
                'torch': torch,
                'device': device,
                'architecture': architecture,
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        inputs = payload.get('inputs')
        tracklets = self._all_items(inputs, 'track')
        attributes = self._all_items(inputs, 'attribute')
        road_context = self._road_context(self._all_items(inputs, 'segmentation'))
        attribute_by_object = {item.get('source_object_id'): item for item in attributes}
        predictions = self._predict_with_model(payload, tracklets, attribute_by_object, road_context)
        if predictions is None:
            predictions = self._fallback_predictions(tracklets, attribute_by_object, road_context)

        return {'trajectory': [{'frame_index': None, 'items': predictions}]}

    def _predict_with_model(self, payload, tracklets, attribute_by_object, road_context):
        if not (self.model and self.model.get('loaded')):
            return None
        torch = self.model['torch']
        architecture = self.model.get('architecture') or {}
        history_len = int(architecture.get('history_len', 8))
        width, height = self._frame_shape(payload, tracklets)
        predictions = []
        for tracklet in tracklets:
            features, last_center = self._track_features(tracklet, width, height, history_len)
            try:
                with torch.no_grad():
                    tensor = torch.tensor([features], dtype=torch.float32, device=self.model['device'])
                    offsets = self.model['model'](tensor)[0].detach().cpu().tolist()
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
                self.model['error'] = str(exc)
                return None
            attr = attribute_by_object.get(tracklet.get('source_object_id'), {})
            attributes = attr.get('attributes') or {}
            future_points = []
            for step, (offset_x, offset_y) in enumerate(offsets, start=1):
                future_points.append([
                    round((last_center[0] + float(offset_x)) * width, 2),
                    round((last_center[1] + float(offset_y)) * height, 2),
                    round(step * 0.5, 2),
                ])
            predictions.append({
                'track_id': tracklet.get('track_id'),
                'vehicle_type': attributes.get('type', tracklet.get('category', tracklet.get('label', 'vehicle'))),
                'future_trajectories': [{'prob': 0.78, 'points': future_points}],
                'abnormal_stop_prob': self._stop_probability(tracklet),
                'road_context_available': bool(road_context),
            })
        return predictions

    @staticmethod
    def _fallback_predictions(tracklets, attribute_by_object, road_context):
        predictions = []
        for tracklet in tracklets:
            bboxes = tracklet.get('bboxes') or [[0, 0, 0, 0]]
            last_box = bboxes[-1]
            center_x = (last_box[0] + last_box[2]) / 2
            center_y = (last_box[1] + last_box[3]) / 2
            speed = float(tracklet.get('speed_px_per_s', 0))
            attr = attribute_by_object.get(tracklet.get('source_object_id'), {})
            attributes = attr.get('attributes') or {}
            future_points = [
                [round(center_x + speed * step * 0.2, 2), round(center_y, 2), round(step * 0.5, 2)]
                for step in range(1, 5)
            ]
            predictions.append({
                'track_id': tracklet.get('track_id'),
                'vehicle_type': attributes.get('type', tracklet.get('category', tracklet.get('label', 'vehicle'))),
                'future_trajectories': [{'prob': 0.72, 'points': future_points}],
                'abnormal_stop_prob': 0.18 if speed < 1 else 0.05,
                'road_context_available': bool(road_context),
            })
        return predictions

    @staticmethod
    def _all_items(inputs, key):
        values = []
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
                            values.append(item)
        return values

    @staticmethod
    def _road_context(segmentation_items):
        return [
            item for item in segmentation_items
            if item.get('type') in {'drivable_area', 'lane_polyline', 'crosswalk_region', 'road_boundary'}
        ]

    @staticmethod
    def _track_features(tracklet, width, height, history_len):
        bboxes = tracklet.get('bboxes') or [[0, 0, 0, 0]]
        features = []
        previous = None
        for bbox in bboxes[-history_len:]:
            x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
            center_x = (x1 + x2) * 0.5 / width
            center_y = (y1 + y2) * 0.5 / height
            box_width = max(0.0, x2 - x1) / width
            box_height = max(0.0, y2 - y1) / height
            if previous is None:
                velocity_x = 0.0
                velocity_y = 0.0
            else:
                velocity_x = center_x - previous[0]
                velocity_y = center_y - previous[1]
            previous = [center_x, center_y]
            features.append([center_x, center_y, box_width, box_height, velocity_x, velocity_y, 0.0, 0.0])
        if not features:
            features = [[0.0] * 8]
        while len(features) < history_len:
            features.insert(0, list(features[0]))
        last_center = features[-1][:2]
        return features[-history_len:], last_center

    @staticmethod
    def _stop_probability(tracklet):
        speed = float(tracklet.get('speed_px_per_s', 0))
        if speed < 1:
            return 0.18
        if speed < 5:
            return 0.08
        return 0.03

    @staticmethod
    def _frame_shape(payload, tracklets):
        frames = payload.get('frames') or []
        if frames:
            height, width = frames[0].shape[:2]
            return max(float(width), 1.0), max(float(height), 1.0)
        max_x = 1.0
        max_y = 1.0
        for tracklet in tracklets:
            for bbox in tracklet.get('bboxes') or []:
                if len(bbox) == 4:
                    max_x = max(max_x, float(bbox[2]))
                    max_y = max(max_y, float(bbox[3]))
        return max_x, max_y

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
