import os

class PedestrianIntentRecognition:
    service_name = 'pedestrian-intent-recognition'

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
            'backend': 'rule-intent',
            'error': '',
        }
        if not model['exists']:
            return model
        try:
            import torch
            from torch import nn

            class IntentGRU(nn.Module):
                def __init__(self, input_dim=12, hidden_dim=96, num_classes=3):
                    super().__init__()
                    self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
                    self.head = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, num_classes),
                    )

                def forward(self, x):
                    _, hidden = self.encoder(x)
                    return self.head(hidden[-1])

            checkpoint = self._torch_load(torch, weight_path)
            architecture = checkpoint.get('architecture', {})
            labels = architecture.get('labels') or ['not-crossing', 'crossing', 'crossing-irrelevant']
            network = IntentGRU(
                input_dim=int(architecture.get('input_dim', 12)),
                hidden_dim=int(architecture.get('hidden_dim', 96)),
                num_classes=len(labels),
            )
            network.load_state_dict(checkpoint['model_state_dict'])
            device = self._torch_device(torch)
            network.to(device)
            network.eval()
            model.update({
                'loaded': True,
                'backend': 'pie-intent-gru',
                'model': network,
                'torch': torch,
                'device': device,
                'architecture': architecture,
                'labels': list(labels),
            })
        except Exception as exc:  # pragma: no cover - depends on optional runtime packages.
            model['error'] = str(exc)
        return model

    def __call__(self, payload):
        inputs = payload.get('inputs')
        skeletons = self._all_items(inputs, 'pose')
        crosswalk_regions = self._crosswalk_regions(self._all_items(inputs, 'segmentation'))
        intents = self._infer_with_model(payload, skeletons, crosswalk_regions)
        if intents is None:
            intents = self._fallback_intents(skeletons, crosswalk_regions)

        return {'text': [{'frame_index': None, 'items': intents}]}

    def _infer_with_model(self, payload, skeletons, crosswalk_regions):
        if not (self.model and self.model.get('loaded')):
            return None
        if not skeletons:
            return []
        torch = self.model['torch']
        architecture = self.model.get('architecture') or {}
        history_len = int(architecture.get('history_len', 8))
        width, height = self._frame_shape(payload, skeletons)
        intents = []
        for person_id, history in self._group_skeletons(skeletons).items():
            features = self._history_features(history, width, height, history_len)
            try:
                with torch.no_grad():
                    tensor = torch.tensor([features], dtype=torch.float32, device=self.model['device'])
                    logits = self.model['model'](tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    score, index = torch.max(probs, dim=0)
            except Exception as exc:  # pragma: no cover - hardware/runtime dependent.
                self.model['error'] = str(exc)
                return None
            skeleton = history[-1]
            label = self.model.get('labels', [])[int(index.item())]
            category = skeleton.get('category', skeleton.get('label', 'pedestrian'))
            action, intent = self._label_to_intent(label, category)
            intents.append({
                'person_id': person_id,
                'label': 'pedestrian_cyclist_intent',
                'category': category,
                'action': action,
                'intent': intent,
                'text': intent,
                'confidence': round(float(score.item()), 4),
                'time_window': [0.0, 2.0],
                'road_context_available': bool(crosswalk_regions),
                'model_label': label,
            })
        return intents

    @staticmethod
    def _fallback_intents(skeletons, crosswalk_regions):
        intents = []
        for skeleton in skeletons:
            category = skeleton.get('category', skeleton.get('label', 'pedestrian'))
            action = 'crossing' if category == 'pedestrian' else 'riding'
            confidence = 0.82 if crosswalk_regions else 0.68
            intents.append({
                'person_id': skeleton.get('person_id'),
                'label': 'pedestrian_cyclist_intent',
                'category': category,
                'action': action,
                'intent': 'likely_to_cross' if action == 'crossing' else 'likely_to_enter_lane',
                'text': 'likely_to_cross' if action == 'crossing' else 'likely_to_enter_lane',
                'confidence': confidence,
                'time_window': [0.0, 2.0],
                'road_context_available': bool(crosswalk_regions),
            })
        return intents

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
    def _crosswalk_regions(segmentation_items):
        return [
            item.get('polygon')
            for item in segmentation_items
            if item.get('type') == 'crosswalk_region' and item.get('polygon')
        ]

    @staticmethod
    def _group_skeletons(skeletons):
        grouped = {}
        for skeleton in sorted(skeletons, key=lambda item: (item.get('person_id') or item.get('source_object_id') or '',
                                                           item.get('frame_id', 0))):
            key = skeleton.get('person_id') or skeleton.get('source_object_id') or f"person-{len(grouped)}"
            grouped.setdefault(key, []).append(skeleton)
        return grouped

    @staticmethod
    def _history_features(history, width, height, history_len):
        features = []
        previous = None
        for skeleton in history[-history_len:]:
            bbox = PedestrianIntentRecognition._bbox_from_skeleton(skeleton)
            x1, y1, x2, y2 = [float(value) for value in bbox]
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
            moving = abs(velocity_x) + abs(velocity_y) > 0.002
            looking = skeleton.get('orientation') == 'toward-road'
            features.append([
                center_x, center_y, box_width, box_height, velocity_x, velocity_y, 0.0, 0.0,
                0.0 if moving else 1.0,
                1.0 if moving else 0.0,
                1.0 if looking else 0.0,
                0.0 if looking else 1.0,
            ])
        if not features:
            features = [[0.0] * 12]
        while len(features) < history_len:
            features.insert(0, list(features[0]))
        return features[-history_len:]

    @staticmethod
    def _bbox_from_skeleton(skeleton):
        bbox = skeleton.get('bbox') or []
        if len(bbox) == 4:
            return bbox
        keypoints = skeleton.get('keypoints') or []
        points = [point for point in keypoints if len(point) >= 2 and (len(point) < 3 or point[2] > 0)]
        if not points:
            return [0, 0, 0, 0]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return [min(xs), min(ys), max(xs), max(ys)]

    @staticmethod
    def _frame_shape(payload, skeletons):
        frames = payload.get('frames') or []
        if frames:
            height, width = frames[0].shape[:2]
            return max(float(width), 1.0), max(float(height), 1.0)
        max_x = 1.0
        max_y = 1.0
        for skeleton in skeletons:
            bbox = PedestrianIntentRecognition._bbox_from_skeleton(skeleton)
            max_x = max(max_x, float(bbox[2]))
            max_y = max(max_y, float(bbox[3]))
        return max_x, max_y

    @staticmethod
    def _label_to_intent(label, category):
        if label == 'crossing':
            return 'crossing', 'likely_to_cross' if category == 'pedestrian' else 'likely_to_enter_lane'
        if label == 'crossing-irrelevant':
            return 'crossing-irrelevant', 'not_relevant_to_lane'
        return 'not-crossing', 'not_likely_to_cross'

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
