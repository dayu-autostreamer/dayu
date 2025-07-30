import cv2
import numpy as np
from ultralytics import YOLO

class PedestrianDetectionYoloV8:
    def __init__(self, weights, device='cuda'):
        self.model = YOLO(weights)
        self.model.to(device)

        self.target_classes = {"person"}
        self.class_name_to_id = {
            name: idx for idx, name in self.model.names.items() if name in self.target_classes
        }
        self.target_class_ids = set(self.class_name_to_id.values())

    def infer(self, img: np.ndarray):

        results = self.model(img)[0]  

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        keep_idx = [i for i, cid in enumerate(class_ids) if cid in self.target_class_ids]

        result_boxes = np.round(boxes[keep_idx]).astype(int)   
        result_scores = scores[keep_idx]
        result_classid = class_ids[keep_idx]

        return result_boxes, result_scores, result_classid
