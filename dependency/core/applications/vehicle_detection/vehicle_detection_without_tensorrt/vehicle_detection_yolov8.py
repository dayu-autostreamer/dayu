import cv2
import numpy as np
from ultralytics import YOLO

class VehicleDetectionYoloV8:
    def __init__(self, weights, device='cuda'):
        self.model = YOLO(weights)
        self.model.to(device)

        self.target_classes = {"car", "bus", "truck", "motorcycle", "bicycle"}
        self.class_name_to_id = {
            name: idx for idx, name in self.model.names.items() if name in self.target_classes
        }
        self.target_class_ids = set(self.class_name_to_id.values())

    def infer(self, img: np.ndarray):
        """
        对图像进行推理，输出车辆相关的检测框、置信度、类别ID
        :param img: 输入图像，格式为 OpenCV 的 BGR ndarray
        :return: result_boxes, result_scores, result_classid
        """
        results = self.model(img)[0]  # 取第一个batch结果

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        keep_idx = [i for i, cid in enumerate(class_ids) if cid in self.target_class_ids]

        result_boxes = np.round(boxes[keep_idx]).astype(int)   # ⭐ 转为整数
        result_scores = scores[keep_idx]
        result_classid = class_ids[keep_idx]

        return result_boxes, result_scores, result_classid
