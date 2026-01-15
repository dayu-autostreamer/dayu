import copy
import cv2
import torch
import numpy as np
import sys
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from plate_recognition.plate_rec import get_plate_result, init_model
from plate_recognition.double_plate_split_merge import get_split_merge


class LicensePlateRecognitionPT:
    def __init__(self, detect_model_path, rec_model_path, device=0, img_size=640, is_color=True):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.is_color = is_color

        # Load models
        self.detect_model = attempt_load(detect_model_path, map_location=self.device)
        self.plate_rec_model = init_model(self.device, rec_model_path, is_color=is_color)

    def _scale_landmarks(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

        coords[:, [0, 2, 4, 6]] -= pad[0]
        coords[:, [1, 3, 5, 7]] -= pad[1]
        coords[:, :8] /= gain
        coords[:, 0::2].clamp_(0, img0_shape[1])
        coords[:, 1::2].clamp_(0, img0_shape[0])
        return coords

    def _warp_plate(self, image, points):
        pts = points.astype('float32')
        (tl, tr, br, bl) = pts
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxWidth = int(max(widthA, widthB))
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype='float32')
        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _extract_plate_text(self, image, xyxy, landmarks, class_num):
        points = np.array([[landmarks[i], landmarks[i + 1]] for i in range(0, 8, 2)], dtype=np.float32)
        plate_img = self._warp_plate(image, points)

        if int(class_num) == 1:
            plate_img = get_split_merge(plate_img)

        if self.is_color:
            plate_no, _, plate_color, _ = get_plate_result(plate_img, self.device, self.plate_rec_model, is_color=True)
            text = f"{plate_no} {plate_color} {'双层' if class_num else ''}"
        else:
            plate_no, _ = get_plate_result(plate_img, self.device, self.plate_rec_model, is_color=False)
            text = f"{plate_no} {'双层' if class_num else ''}"
        return text.strip()

    def infer(self, image: np.ndarray):
        """
        对单张图像进行车牌识别推理，返回识别结果列表
        :param image: BGR格式的图像
        :return: List[str]，每张图像中检测到的所有车牌文本
        """
        img0 = copy.deepcopy(image)
        h0, w0 = img0.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.img_size, s=self.detect_model.stride.max())
        img = letterbox(img0, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR → RGB, HWC → CHW
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.detect_model(img)[0]
            pred = non_max_suppression_face(pred, conf_thres=0.3, iou_thres=0.5)

        results = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                det[:, 5:13] = self._scale_landmarks(img.shape[2:], det[:, 5:13], img0.shape).round()

                for d in det:
                    xyxy = d[:4].tolist()
                    landmarks = d[5:13].tolist()
                    class_num = int(d[13].item())
                    text = self._extract_plate_text(img0, xyxy, landmarks, class_num)
                    results.append(text)

        return results
