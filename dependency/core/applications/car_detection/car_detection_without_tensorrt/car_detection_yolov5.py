# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse

import os

import sys
from pathlib import Path

import torch

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, print_args, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def parse_opt(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='', help='model path or triton URL')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    opt = parser.parse_args(args)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


class CarDetectionYoloV5:

    def __init__(self, **args):
        # write code to change the args dict to command line args
        args_list = []
        for k, v in args.items():
            args[k] = '--' + k
            args_list.append(args[k])
            args_list.append(str(v))
        args = args_list

        self.target_categories = np.asarray(["car", "bus", "truck"])
        self.class_id = 'car'

        self.opt = parse_opt(args)

        self.data = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
            29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
            48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
            62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
            69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
        }

        # Load model
        self.device = select_device(self.opt.device)
        self.model = DetectMultiBackend(weights=self.opt.weights, device=self.device, dnn=self.opt.dnn,
                                        data=self.data, fp16=self.opt.half)
        self.model.eval()
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.opt.imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1, 3, *self.imgsz))  # warmup

    def infer(self, image):
        model = self.model
        conf_thres = self.opt.conf_thres
        iou_thres = self.opt.iou_thres
        classes = self.opt.classes
        agnostic_nms = self.opt.agnostic_nms
        max_det = self.opt.max_det

        # Run inference
        im0 = image
        im = im0.copy()
        im = letterbox(im, (im.shape[:2][0], im.shape[:2][1]), stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1].copy()  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # with dt[0]:
        im = torch.from_numpy(im).to(model.device)

        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        # with dt[1]:
        pred = model(im)

        # NMS
        # with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            annotator = Annotator(im0, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # print(f"{n} {self.names[int(c)]}{'s' * (n > 1)}, ")  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

        # result: [x1,y1,x2,y2,prob,cls]
        results = np.array(det.tolist())
        result_boxes, result_scores, result_classid = results[:, :4], results[:, 4], results[:, 5]

        mask = np.isin(self.names[result_classid.astype(int)], self.target_categories)

        result_boxes = result_boxes.astype(int)[mask].tolist()
        result_scores = result_scores[mask].tolist()
        result_classid = np.full(len(result_boxes), self.class_id).tolist()

        return result_boxes, result_scores, result_classid
