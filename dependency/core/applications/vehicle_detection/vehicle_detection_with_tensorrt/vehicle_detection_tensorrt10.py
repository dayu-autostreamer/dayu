import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import cv2
from typing import List

from core.lib.common import Context

__all__ = ('VehicleDetectionTensorRT',)

class VehicleDetectionTensorRT10:
    def __init__(self, weights, plugin_library, device=0):

        self.weights = weights
        self.plugin_library = plugin_library
        ctypes.CDLL(self.plugin_library)

        self.ctx = cuda.Device(device).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(self.weights, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.input_binding_names = []
        self.output_binding_names = []

        for binding_name in self.engine:
            shape = self.engine.get_tensor_shape(binding_name)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                self.input_binding_names.append(binding_name)
                self.input_w = shape[-1]
                self.input_h = shape[-2]
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            elif self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                self.output_binding_names.append(binding_name)
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

        self.batch_size = self.engine.get_tensor_shape(self.input_binding_names[0])[0]
        self.det_output_length = self.host_outputs[0].shape[0]

        self.conf_thres = 0.5
        self.iou_thres = 0.4

        self.categories = np.array(["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light",
                  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
                  "frisbee",
                  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                  "surfboard",
                  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                  "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear",
                  "hair drier", "toothbrush"])  # 同原始COCO categories
        self.target_categories = np.asarray(["car", "bus", "motorcycle", "bicycle", "truck"])
        self.class_id = 'vehicle'

        self.DET_NUM = 6
        self.SEG_NUM = 32
        self.POSE_NUM = 17 * 3
        self.num_values_per_detection = self.DET_NUM + self.SEG_NUM + self.POSE_NUM

        self._warm_up()

    def _warm_up(self, turns: int = 5):
        for _ in range(turns):
            image = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)
            self.infer(image)

    def __del__(self):
        self.destroy()

    def destroy(self):
        self.ctx.pop()

    def infer(self, raw_image):
        self.ctx.push()

        batch_input_image = np.empty([self.batch_size, 3, self.input_h, self.input_w])
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(raw_image)
        np.copyto(batch_input_image[0], input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)
        np.copyto(self.host_inputs[0], batch_input_image.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.set_tensor_address(self.input_binding_names[0], self.cuda_inputs[0])
        self.context.set_tensor_address(self.output_binding_names[0], self.cuda_outputs[0])
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        self.ctx.pop()

        output = self.host_outputs[0]
        result_boxes, result_scores, result_classid = self.post_process(output, origin_h, origin_w)

        # Filter vehicle classes
        mask = np.isin(self.categories[result_classid.astype(int)], self.target_categories)

        result_boxes = result_boxes.astype(int)[mask].tolist()
        result_scores = result_scores[mask].tolist()
        result_classid = np.full(len(result_boxes), self.class_id).tolist()

        return result_boxes, result_scores, result_classid

    def preprocess_image(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, _ = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return np.ascontiguousarray(image), image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0]
            y[:, 2] = x[:, 2]
            y[:, 1] = x[:, 1] - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 3] - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 2] - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1]
            y[:, 3] = x[:, 3]
            y /= r_h
        return y

    def post_process(self, output, origin_h, origin_w):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, self.num_values_per_detection))[:num, :]
        boxes = self.non_max_suppression(pred, origin_h, origin_w, self.conf_thres, self.iou_thres)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        return inter_area / (b1_area + b2_area - inter_area + 1e-16)

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            invalid = large_overlap & label_match
            keep_boxes.append(boxes[0])
            boxes = boxes[~invalid]
        return np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])

