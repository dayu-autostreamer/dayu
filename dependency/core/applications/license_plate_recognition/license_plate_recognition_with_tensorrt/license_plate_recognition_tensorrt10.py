import cv2
import numpy as np
from typing import List, Tuple

import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from core.lib.common import LOGGER


class LicensePlateRecognitionTensorRT10:
    def __init__(self, detect_weights: str, recognize_weights: str, device=0):
        self.detect_weights = detect_weights
        self.recognize_weights = recognize_weights

        self.ctx = cuda.Device(device).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        LOGGER.info("Loading detection engine...")
        self.detect_engine = self._load_engine(self.detect_weights, runtime)
        self.detect_context = self.detect_engine.create_execution_context()
        self.detect_bindings = self._allocate_buffers(self.detect_engine)

        # Get input/output tensor names and shapes for TensorRT 10
        detect_input_name = self.detect_bindings['input_binding_names'][0]
        detect_output_name = self.detect_bindings['output_binding_names'][0]
        self.detect_input_shape = self.detect_engine.get_tensor_shape(detect_input_name)
        self.detect_output_shape = self.detect_engine.get_tensor_shape(detect_output_name)
        LOGGER.info(f"Detection engine - Input: {self.detect_input_shape}, Output: {self.detect_output_shape}")

        LOGGER.info("Loading recognition engine...")
        self.recognize_engine = self._load_engine(self.recognize_weights, runtime)
        self.recognize_context = self.recognize_engine.create_execution_context()
        self.recognize_bindings = self._allocate_buffers(self.recognize_engine)

        # Get input/output tensor names and shapes for TensorRT 10
        recognize_input_name = self.recognize_bindings['input_binding_names'][0]
        recognize_output_names = self.recognize_bindings['output_binding_names']
        self.recognize_input_shape = self.recognize_engine.get_tensor_shape(recognize_input_name)
        self.recognize_output_shape = [self.recognize_engine.get_tensor_shape(name) for name in recognize_output_names]
        LOGGER.info(f"Recognition engine - Input: {self.recognize_input_shape}, Outputs: {self.recognize_output_shape}")

        self.characters = '#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品'

        self.plate_colors = ['黑色', '蓝色', '绿色', '白色', '黄色']

        self.warm_up_turns = 5
        self.warm_up()

    def __del__(self):
        self.destroy()

    def destroy(self):
        self.ctx.pop()
        LOGGER.info("LicensePlateRecognition resources released")

    def _load_engine(self, engine_path: str, runtime: trt.Runtime):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        return runtime.deserialize_cuda_engine(engine_data)

    def _allocate_buffers(self, engine: trt.ICudaEngine):
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        input_binding_names = []
        output_binding_names = []

        for binding_name in engine:
            shape = engine.get_tensor_shape(binding_name)
            size = trt.volume(shape)
            dtype = trt.nptype(engine.get_tensor_dtype(binding_name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                input_binding_names.append(binding_name)
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                LOGGER.info(f"Input tensor: name='{binding_name}', dtype={dtype}, shape={shape}")
            elif engine.get_tensor_mode(binding_name) == trt.TensorIOMode.OUTPUT:
                output_binding_names.append(binding_name)
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                LOGGER.info(f"Output tensor: name='{binding_name}', dtype={dtype}, shape={shape}")
            else:
                # Unknown tensor mode, skip it
                pass

        return {
            'host_inputs': host_inputs,
            'cuda_inputs': cuda_inputs,
            'host_outputs': host_outputs,
            'cuda_outputs': cuda_outputs,
            'input_binding_names': input_binding_names,
            'output_binding_names': output_binding_names
        }

    def _preprocess(self, image: np.ndarray, target_size: Tuple[int, int], normalize=True):
        img_resized = cv2.resize(image, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        if normalize:
            img_norm = img_rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_norm - mean) / std
        else:
            img_norm = img_rgb.astype(np.float32)

        img_chw = np.transpose(img_norm, [2, 0, 1])
        return np.ascontiguousarray(img_chw)

    def _infer(self, context: trt.IExecutionContext, bindings: dict, input_data: np.ndarray):

        self.ctx.push()

        host_inputs = bindings['host_inputs']
        cuda_inputs = bindings['cuda_inputs']
        host_outputs = bindings['host_outputs']
        cuda_outputs = bindings['cuda_outputs']
        input_binding_names = bindings['input_binding_names']
        output_binding_names = bindings['output_binding_names']

        np.copyto(host_inputs[0], input_data.ravel())

        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], self.stream)

        # Set tensor addresses for TensorRT 10
        context.set_tensor_address(input_binding_names[0], cuda_inputs[0])
        for i, output_name in enumerate(output_binding_names):
            context.set_tensor_address(output_name, cuda_outputs[i])

        context.execute_async_v3(stream_handle=self.stream.handle)

        outputs = []
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], self.stream)

        self.stream.synchronize()
        self.ctx.pop()

        return [output.copy() for output in host_outputs]

    def detect_plates(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        _, _, input_h, input_w = self.detect_input_shape

        input_image = self._preprocess(image, (input_w, input_h))

        outputs = self._infer(self.detect_context, self.detect_bindings, input_image)

        detection_output = outputs[0].reshape(self.detect_output_shape)

        plate_boxes = self._parse_detection_output(detection_output, image.shape)

        plates = []
        for box in plate_boxes:
            x1, y1, x2, y2, conf = box
            plate_img = image[int(y1):int(y2), int(x1):int(x2)]
            if plate_img.size > 0:
                plates.append((plate_img, (x1, y1, x2, y2)))

        return plates

    def _parse_detection_output(self, detection_output: np.ndarray, img_shape: tuple, conf_threshold=0.5):
        h, w = img_shape[:2]
        boxes = []

        num_boxes = detection_output.shape[1]

        for i in range(num_boxes):
            box_data = detection_output[0, i]

            x_center, y_center, box_w, box_h, conf = box_data[:5]

            if conf < conf_threshold:
                continue

            x1 = (x_center - box_w / 2) * w
            y1 = (y_center - box_h / 2) * h
            x2 = (x_center + box_w / 2) * w
            y2 = (y_center + box_h / 2) * h

            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))

            boxes.append((x1, y1, x2, y2, conf))

        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        filtered_boxes = []

        while boxes:
            current = boxes.pop(0)
            filtered_boxes.append(current)
            boxes = [box for box in boxes if self._iou(current, box) < 0.5]

        return filtered_boxes

    def _iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1, _ = box1
        x1_2, y1_2, x2_2, y2_2, _ = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def recognize_plate(self, plate_image: np.ndarray) -> Tuple[str, str]:

        _, _, input_h, input_w = self.recognize_input_shape

        input_image = self._preprocess(plate_image, (input_w, input_h))

        outputs = self._infer(self.recognize_context, self.recognize_bindings, input_image)

        color_probs = outputs[0].flatten()
        char_probs = outputs[1]

        plate_text = self._decode_text_output(char_probs)

        plate_color = self._decode_color_output(color_probs)

        return plate_text, plate_color

    def _decode_text_output(self, char_probs: np.ndarray) -> str:

        char_probs = char_probs[0]

        char_indices = np.argmax(char_probs, axis=1)

        plate_str = []
        prev_char = None
        for idx in char_indices:
            if idx == 0:
                prev_char = None
                continue

            char = self.characters[idx]

            if char != prev_char:
                plate_str.append(char)
                prev_char = char

        return ''.join(plate_str)

    def _decode_color_output(self, color_probs: np.ndarray) -> str:
        color_idx = np.argmax(color_probs)
        return self.plate_colors[color_idx] if color_idx < len(self.plate_colors) else "unknown"

    def warm_up(self):
        LOGGER.info('Warming up engines...')
        _, _, detect_h, detect_w = self.detect_input_shape
        dummy_detect = np.zeros((detect_h, detect_w, 3), dtype=np.uint8)

        _, _, rec_h, rec_w = self.recognize_input_shape
        dummy_plate = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)

        for _ in range(self.warm_up_turns):
            plates = self.detect_plates(dummy_detect)
            for plate_img, _ in plates:
                self.recognize_plate(plate_img)
        LOGGER.info('Warm up completed')

    def infer(self, image: np.ndarray) -> List[Tuple[str, str]]:
        plates = self.detect_plates(image)

        results = []
        for plate_img, bbox in plates:
            if plate_img.size == 0:
                LOGGER.warning("Skipping empty plate image")
                continue

            try:
                plate_text, plate_color = self.recognize_plate(plate_img)
                results.append((plate_text, plate_color))
                LOGGER.info(f"Detected plate: {plate_text} ({plate_color}) at {bbox}")
            except Exception as e:
                LOGGER.error(f"Plate recognition error: {str(e)}")
                results.append(("", ""))

        return results
