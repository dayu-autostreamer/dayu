import cv2 
import numpy as np 
from typing import List, Tuple, Union

import pycuda.autoinit 
import pycuda.driver as cuda 
import tensorrt as trt 

from core.lib.common import Context, LOGGER 

class LicensePlateRecognitionTensorRT:
    def __init__(self, detect_weights: str, recognize_weights: str, device=0):
        self.detect_weights = detect_weights
        self.recognize_weights = recognize_weights
        
        self.ctx = cuda.Device(device).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        
        # 初始化车牌检测引擎
        LOGGER.info("Loading detection engine...")
        self.detect_engine = self._load_engine(self.detect_weights, runtime)
        self.detect_context = self.detect_engine.create_execution_context()
        self.detect_bindings = self._allocate_buffers(self.detect_engine)
        
        # 获取检测引擎输入输出信息
        self.detect_input_shape = self.detect_engine.get_binding_shape(0)
        self.detect_output_shape = self.detect_engine.get_binding_shape(1)
        LOGGER.info(f"Detection engine - Input: {self.detect_input_shape}, Output: {self.detect_output_shape}")
        
        # 初始化车牌识别引擎
        LOGGER.info("Loading recognition engine...")
        self.recognize_engine = self._load_engine(self.recognize_weights, runtime)
        self.recognize_context = self.recognize_engine.create_execution_context()
        self.recognize_bindings = self._allocate_buffers(self.recognize_engine)
        
        # 获取识别引擎输入输出信息
        self.recognize_input_shape = self.recognize_engine.get_binding_shape(0)
        self.recognize_output_shape = self.recognize_engine.get_binding_shape(1)
        LOGGER.info(f"Recognition engine - Input: {self.recognize_input_shape}, Outputs: {self.recognize_engine.get_binding_shape(1)} and {self.recognize_engine.get_binding_shape(2)}")
        
        # 车牌字符集（根据实际模型调整）
        self.characters = '#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品'
        
        # 车牌颜色类别（根据实际模型调整）
        self.plate_colors = ['黑色','蓝色','绿色','白色','黄色']
        
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
        bindings = []
        
        for binding_idx in range(engine.num_bindings):
            binding_name = engine.get_binding_name(binding_idx)
            binding_shape = engine.get_binding_shape(binding_idx)
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
            
            # 计算内存大小
            size = trt.volume(binding_shape) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            
            if engine.binding_is_input(binding_idx):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                LOGGER.info(f"Input binding {binding_idx}: name='{binding_name}', dtype={dtype}, shape={binding_shape}")
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                LOGGER.info(f"Output binding {binding_idx}: name='{binding_name}', dtype={dtype}, shape={binding_shape}")
        
        return {
            'host_inputs': host_inputs,
            'cuda_inputs': cuda_inputs,
            'host_outputs': host_outputs,
            'cuda_outputs': cuda_outputs,
            'bindings': bindings
        }
    
    def _preprocess(self, image: np.ndarray, target_size: Tuple[int, int], normalize=True):
        """通用预处理函数"""
        # 调整大小并转换颜色通道
        img_resized = cv2.resize(image, target_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        if normalize:
            # 归一化处理
            img_norm = img_rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_norm = (img_norm - mean) / std
        else:
            img_norm = img_rgb.astype(np.float32)
            
        # 转换为CHW格式
        img_chw = np.transpose(img_norm, [2, 0, 1])
        return np.ascontiguousarray(img_chw)
    
    def _infer(self, context: trt.IExecutionContext, bindings: dict, input_data: np.ndarray):
        """执行推理"""
        self.ctx.push()
        
        host_inputs = bindings['host_inputs']
        cuda_inputs = bindings['cuda_inputs']
        host_outputs = bindings['host_outputs']
        cuda_outputs = bindings['cuda_outputs']
        bindings_list = bindings['bindings']
        
        # 复制数据到主机缓冲区
        np.copyto(host_inputs[0], input_data.ravel())
        
        # 传输数据到GPU
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], self.stream)
        
        # 执行推理
        context.execute_async_v2(bindings=bindings_list, stream_handle=self.stream.handle)
        
        # 传回结果
        outputs = []
        for i in range(len(host_outputs)):
            cuda.memcpy_dtoh_async(host_outputs[i], cuda_outputs[i], self.stream)
        
        self.stream.synchronize()
        self.ctx.pop()
        
        # 返回所有输出
        return [output.copy() for output in host_outputs]
    
    def detect_plates(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        """
        检测车牌位置并返回车牌区域列表
        返回: (车牌图像, 边界框坐标(x1, y1, x2, y2))
        """
        # 获取检测引擎输入尺寸 (1, 3, 640, 640)
        _, _, input_h, input_w = self.detect_input_shape
        
        # 预处理
        input_image = self._preprocess(image, (input_w, input_h))
        
        # 执行推理
        outputs = self._infer(self.detect_context, self.detect_bindings, input_image)
        
        # 解析检测结果
        # 输出形状: (1, 25200, 15)
        # 假设格式: [batch, num_boxes, (x, y, w, h, conf, class_conf1, class_conf2, ...)]
        detection_output = outputs[0].reshape(self.detect_output_shape)
        
        # 后处理 - 解析边界框
        plate_boxes = self._parse_detection_output(detection_output, image.shape)
        
        # 裁剪车牌区域
        plates = []
        for box in plate_boxes:
            x1, y1, x2, y2, conf = box
            plate_img = image[int(y1):int(y2), int(x1):int(x2)]
            if plate_img.size > 0:  # 确保有有效区域
                plates.append((plate_img, (x1, y1, x2, y2)))
            
        return plates
    
    def _parse_detection_output(self, detection_output: np.ndarray, img_shape: tuple, conf_threshold=0.5):
        """
        解析检测输出为边界框列表 (x1, y1, x2, y2, confidence)
        使用简单的阈值过滤
        """
        h, w = img_shape[:2]
        boxes = []
        
        # 输出形状: (1, 25200, 15)
        # 假设每个检测框有15个值: [x, y, w, h, conf, class_scores...]
        num_boxes = detection_output.shape[1]
        
        for i in range(num_boxes):
            box_data = detection_output[0, i]
            
            # 提取边界框信息
            x_center, y_center, box_w, box_h, conf = box_data[:5]
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
                
            # 转换到原始图像坐标
            x1 = (x_center - box_w / 2) * w
            y1 = (y_center - box_h / 2) * h
            x2 = (x_center + box_w / 2) * w
            y2 = (y_center + box_h / 2) * h
            
            # 确保坐标在图像范围内
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            
            boxes.append((x1, y1, x2, y2, conf))
            
        # 简单NMS处理 (实际应用中应使用更复杂的NMS)
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)  # 按置信度排序
        filtered_boxes = []
        
        while boxes:
            current = boxes.pop(0)
            filtered_boxes.append(current)
            boxes = [box for box in boxes if self._iou(current, box) < 0.5]
            
        return filtered_boxes
    
    def _iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1, _ = box1
        x1_2, y1_2, x2_2, y2_2, _ = box2
        
        # 计算交集区域
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # 计算并集区域
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def recognize_plate(self, plate_image: np.ndarray) -> Tuple[str, str]:
        """
        识别单个车牌图像中的文本和颜色
        返回: (车牌文本, 车牌颜色)
        """
        # 获取识别引擎输入尺寸 (1, 3, 48, 168)
        _, _, input_h, input_w = self.recognize_input_shape
        
        # 预处理（识别模型可能需要不同处理）
        input_image = self._preprocess(plate_image, (input_w, input_h))
        
        # 执行推理
        outputs = self._infer(self.recognize_context, self.recognize_bindings, input_image)
        
        # 解析输出
        # 输出1: (1, 5) - 车牌颜色概率
        # 输出2: (1, 21, 78) - 车牌字符概率
        color_probs = outputs[0].flatten()
        char_probs = outputs[1]
        
        # 解析车牌文本
        plate_text = self._decode_text_output(char_probs)
        
        # 解析车牌颜色
        plate_color = self._decode_color_output(color_probs)
        
        return plate_text, plate_color
    
    def _decode_text_output(self, char_probs: np.ndarray) -> str:
        """
        将模型输出解码为车牌字符串
        使用CTC风格解码
        """
        # 形状: (1, 21, 78) -> 取第一个批次
        char_probs = char_probs[0]
        
        # 获取每个时间步的最大概率字符索引
        char_indices = np.argmax(char_probs, axis=1)
        
        # 解码字符序列
        plate_str = []
        prev_char = None
        for idx in char_indices:
            # 跳过空白符（索引0对应'#'）
            if idx == 0:
                prev_char = None  # 重置前一个字符
                continue
                
            char = self.characters[idx]
            
            # 跳过重复字符（除非是分隔符）
            if char != prev_char:
                plate_str.append(char)
                prev_char = char
        
        return ''.join(plate_str)
    
    def _decode_color_output(self, color_probs: np.ndarray) -> str:
        """解码车牌颜色"""
        color_idx = np.argmax(color_probs)
        return self.plate_colors[color_idx] if color_idx < len(self.plate_colors) else "unknown"
    
    def warm_up(self):
        LOGGER.info('Warming up engines...')
        # 创建符合检测引擎输入尺寸的假图像
        _, _, detect_h, detect_w = self.detect_input_shape
        dummy_detect = np.zeros((detect_h, detect_w, 3), dtype=np.uint8)
        
        # 创建符合识别引擎输入尺寸的假车牌
        _, _, rec_h, rec_w = self.recognize_input_shape
        dummy_plate = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
        
        for _ in range(self.warm_up_turns):
            plates = self.detect_plates(dummy_detect)
            for plate_img, _ in plates:
                self.recognize_plate(plate_img)
        LOGGER.info('Warm up completed')
    
    def infer(self, image: np.ndarray) -> List[Tuple[str, str]]:
        """
        端到端车牌识别流程
        返回: [(车牌文本1, 颜色1), (车牌文本2, 颜色2), ...]
        """
        # 步骤1：检测车牌区域
        plates = self.detect_plates(image)
        
        # 步骤2：识别每个车牌
        results = []
        for plate_img, bbox in plates:
            # 确保车牌区域有效
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