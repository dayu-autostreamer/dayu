import os
import numpy as np
from typing import List

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class FaceDetection:

    def __init__(self, trt_weights, trt_plugin_library, non_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.trt_weights = Context.get_file_path(trt_weights)
        self.trt_plugin_library = Context.get_file_path(trt_plugin_library)
        self.non_trt_weights = Context.get_file_path(non_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            # 检查 TensorRT 版本环境变量
            trt_version = os.environ.get('version', '8')
            
            if trt_version == '10':
                LOGGER.info('Using TensorRT 10')
                from .face_detection_with_tensorrt import FaceDetectionTensorRT10
                self.model = FaceDetectionTensorRT10(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device
                )
            elif trt_version == '8':
                LOGGER.info('Using TensorRT 8')
                from .face_detection_with_tensorrt import FaceDetectionTensorRT8
                self.model = FaceDetectionTensorRT8(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device
                )
            else:
                LOGGER.warning(f'不支持的 TensorRT 版本: {trt_version}，仅支持版本 8 和 10。将使用非 TensorRT 模式。')
                from .face_detection_without_tensorrt import FaceDetectionRetinaFace
                self.model = FaceDetectionRetinaFace(
                    weights=self.non_trt_weights,
                    device=self.device
                )
        else:
            from .face_detection_without_tensorrt import FaceDetectionRetinaFace
            self.model = FaceDetectionRetinaFace(
                weights=self.non_trt_weights,
                device=self.device
            )

    def _infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            result_boxes, result_scores, result_class_id, result_roi_id = self._infer(image)
            output.append((result_boxes, result_scores, result_class_id, result_roi_id))
        return output

    def _calculate_flops(self):
        try:
            from .face_detection_without_tensorrt import FaceDetectionRetinaFace
            model = FaceDetectionRetinaFace(weights=self.non_trt_weights, device=self.device)
            if hasattr(model, 'model'):
                self.flops = FlopsEstimator(model=model.model, input_shape=(3, 640, 640)).compute_flops()
            del model
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed: {e}')
            LOGGER.exception(e)
