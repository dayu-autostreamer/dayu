import numpy as np
from typing import List

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class CarDetection:

    def __init__(self, trt_weights, trt_plugin_library, non_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.trt_weights = Context.get_file_path(trt_weights)
        self.trt_plugin_library = Context.get_file_path(trt_plugin_library)
        self.non_trt_weights = Context.get_file_path(non_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            jetpack_version = Context.get_parameter('JETPACK', direct=False)

            # JETPACK 6 uses TensorRT10, JETPACK 4/5 uses TensorRT8
            if jetpack_version == 6:
                LOGGER.info('Using TensorRT 10 (JetPack 6)')
                from .car_detection_with_tensorrt import CarDetectionTensorRT10
                self.model = CarDetectionTensorRT10(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device)
            elif jetpack_version in [4, 5]:
                LOGGER.info(f'Using TensorRT 8 (JetPack {jetpack_version})')
                from .car_detection_with_tensorrt import CarDetectionTensorRT8
                self.model = CarDetectionTensorRT8(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device)
            else:
                LOGGER.warning(f'Unknown JETPACK version: {jetpack_version}, attempting to use TensorRT 8')
                from .car_detection_with_tensorrt import CarDetectionTensorRT8
                self.model = CarDetectionTensorRT8(
                    weights=self.trt_weights,
                    plugin_library=self.trt_plugin_library,
                    device=self.device)
        else:
            from .car_detection_without_tensorrt import CarDetectionYoloV5
            self.model = CarDetectionYoloV5(weights=self.non_trt_weights, device=self.device)

    def infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            result_boxes, result_scores, result_class_id = self.infer(image)
            result_roi_id = list(range(len(result_boxes)))
            output.append((result_boxes, result_scores, result_class_id, result_roi_id))
        return output

    def _calculate_flops(self):
        try:
            from .car_detection_without_tensorrt import CarDetectionYoloV5
            model = CarDetectionYoloV5(weights=self.non_trt_weights, device=self.device)
            self.flops = FlopsEstimator(model=model.model, input_shape=(3, 640, 640)).compute_flops()
            del model
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed: {e}')
            LOGGER.exception(e)
