import numpy as np
from typing import List

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class VehicleDetection:

    def __init__(self, trt_weights, trt_plugin_library, non_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.trt_weights = Context.get_file_path(trt_weights)
        self.trt_plugin_library = Context.get_file_path(trt_plugin_library)
        self.non_trt_weights = Context.get_file_path(non_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            from .vehicle_detection_with_tensorrt import VehicleDetectionTensorRT
            self.model = VehicleDetectionTensorRT(weights=self.trt_weights,
                                              plugin_library=self.trt_plugin_library, device=self.device)
        else:
            from .vehicle_detection_without_tensorrt import VehicleDetectionYoloV8
            self.model = VehicleDetectionYoloV8(weights=self.non_trt_weights, device=self.device)

    def _infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            output.append(self._infer(image))
        return output

    def _calculate_flops(self):
        try:
            from .vehicle_detection_without_tensorrt import VehicleDetectionYoloV8
            model = VehicleDetectionYoloV8(weights=self.non_trt_weights, device=self.device)
            self.flops = FlopsEstimator(model = model.model, input_shape=(3, 640, 640)).compute_flops()
            del model
            
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed:{e}')
            LOGGER.exception(e)  


