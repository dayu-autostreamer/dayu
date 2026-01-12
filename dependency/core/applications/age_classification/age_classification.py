import numpy as np
from typing import List

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class AgeClassification:
    def __init__(self, trt_weights, non_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.trt_weights = Context.get_file_path(trt_weights)
        self.non_trt_weights = Context.get_file_path(non_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            jetpack_version = Context.get_parameter('JETPACK', direct=False)

            # JETPACK 6 uses TensorRT10, JETPACK 4/5 uses TensorRT8
            if jetpack_version == 6:
                LOGGER.info('Using TensorRT 10 (JetPack 6)')
                from .age_classification_with_tensorrt import AgeClassificationTensorRT10
                self.model = AgeClassificationTensorRT10(weights=self.trt_weights, device=self.device)
            elif jetpack_version in [4, 5]:
                LOGGER.info(f'Using TensorRT 8 (JetPack {jetpack_version})')
                from .age_classification_with_tensorrt import AgeClassificationTensorRT8
                self.model = AgeClassificationTensorRT8(weights=self.trt_weights, device=self.device)
            else:
                LOGGER.warning(f'Unknown JETPACK version: {jetpack_version}，attempting to use TensorRT 8')
                from .age_classification_with_tensorrt import AgeClassificationTensorRT8
                self.model = AgeClassificationTensorRT8(weights=self.trt_weights, device=self.device)
        else:
            from .age_classification_without_tensorrt import AgeClassificationResNet18
            self.model = AgeClassificationResNet18(weights=self.non_trt_weights, device=self.device)

    def infer(self, image):
        return self.model.infer(image)

    def __call__(self, faces: List[np.ndarray]):
        output = []

        for face in faces:
            output.append(self.infer(face))

        return output

    def _calculate_flops(self):
        try:
            from .age_classification_without_tensorrt import AgeClassificationResNet18
            model = AgeClassificationResNet18(weights=self.non_trt_weights, device=self.device)
            self.flops = FlopsEstimator(model=model.model, input_shape=(3, 224, 224)).compute_flops()
            del model
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed: {e}')
            LOGGER.exception(e)
