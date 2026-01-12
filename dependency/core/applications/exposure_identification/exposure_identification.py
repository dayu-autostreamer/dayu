import numpy as np
from typing import List, Dict

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class ExposureIdentification:

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
                from .exposure_identification_with_tensorrt import ExposureIdentificationTensorRT10
                self.model = ExposureIdentificationTensorRT10(weights=self.trt_weights, device=self.device)
            elif jetpack_version in [4, 5]:
                LOGGER.info(f'Using TensorRT 8 (JetPack {jetpack_version})')
                from .exposure_identification_with_tensorrt import ExposureIdentificationTensorRT8
                self.model = ExposureIdentificationTensorRT8(weights=self.trt_weights, device=self.device)
            else:
                LOGGER.warning(f'Unknown JETPACK version: {jetpack_version}，attempting to use TensorRT 8')
                from .exposure_identification_with_tensorrt import ExposureIdentificationTensorRT8
                self.model = ExposureIdentificationTensorRT8(weights=self.trt_weights, device=self.device)
        else:
            from .exposure_identification_without_tensorrt import ExposureIdentificationResNet50
            self.model = ExposureIdentificationResNet50(weights=self.non_trt_weights, device=self.device)

    def _infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            output.append(self._infer(image))
        return output

    def _calculate_flops(self):
        try:
            from .exposure_identification_without_tensorrt import ExposureIdentificationResNet50
            model = ExposureIdentificationResNet50(weights=self.non_trt_weights, device=self.device)
            self.flops = FlopsEstimator(model=model.model, input_shape=(3, 224, 224)).compute_flops()
            del model
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed:{e}')
            LOGGER.exception(e)


class ExposureIdentificationRoi:
    """ROI-aware wrapper with per-roi_id cache."""
    def __init__(self, trt_weights, non_trt_weights, device=0):
        self.model = ExposureIdentification(trt_weights=trt_weights, non_trt_weights=non_trt_weights, device=device)
        self.cache: Dict[int, any] = {}

    def reset_cache(self):
        self.cache.clear()

    @property
    def flops(self):
        return getattr(self.model, 'flops', 0)

    def __call__(self, images: List[np.ndarray], roi_ids: List[int]):
        results = []
        for img, rid in zip(images, roi_ids):
            if rid in self.cache:
                results.append(self.cache[rid])
                continue
            res = self.model._infer(img)
            self.cache[rid] = res
            results.append(res)
        return results
