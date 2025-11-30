import numpy as np
from typing import List, Dict

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class CategoryIdentification:

    def __init__(self, trt_weights, trt_plugin_library, non_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.trt_weights = Context.get_file_path(trt_weights)
        self.trt_plugin_library = Context.get_file_path(trt_plugin_library)
        self.non_trt_weights = Context.get_file_path(non_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            from .category_identification_with_tensorrt import CategoryIdentificationTensorRT
            self.model = CategoryIdentificationTensorRT(weights=self.trt_weights,
                                              plugin_library=self.trt_plugin_library, device=self.device)
        else:
            from .category_identification_without_tensorrt import CategoryIdentificationYolov8
            self.model = CategoryIdentificationYolov8(weights=self.non_trt_weights, device=self.device)

    def _infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            output.append(self._infer(image))
        return output

    def _calculate_flops(self):
        try:
            from .category_identification_without_tensorrt import CategoryIdentificationYolov8
            model = CategoryIdentificationYolov8(weights=self.non_trt_weights, device=self.device)
            self.flops = FlopsEstimator(model = model.model, input_shape=(3, 640, 640)).compute_flops()
            del model
            
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed:{e}')
            LOGGER.exception(e)


class CategoryIdentificationRoi:
    """ROI-aware wrapper with simple per-roi_id cache."""
    def __init__(self, trt_weights, trt_plugin_library, non_trt_weights, device=0):
        self.model = CategoryIdentification(trt_weights=trt_weights, trt_plugin_library=trt_plugin_library,
                                            non_trt_weights=non_trt_weights, device=device)
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
