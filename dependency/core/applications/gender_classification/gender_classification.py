import cv2
import numpy as np
from typing import List, Dict

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class GenderClassification:
    def __init__(self, trt_weights, non_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.trt_weights = Context.get_file_path(trt_weights)
        self.non_trt_weights = Context.get_file_path(non_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            from .gender_classification_with_tensorrt import GenderClassificationTensorRT
            self.model = GenderClassificationTensorRT(weights=self.trt_weights, device=self.device)
        else:
            from .gender_classification_without_tensorrt import GenderClassificationResNet18
            self.model = GenderClassificationResNet18(weights=self.non_trt_weights, device=self.device)

    def _infer(self, image):
        return self.model.infer(image)

    def __call__(self, faces: List[np.ndarray]):
        output = []

        for face in faces:
            output.append(self._infer(face))

        return output

    def _calculate_flops(self):
        try:
            from .gender_classification_without_tensorrt import GenderClassificationResNet18
            model = GenderClassificationResNet18(weights=self.non_trt_weights, device=self.device)
            self.flops = FlopsEstimator(model=model.model, input_shape=(3, 224, 224)).compute_flops()
            del model
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed: {e}')
            LOGGER.exception(e)


class GenderClassificationRoi:
    """
    ROI-aware wrapper: caches classification result per roi_id to avoid redundant inference across frames.
    __call__(faces, roi_ids) -> list of results aligned with inputs.
    """
    def __init__(self, trt_weights, non_trt_weights, device=0):
        self.model = GenderClassification(trt_weights=trt_weights, non_trt_weights=non_trt_weights, device=device)
        self.cache: Dict[int, str] = {}

    def reset_cache(self):
        self.cache.clear()

    @property
    def flops(self):
        return getattr(self.model, 'flops', 0)

    def __call__(self, faces: List[np.ndarray], roi_ids: List[int]):
        results = []
        for face, rid in zip(faces, roi_ids):
            if rid in self.cache:
                results.append(self.cache[rid])
                continue
            res = self.model._infer(face)
            self.cache[rid] = res
            results.append(res)
        return results
