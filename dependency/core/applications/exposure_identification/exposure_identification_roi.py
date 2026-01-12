import numpy as np
from typing import List, Dict

from .exposure_identification import ExposureIdentification


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
            res = self.model.infer(img)
            self.cache[rid] = res
            results.append(res)
        return results
