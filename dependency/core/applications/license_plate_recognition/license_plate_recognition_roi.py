import numpy as np
from typing import List, Dict

from .license_plate_recognition import LicensePlateRecognition


class LicensePlateRecognitionRoi:
    """ROI-aware wrapper with per-roi_id cache; delegates to existing pipeline on ROI crops."""

    def __init__(self, det_trt_weights, non_det_trt_weights, rec_trt_weights, non_rec_trt_weights, device=0):
        self.model = LicensePlateRecognition(det_trt_weights=det_trt_weights,
                                             non_det_trt_weights=non_det_trt_weights,
                                             rec_trt_weights=rec_trt_weights,
                                             non_rec_trt_weights=non_rec_trt_weights,
                                             device=device)
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
