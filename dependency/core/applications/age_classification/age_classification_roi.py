import numpy as np
from typing import List, Dict
from .age_classification import AgeClassification



class AgeClassificationRoi:
    """ROI-aware wrapper with simple per-roi_id cache."""
    def __init__(self, trt_weights, non_trt_weights, device=0):
        self.model = AgeClassification(trt_weights=trt_weights, non_trt_weights=non_trt_weights, device=device)
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
            res = self.model.infer(face)
            self.cache[rid] = res
            results.append(res)
        return results
