import numpy as np
from typing import List, Dict

from core.lib.common import Context, LOGGER
from core.lib.estimation import FlopsEstimator


class LicensePlateRecognition:

    def __init__(self, det_trt_weights, non_det_trt_weights, rec_trt_weights, non_rec_trt_weights, device=0):

        use_tensorrt = Context.get_parameter('USE_TENSORRT', direct=False)
        self.det_trt_weights = Context.get_file_path(det_trt_weights)
        self.non_det_trt_weights = Context.get_file_path(non_det_trt_weights)
        self.rec_trt_weights = Context.get_file_path(rec_trt_weights)
        self.non_rec_trt_weights = Context.get_file_path(non_rec_trt_weights)
        self.device = device

        self.flops = 0
        self._calculate_flops()

        if use_tensorrt:
            from .license_plate_recognition_with_tensorrt import LicensePlateRecognitionTensorRT
            self.model = LicensePlateRecognitionTensorRT(detect_weights=self.det_trt_weights,
                                              recognize_weights=self.rec_trt_weights, device=self.device)
        else:
            from .license_plate_recognition_without_tensorrt import LicensePlateRecognitionPT
            self.model = LicensePlateRecognitionPT(detect_model_path=self.non_det_trt_weights, rec_model_path=self.non_rec_trt_weights, device=self.device)

    def _infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            output.append(self._infer(image))
        return output

    def _calculate_flops(self):
        try:
            from .license_plate_recognition_without_tensorrt import LicensePlateRecognitionPT
            model = LicensePlateRecognitionPT(detect_model_path=self.non_det_trt_weights, rec_model_path=self.non_rec_trt_weights, device=self.device)
            flops_det = FlopsEstimator(model=model.detect_model, input_shape=(3, 512, 640)).compute_flops()
            flops_rec = FlopsEstimator(model=model.plate_rec_model, input_shape=(3, 48, 216)).compute_flops()
            self.flops = flops_det + flops_rec
            del model
            
        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed:{e}')
            LOGGER.exception(e)


class LicensePlateRecognitionRoi:
    """ROI-aware wrapper with per-roi_id cache; delegates to existing pipeline on ROI crops."""
    def __init__(self, det_trt_weights, non_det_trt_weights, rec_trt_weights, non_rec_trt_weights, device=0):
        self.model = LicensePlateRecognition(det_trt_weights=det_trt_weights,
                                             non_det_trt_weights=non_det_trt_weights,
                                             rec_trt_weights=rec_trt_weights,
                                             non_rec_trt_weights=non_rec_trt_weights,
                                             device=device)
        self.cache: Dict[int, any] = {}

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
