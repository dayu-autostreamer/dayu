import numpy as np
from typing import List

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
            jetpack_version = Context.get_parameter('JETPACK', direct=False)

            # JETPACK 6 uses TensorRT10, JETPACK 4/5 uses TensorRT8
            if jetpack_version == 6:
                LOGGER.info('Using TensorRT 10 (JetPack 6)')
                from .license_plate_recognition_with_tensorrt import LicensePlateRecognitionTensorRT10
                self.model = LicensePlateRecognitionTensorRT10(detect_weights=self.det_trt_weights,
                                                               recognize_weights=self.rec_trt_weights,
                                                               device=self.device)
            elif jetpack_version in [4, 5]:
                LOGGER.info(f'Using TensorRT 8 (JetPack {jetpack_version})')
                from .license_plate_recognition_with_tensorrt import LicensePlateRecognitionTensorRT8
                self.model = LicensePlateRecognitionTensorRT8(detect_weights=self.det_trt_weights,
                                                              recognize_weights=self.rec_trt_weights,
                                                              device=self.device)
            else:
                LOGGER.warning(f'Unknown JETPACK version: {jetpack_version}，attempting to use TensorRT 8')
                from .license_plate_recognition_with_tensorrt import LicensePlateRecognitionTensorRT8
                self.model = LicensePlateRecognitionTensorRT8(detect_weights=self.det_trt_weights,
                                                              recognize_weights=self.rec_trt_weights,
                                                              device=self.device)
        else:
            from .license_plate_recognition_without_tensorrt import LicensePlateRecognitionPT
            self.model = LicensePlateRecognitionPT(detect_model_path=self.non_det_trt_weights,
                                                   rec_model_path=self.non_rec_trt_weights,
                                                   device=self.device)

    def infer(self, image):
        return self.model.infer(image)

    def __call__(self, images: List[np.ndarray]):
        output = []

        for image in images:
            output.append(self.infer(image))
        return output

    def _calculate_flops(self):
        try:
            from .license_plate_recognition_without_tensorrt import LicensePlateRecognitionPT
            model = LicensePlateRecognitionPT(detect_model_path=self.non_det_trt_weights,
                                              rec_model_path=self.non_rec_trt_weights, device=self.device)
            flops_det = FlopsEstimator(model=model.detect_model, input_shape=(3, 512, 640)).compute_flops()
            flops_rec = FlopsEstimator(model=model.plate_rec_model, input_shape=(3, 48, 216)).compute_flops()
            self.flops = flops_det + flops_rec
            del model

        except Exception as e:
            LOGGER.warning(f'Get model FLOPs failed:{e}')
            LOGGER.exception(e)
