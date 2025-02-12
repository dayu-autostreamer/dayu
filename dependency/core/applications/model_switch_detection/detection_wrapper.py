# This file is the wrapper for the detection. 
# It implements the detection interface and uses the model switch to switch between the models.

from typing import List
import numpy as np
from inference_module.yolo_inference import YoloInference
from switch_module.random_switch import RandomSwitch
import cv2

class ModelSwitchDetection:

    def __init__(self, type: str, decision_interval: int,
                 *args, **kwargs):
        if type == 'yolo':
            self.detector = YoloInference(*args, **kwargs)
            self.switcher = RandomSwitch(self.detector.get_models_num(), 
                                         decision_interval, 
                                         self.detector)
        elif type == 'efficientdet':
            pass
        elif type == 'ofa':
            pass
        else:
            raise ValueError('Invalid type')


    def infer(self, image: np.ndarray):
        return self.detector.infer(image)

    def __call__(self, images: List[np.ndarray]):

        output = []

        for image in images:
            output.append(self.infer(image))

        return output
    
if __name__ == '__main__':

    # test yolo
    import time
    # these params will be specified in the field of DETECTOR_PARAMETERS of yaml file
    detector_wrapper = ModelSwitchDetection('yolo', 
                                    20,
                                    weights_dir='yolov5_weights')
    input_path = "test_data/img2.jpg"
    image = cv2.imread(input_path)
    for i in range(100):
        result = detector_wrapper([image])
        print(len(result[0][0]))
