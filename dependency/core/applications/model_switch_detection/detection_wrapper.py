# This file is the wrapper for the detection. 
# It implements the detection interface and uses the model switch to switch between the models.

def _import_modules():
    """Import modules based on execution context"""

    if __name__ == '__main__':

        from switch_module.random_switch import RandomSwitch
        from switch_module.rule_based_switch import RuleBasedSwitch
        from inference_module.yolo_inference import YoloInference
        from inference_module.ofa_inference import OfaInference
    else:
        from .switch_module.random_switch import RandomSwitch
        from .switch_module.rule_based_switch import RuleBasedSwitch
        from .inference_module.yolo_inference import YoloInference
        from .inference_module.ofa_inference import OfaInference
    return RandomSwitch, RuleBasedSwitch, YoloInference, OfaInference

from typing import List
import numpy as np
import cv2

RandomSwitch, RuleBasedSwitch, YoloInference, OfaInference = _import_modules()

class ModelSwitchDetection:

    def __init__(self, model_type: str, switch_type: str,
                 decision_interval: int,
                 *args, **kwargs):
        
        if model_type == 'yolo':
            self.detector = YoloInference(*args, **kwargs)
        elif model_type == 'ofa':
            self.detector = OfaInference(*args, **kwargs)

        else:
            raise ValueError('Invalid type')
        
        if switch_type == 'random':
            self.switcher = RandomSwitch(decision_interval, self.detector)
        elif switch_type == 'rule_based':
            self.switcher = RuleBasedSwitch(decision_interval, self.detector)
        else:
            raise ValueError('Invalid switch type')


    def infer(self, image: np.ndarray):
        return self.detector.infer(image)

    def __call__(self, images: List[np.ndarray]):

        output = []

        for image in images:
            output.append(self.infer(image))

        return output
    
if __name__ == '__main__':

    # ============================ test yolo ============================
    import time
    # these params will be specified in the field of DETECTOR_PARAMETERS of yaml file
    detector_wrapper = ModelSwitchDetection(model_type='yolo', 
                                    switch_type='rule_based',
                                    decision_interval=10,
                                    weights_dir='yolov5_weights',
                                    model_names = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                                    model_accuracy =[28.0, 37.4, 45.4, 49.0, 50.7])
    input_path = "test_data/img.png"
    image = cv2.imread(input_path)
    while True:
        result = detector_wrapper([image])
        # print(len(result[0][0]))

    # ============================ test ofa ============================
    weights_dir = 'ofa_weights'
    subnet_nums = 5
    ofa_det_type = 'mbv3_faster_rcnn'
    subnet_archs = [
                    {'ks': [7, 7, 3, 3, 3, 7, 7, 3, 5, 7, 5, 5, 5, 3, 3, 7, 7, 7, 3, 3], 'e': [3, 4, 4, 6, 6, 4, 4, 4, 3, 3, 4, 3, 3, 4, 4, 4, 6, 6, 4, 6], 'd': [2, 2, 3, 2, 2]},
                    {'ks': [5, 7, 3, 3, 7, 7, 7, 3, 5, 7, 3, 3, 5, 3, 7, 7, 5, 5, 5, 5], 'e': [4, 6, 4, 3, 4, 6, 6, 6, 3, 6, 6, 6, 4, 6, 4, 4, 4, 6, 4, 6], 'd': [4, 2, 2, 3, 4]},
                    {'ks': [5, 3, 5, 7, 7, 7, 7, 3, 3, 7, 3, 3, 3, 5, 3, 7, 5, 5, 7, 3], 'e': [6, 6, 6, 3, 4, 4, 6, 4, 3, 6, 6, 4, 3, 4, 4, 4, 3, 3, 4, 6], 'd': [2, 2, 3, 2, 4]},
                    {'ks': [3, 3, 3, 7, 5, 7, 3, 5, 7, 7, 5, 3, 5, 5, 3, 7, 7, 5, 5, 5], 'e': [3, 3, 4, 6, 3, 3, 6, 3, 4, 4, 3, 3, 3, 4, 4, 3, 4, 3, 3, 3], 'd': [4, 4, 2, 4, 4]},
                    {'ks': [3, 5, 5, 5, 5, 3, 3, 7, 3, 5, 5, 3, 5, 3, 3, 3, 7, 3, 7, 3], 'e': [3, 6, 6, 6, 4, 6, 3, 4, 6, 6, 4, 3, 3, 6, 3, 6, 6, 6, 4, 3], 'd': [3, 3, 2, 4, 3]}
                    ]
    subnet_accuracy = [10.0, 12.0, 14.0, 16.0, 18.0]

    detector_wrapper = ModelSwitchDetection(model_type='ofa',
                                    switch_type='random',
                                    decision_interval=10,
                                    ofa_det_type=ofa_det_type,
                                    subnet_nums=subnet_nums,
                                    subnet_archs=subnet_archs,
                                    subnet_accuracy=subnet_accuracy,
                                    weights_dir=weights_dir)
    
    input_path = "test_data/img.png"
    image = cv2.imread(input_path)
    while True:
        result = detector_wrapper([image])
        print(len(result[0][0]))
