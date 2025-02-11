import threading
from .base_inference import BaseInference
from typing import List
import numpy as np
import os
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{cur_dir}/yolov5")
from models.common import AutoShape
from models.experimental import attempt_load
import torch
import warnings

warnings.filterwarnings("ignore")

class YoloInference(BaseInference):
    def __init__(self, *args, **kwargs):
        '''
        Load all models, do all the necessary initializations.
        '''
        # models should be a sorted list of pareto optimal models, so that the switcher can switch between them.
        self.allowed_yolo_models = ['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
        self.models = []
        assert 'weights_dir' in kwargs, 'weights_dir not provided'
        self.weights_dir = kwargs['weights_dir']
        for model_name in self.allowed_yolo_models:
            model_path = f"{self.weights_dir}/{model_name}.pt"
            assert os.path.exists(model_path), f"Model weights file not found: {model_path}"
        self.current_model_index = None
        self.model_switch_lock = threading.Lock()
        self._load_all_models()

    def _load_all_models(self):
        print('Loading all YOLOv5 models...')
        for model_name in self.allowed_yolo_models:
            model_path = f"{self.weights_dir}/{model_name}.pt"
            try:
                print(f'Loading model: {model_name}...')
                model = attempt_load(model_path)
                model = AutoShape(model)
                if torch.cuda.is_available():
                    model = model.cuda()
                self.models.append(model)
                print(f'Model loaded: {model_name}.')
            except Exception as e:
                print(f'Error loading model {model_name}: {str(e)}!')
            
        print('All models loaded.')
        self.current_model_index = 0
        print(f'Switched to model: {self.allowed_yolo_models[self.current_model_index]}.')

    def switch_model(self, index: int):
        '''
        Switch the model to the one specified in the arguments.
        '''
        with self.model_switch_lock:
            if index >= len(self.models) or index < 0:
                raise ValueError('Invalid model index')
            self.current_model_index = index
            print(f'Switched to model: {self.allowed_yolo_models[self.current_model_index]}')

    def get_models_num(self):
        '''
        Get the number of models.
        '''
        return len(self.models)

    def infer(self, image: np.ndarray):
        '''
        Do the inference on the image.
        '''
        with self.model_switch_lock:
            if self.current_model_index is None:
                raise ValueError('Model not loaded')
            model = self.models[self.current_model_index]
            with torch.no_grad():
                result = model(image)
            result_data = self.process_results(result)
            return result_data
        
    def process_results(self, results):
        '''
        Extract from yolo detection results three np lists: boxes, scores, labels
        '''
        boxes = results.xyxy[0][:, :4].cpu().numpy().tolist()
        scores = results.xyxy[0][:, 4].cpu().numpy().tolist()
        labels = results.xyxy[0][:, 5].cpu().numpy().tolist()

        return boxes, scores, labels

