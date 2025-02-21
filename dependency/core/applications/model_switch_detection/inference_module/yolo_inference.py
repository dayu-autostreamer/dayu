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
import time
import cv2

from core.lib.common import Context

warnings.filterwarnings("ignore")

class YoloInference(BaseInference):
    def __init__(self, *args, **kwargs):
        '''
        Load all models, do all the necessary initializations.
        '''
        super().__init__(*args, **kwargs)
        # models should be a sorted list of pareto optimal models, so that the switcher can switch between them.
        self.allowed_yolo_models = kwargs['model_names']
        # official mAP values.
        self.model_accuracy =kwargs['model_accuracy']
        # assert len(self.allowed_yolo_models) == len(self.model_accuracy), 'Model names and accuracies do not match'
        self.model_latency = []
        # ema_alpha for model latency updates
        self.ema_alpha = 0.2
        self.models = []
        # assert 'weights_dir' in kwargs, 'weights_dir not provided'
        self.weights_dir = kwargs['weights_dir']
        # for model_name in self.allowed_yolo_models:
        #     model_path = f"{self.weights_dir}/{model_name}.pt"
        #     assert os.path.exists(model_path), f"Model weights file not found: {model_path}"
        self.current_model_index = None
        self.model_switch_lock = threading.Lock()
        self._load_all_models()
        self._measure_initial_latencies()

    def _load_all_models(self):
        print('Loading all YOLOv5 models...')
        for model_name in self.allowed_yolo_models:
            # model_path = f"{self.weights_dir}/{model_name}.pt"
            relative_model_path = f"{self.weights_dir}/{model_name}.pt"
            model_path = Context.get_file_path(relative_model_path)
            try:
                print(f'Loading model: {model_name}...')
                model = attempt_load(model_path)
                model = AutoShape(model)
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                self.models.append(model)
                print(f'Model loaded: {model_name}.')
            except Exception as e:
                print(f'Error loading model {model_name}: {str(e)}!')
            
        print('All models loaded.')
        self.current_model_index = 0
        print(f'Switched to model: {self.allowed_yolo_models[self.current_model_index]}.')

    def _measure_initial_latencies(self):
        print("Measuring initial latencies...")
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for idx, model in enumerate(self.models):
            times = []
            # 预热
            for _ in range(5):
                model(dummy_input)
            
            # 测量
            for _ in range(10):
                start_time = time.perf_counter()
                with torch.no_grad():
                    model(dummy_input)
                times.append(time.perf_counter() - start_time)
            self.model_latency.append(np.mean(times))
            print(f'Model {idx} takes time: {self.model_latency[idx]}')

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
    
    def get_models_accuracy(self):
        '''
        Get the accuracy of the models.
        Returns a list of floats.
        '''
        return self.model_accuracy
    
    def get_models_latency(self):
        '''
        Get the latency of the models.
        Returns a list of floats.
        '''
        return self.model_latency

    def infer(self, image: np.ndarray):
        '''
        Do the inference on the image.
        '''
        with self.model_switch_lock:
            model = self.models[self.current_model_index]
            start_time = time.perf_counter()
            with torch.no_grad():
                results = model(image)
            inference_latency = time.perf_counter() - start_time
            # use ema to update latency
            self.model_latency[self.current_model_index] = self.ema_alpha * inference_latency + (1 - self.ema_alpha) * self.model_latency[self.current_model_index]
            boxes, scores, labels = self.process_results(results)
        
        # start a new thread to update stats
        update_stats_thread = threading.Thread(target=self.prepare_update_stats, args=(image, boxes, scores, labels, inference_latency))
        update_stats_thread.start()
        
        return boxes, scores, labels
    
    def prepare_update_stats(self, image: np.ndarray, boxes, scores, labels, inference_latency):
        '''
        Prepare the stats for updating.
        '''
        super().prepare_update_stats(image, boxes, scores, labels, inference_latency)

        
    def process_results(self, results):
        '''
        Extract from yolo detection results three np lists: boxes, scores, labels
        '''
        boxes = results.xyxy[0][:, :4].cpu().numpy().tolist()
        scores = results.xyxy[0][:, 4].cpu().numpy().tolist()
        labels = results.xyxy[0][:, 5].cpu().numpy().tolist()

        return boxes, scores, labels
