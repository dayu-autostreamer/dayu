from abc import ABC, abstractmethod
import numpy as np
from .stats_manager import StatsManager, StatsEntry
import cv2
import time

from core.lib.network import get_merge_address, http_request
from core.lib.network import NodeInfo, PortInfo
from core.lib.network import NetworkAPIPath, NetworkAPIMethod
from core.lib.common import Context

class BaseInference(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        '''
        Load all models, do all the necessary initializations.
        Notice that the models should be a sorted list of pareto optimal models, so that the switcher can switch between them.
        '''
        self.stats_manager = StatsManager()
        
    @abstractmethod
    def switch_model(self, index: int):
        '''
        Switch the model to the one specified in the arguments.
        '''
        pass

    @abstractmethod
    def get_models_num(self):
        '''
        Get the number of models.
        '''
        pass

    @abstractmethod
    def get_models_accuracy(self):
        '''
        Get the accuracy of the models.
        Returns a list of floats.
        '''
        pass

    @abstractmethod
    def get_models_latency(self):
        '''
        Get the latency of the models.
        Returns a list of floats.
        '''
        pass

    @abstractmethod
    def infer(self, image: np.ndarray):
        '''
        Do the inference on the image.
        '''
        pass

    @abstractmethod
    def get_current_model_index(self):
        '''
        Get the current model index.
        '''
        pass

    def get_queue(self):
        self.local_device = NodeInfo.get_local_device()
        self.processor_port = Context.get_parameter('GUNICORN_PORT')
        queue_url = get_merge_address('127.0.0.1',
                                        port=self.processor_port,
                                        path=NetworkAPIPath.PROCESSOR_QUEUE_LENGTH) 
        result = http_request(url=queue_url, method=NetworkAPIMethod.PROCESSOR_QUEUE_LENGTH, timeout=5)
        return result

    @abstractmethod
    def prepare_update_stats(self, image: np.ndarray, boxes, scores, labels, inference_latency):
        '''
        Prepare the stats for updating.
        '''
        # prepare the stats entry
        stats_entry = StatsEntry()
        stats_entry.timestamp = time.time()
        stats_entry.queue_length = int(self.get_queue())
        stats_entry.cur_model_index = self.get_current_model_index()
        stats_entry.cur_model_accuracy = self.get_models_accuracy()[stats_entry.cur_model_index]
        stats_entry.processing_latency = inference_latency
        stats_entry.target_nums = len(boxes)
        if len(boxes) > 0:
            stats_entry.avg_confidence = np.mean(scores)
            stats_entry.std_confidence = np.std(scores)
        else:
            stats_entry.avg_confidence = 0
            stats_entry.std_confidence = 0
        # 获取图像尺寸
        image_height, image_width = image.shape[:2]
        image_area = image_height * image_width
        
        # 计算相对尺寸（框面积/图像面积）
        boxes = np.array(boxes)
        if len(boxes) > 0:
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            relative_sizes = np.sqrt(box_areas / image_area)
            stats_entry.avg_size = np.mean(relative_sizes)
            stats_entry.std_size = np.std(relative_sizes)
        else:
            stats_entry.avg_size = 0
            stats_entry.std_size = 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stats_entry.brightness = np.mean(gray_image)
        stats_entry.contrast = np.std(gray_image)
        print(f'Updating stats: {stats_entry}')
        self.stats_manager.update_stats(stats_entry)