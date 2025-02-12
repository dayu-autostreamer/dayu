from abc import ABC, abstractmethod
from typing import List
import numpy as np
import time

class BaseSwitch(ABC):
    @abstractmethod
    def __init__(self, models_num: int, decision_interval: int, 
                 detector_instance: object,
                 *args, **kwargs):
        pass

    @abstractmethod
    def switch_model(self, index: int):
        '''
        Switch the model to the one specified in the arguments.
        '''
        pass

    @abstractmethod
    def get_detector_stats(*args, **kwargs):
        '''
        Get the stats for switch decision.
        '''
        pass
