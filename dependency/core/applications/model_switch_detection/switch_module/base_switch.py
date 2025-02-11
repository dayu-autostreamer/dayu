from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseSwitch(ABC):
    @abstractmethod
    def __init__(self, models_num: int, decision_interval: int, *args, **kwargs):
        pass

    @abstractmethod
    def switch_model(self, index: int):
        '''
        Switch the model to the one specified in the arguments.
        '''
        pass
