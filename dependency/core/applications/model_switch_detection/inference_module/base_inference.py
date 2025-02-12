from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseInference(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        '''
        Load all models, do all the necessary initializations.
        Notice that the models should be a sorted list of pareto optimal models, so that the switcher can switch between them.
        '''

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
    def infer(self, image: np.ndarray):
        '''
        Do the inference on the image.
        '''
        pass