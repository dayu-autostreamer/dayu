from abc import ABC, abstractmethod
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