from abc import ABC, abstractmethod

class BaseSwitch(ABC):
    @abstractmethod
    def __init__(self, decision_interval: int, 
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
