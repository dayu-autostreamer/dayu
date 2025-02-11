from .base_switch import BaseSwitch
from typing import List
import time
import threading
import numpy as np

class RandomSwitch(BaseSwitch):
    def __init__(self, models_num: int, decision_interval: int, 
                 detector_instance: object,
                 *args, **kwargs):
        '''
        Initialize the random switcher.
        '''
        self.models_num = models_num
        self.decision_interval = decision_interval
        self.detector_instance = detector_instance
        self.last_switch_time = time.time()
        self.switch_thread = threading.Thread(target=self._switch_loop)
        self.switch_thread.start()

    def _switch_loop(self):
        while True:
            if time.time() - self.last_switch_time > self.decision_interval:
                model_index = np.random.randint(0, self.models_num)
                self.switch_model(model_index)
                print(f'Random switched model to {model_index}')
                self.last_switch_time = time.time()


    def switch_model(self, index: int):
        '''
        Switch the model to the one specified in the arguments.
        '''
        self.detector_instance.switch_model(index)
        