import numpy as np
from core.lib.common import LOGGER


class StateBuffer:
    def __init__(self, window_size):
        self.resources = []
        self.delay = []
        self.buffer_size = []
        self.segment_size = []
        self.content_dynamics = []
        self.decisions = []
        self.tasks = []

        self.window_size = window_size
        self.max_size = window_size * 2

    def add_resource_buffer(self, resource):
        self.resources.append(resource)
        while len(self.resources) > self.max_size:
            self.resources.pop(0)

    def add_scenario_buffer(self, scenario):
        self.delay.append(scenario['delay'])
        self.buffer_size.append(scenario['buffer_size'])
        self.segment_size.append(scenario['segment_size'])
        self.content_dynamics.append(scenario['content_dynamics'])
        while len(self.delay) > self.max_size:
            self.delay.pop(0)
            self.buffer_size.pop(0)
            self.segment_size.pop(0)
            self.content_dynamics.pop(0)

    def add_decision_buffer(self, decision):
        self.decisions.append(decision)
        while len(self.decisions) > self.max_size:
            self.decisions.pop(0)

    def add_task_buffer(self, task):
        self.tasks.append(task)

    def get_state_buffer(self):

        # TODO: normalization the state ?
        resources = self.resources.copy()

        delay = self.delay.copy()
        buffer_size = self.buffer_size.copy()
        segment_size = self.segment_size.copy()
        content_dynamics = self.content_dynamics.copy()
        decisions = self.decisions.copy()
        tasks = self.tasks.copy()

        if len(tasks) == 0:
            evaluation_info = None
        else:
            evaluation_info = tasks

        if len(resources) == 0 or len(delay) == 0 or len(decisions) == 0:
            state = None
        else:

            LOGGER.debug(f'[Resource Buffer] length: {len(resources)}, content: {resources}')
            LOGGER.debug(f'[Scenario Buffer] length: {len(delay)}')
            LOGGER.debug(f'[Decision Buffer] length: {len(decisions)}, content: {decisions}')

            resources = np.array(self.resample_buffer(resources, self.window_size))
            delay = np.array(self.resample_buffer(delay, self.window_size))
            buffer_size = np.array(self.resample_buffer(buffer_size, self.window_size))
            segment_size = np.array(self.resample_buffer(segment_size, self.window_size))
            content_dynamics = np.array(self.resample_buffer(content_dynamics, self.window_size))
            decisions = np.array(self.resample_buffer(decisions, self.window_size))

            LOGGER.debug(f'[Resample Resource Buffer] length: {len(resources)}, content: {resources}')
            LOGGER.debug(f'[Resample Scenario Buffer] length: {len(delay)}')
            LOGGER.debug(f'[Resample Decision Buffer] length: {len(decisions)}, content: {decisions}')

            state = np.vstack((resources.T, delay.T, buffer_size.T, segment_size.T, content_dynamics.T, decisions.T))
            LOGGER.debug(f'[State Buffer] content: {state}')

        self.clear_state_buffer()
        return state, evaluation_info

    def clear_state_buffer(self):
        # self.resources.clear()
        # self.scenarios.clear()
        # self.decisions.clear()
        self.tasks.clear()

    @staticmethod
    def resample_buffer(buffer, size):
        buffer_length = len(buffer)
        assert buffer_length != 0, 'Resample buffer size is 0!'

        if buffer_length > size:
            indices = np.linspace(0, buffer_length - 1, num=size, dtype=int)
            resized_buffer = [buffer[idx] for idx in indices]
        elif buffer_length < size:
            resized_buffer = []
            repeat_factor = size // buffer_length
            extra_slots = size % buffer_length
            for i in range(buffer_length):
                resized_buffer.extend([buffer[i]] * repeat_factor)
                if extra_slots > 0:
                    resized_buffer.append(buffer[i])
                    extra_slots -= 1
        else:
            resized_buffer = buffer[:]
        return resized_buffer
