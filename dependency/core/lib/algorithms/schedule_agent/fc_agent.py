import abc

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('FCAgent',)

"""
Feedback Controlling Agent Class

Implementation of the Feedback Controlling

Nigade V, Winder R, Bal H, et al. Better never than late: Timely edge video analytics over the air[C]//Proceedings of the 19th ACM Conference on Embedded Networked Sensor Systems. 2021: 426-432.
"""


@ClassFactory.register(ClassType.SCH_AGENT, alias='fc')
class FCAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int,
                 reference_policy: dict,
                 window_length: int = 20,
                 delay_constraint: float = 0.2,
                 theta: float = 3):
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device

        self.reference_policy = reference_policy

        self.resolution_list = system.resolution_list.copy()

        self.latest_resolution_index = self.resolution_list.index(reference_policy['resolution'])

        self.history_window = []

        self.window_length = window_length
        self.delay_constraint = delay_constraint

        self.a = 1
        self.b = 2

        self.theta_high = 0
        self.theta_low = 0 - theta / 1000 * self.window_length

        self.overhead_estimator = OverheadEstimator('FC', 'scheduler/fc')

    def get_schedule_plan(self, info):
        if len(self.history_window) < self.window_length:
            LOGGER.info('[FC adjust] not enough history window')
            return None

        policy = self.reference_policy.copy()

        with self.overhead_estimator:
            updated_resolution_index = self.feed_back_control()

        policy.update({'resolution': self.resolution_list[updated_resolution_index]})
        cloud_device = self.cloud_device
        pipeline = info['pipeline']
        pipeline = [{**p, 'execute_device': cloud_device} for p in pipeline]

        policy.update({'pipeline': pipeline})
        return policy

    def calculate_delay_error(self, delay):
        term_deviation = abs(self.delay_constraint - delay)

        if delay <= self.delay_constraint:
            error = - term_deviation ** self.a
        else:
            error = term_deviation ** self.b

        return error

    def feed_back_control(self):

        avg_error = sum(self.calculate_delay_error(delay) for delay in self.history_window) / len(self.history_window)
        LOGGER.debug(f'avg_error:{avg_error}  theta_high:{self.theta_high}  theta_low:{self.theta_low}')
        if avg_error <= self.theta_low:
            if self.latest_resolution_index < len(self.resolution_list) - 1:
                update_resolution_index = self.latest_resolution_index + 1
            else:
                update_resolution_index = self.latest_resolution_index
        elif avg_error >= self.theta_high:
            update_resolution_index = self.latest_resolution_index // 2

        else:
            update_resolution_index = self.latest_resolution_index

        LOGGER.info(f'[FC adjust] resolution update: {self.resolution_list[self.latest_resolution_index]}'
                    f' -> {self.resolution_list[update_resolution_index]}')
        self.latest_resolution_index = update_resolution_index
        return self.latest_resolution_index

    def run(self):
        pass

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def update_task(self, task):
        if len(self.history_window) >= self.window_length:
            self.history_window = self.history_window[1:]
        self.history_window.append(task.calculate_total_time() / task.get_metadata()['buffer_size'])
