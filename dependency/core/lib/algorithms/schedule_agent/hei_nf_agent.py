import abc
import os.path
import threading
import time
import numpy as np

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context
from core.lib.estimation import AccEstimator
from core.lib.common import VideoOps

from .base_agent import BaseAgent

__all__ = ('HEINFAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='hei_nf')
class HEINFAgent(BaseAgent, abc.ABC):

    def __init__(self, system,
                 agent_id: int,
                 window_size: int = 10,
                 mode: str = 'inference'):
        from .hei_nf import NegativeFeedback_Single

        self.agent_id = agent_id
        self.system = system

        drl_params = system.drl_params.copy()
        hyper_params = system.hyper_params.copy()
        drl_params['state_dims'] = [drl_params['state_dims'], window_size]

        self.window_size = window_size
        self.mode = mode

        self.nf_agent = NegativeFeedback_Single(system, agent_id)

        self.drl_schedule_interval = hyper_params['drl_schedule_interval']
        self.nf_schedule_interval = hyper_params['nf_schedule_interval']

        self.state_dim = drl_params['state_dims']
        self.action_dim = drl_params['action_dim']

        self.gt_file_path = Context.get_file_path('gt_file.txt')
        self.hash_file_path = Context.get_file_path('hash_file.ann')
        self.acc_estimator = AccEstimator(self.hash_file_path, self.gt_file_path)

        self.model_dir = Context.get_file_path(os.path.join(hyper_params['model_dir'], f'agent_{self.agent_id}'))
        FileOps.create_directory(self.model_dir)

        self.latest_policy = None
        self.latest_task_delay = None
        self.schedule_plan = None

    def update_scenario(self, scenario):
        try:
            task_delay = scenario['delay']

            self.latest_task_delay = task_delay
        except Exception as e:
            LOGGER.warning('Wrong scenario from Distributor!')

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        self.set_latest_policy(policy)

    def update_task(self, task):
        pass

    def set_latest_policy(self, policy):
        self.latest_policy = policy

    def get_schedule_plan(self, info):
        return self.schedule_plan

    def run(self):
        LOGGER.info(f'[NF Inference] (agent {self.agent_id}) Start inference nf agent ..')

        while True:
            time.sleep(self.nf_schedule_interval)

            self.schedule_plan = self.nf_agent(self.latest_policy, self.latest_task_delay)
            LOGGER.debug(f'[NF Update] (agent {self.agent_id}) schedule: {self.schedule_plan}')