import abc
import math
import os.path
import time
import numpy as np

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context, VideoOps, TaskConstant
from core.lib.estimation import AccEstimator, OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('CASVAAgent',)

"""
CASVA Agent Class

Implementation of CASVA

In this implementation , for the sake of uniformity, we also set buffer size in state as a sequence not a single value.

Zhang M, Wang F, Liu J. Casva: Configuration-adaptive streaming for live video analytics[C]//IEEE INFOCOM 2022-IEEE Conference on Computer Communications. IEEE, 2022: 2168-2177.
"""


@ClassFactory.register(ClassType.SCH_AGENT, alias='casva')
class CASVAAgent(BaseAgent, abc.ABC):

    def __init__(self, system,
                 agent_id: int,
                 window_size: int = 8,
                 mode: str = 'inference',
                 streaming_mode: str = 'latency_first',
                 segment_length: int = 2,
                 model_dir: str = 'model',
                 load_model: bool = False,
                 load_model_episode: int = 0,
                 acc_gt_dir: str = ''):
        super().__init__(system, agent_id)
        from .casva import DualClippedPPO, RandomBuffer, Adapter, StateBuffer

        assert streaming_mode in ['latency_first', 'delivery_first'], \
            '"streaming_mode" must be "latency_first" or "delivery_first"'

        self.agent_id = agent_id
        self.system = system

        self.cloud_device = system.cloud_device
        self.edge_device = None

        self.fps_list = system.fps_list
        self.resolution_list = system.resolution_list
        self.qp_list = system.qp_list

        drl_params = system.drl_params.copy()
        hyper_params = system.hyper_params.copy()
        drl_params['state_dims'] = [drl_params['state_dims'], window_size]

        self.window_size = window_size
        self.state_buffer = StateBuffer(self.window_size)
        self.mode = mode
        self.streaming_mode = streaming_mode
        self.segment_length = segment_length

        self.drl_agent = DualClippedPPO(**drl_params)
        self.replay_buffer = RandomBuffer(**drl_params)
        self.adapter = Adapter

        self.drl_schedule_interval = hyper_params['drl_schedule_interval']

        self.state_dim = drl_params['state_dims']
        self.action_dim = drl_params['action_dim']

        self.model_dir = Context.get_file_path(os.path.join('scheduler/casva', model_dir, f'agent_{self.agent_id}'))
        FileOps.create_directory(self.model_dir)
        if load_model:
            self.drl_agent.load(self.model_dir, load_model_episode)

        self.total_steps = hyper_params['drl_total_steps']
        self.save_interval = hyper_params['drl_save_interval']
        self.update_interval = hyper_params['drl_update_interval']
        self.update_after = hyper_params['drl_update_after']

        self.latest_policy = None
        self.schedule_plan = None

        self.acc_gt_dir = acc_gt_dir
        self.acc_estimator = None

        self.past_buffer_size_value = 0
        self.latest_skip_count = 0

        self.overhead_estimator = OverheadEstimator('CASVA', 'scheduler/casva', agent_id=self.agent_id)

        self.reward_file = Context.get_file_path(os.path.join('scheduler/casva', 'reward.txt'))
        FileOps.remove_file(self.reward_file)

    def get_drl_state_buffer(self):
        while True:
            state, evaluation_info = self.state_buffer.get_state_buffer()
            if state is not None and evaluation_info is not None:
                return state, evaluation_info
            LOGGER.info(f'[Wait for State] (agent {self.agent_id}) State empty, '
                        f'wait for resource state or scenario state ..')
            time.sleep(1)

    def map_drl_action_to_decision(self, action):
        """
        map [-1, 1] to {-1, 0, 1}
        """
        if self.latest_policy is None:
            LOGGER.info(f'[CASVA Lack Latest Policy] (agent {self.agent_id}) No latest policy, none decision make ..')
            return

        self.edge_device = self.latest_policy['edge_device']
        resolution_index = min(int((action[0] + 1) / 2 * len(self.resolution_list)), len(self.resolution_list) - 1)
        fps_index = min(int((action[1] + 1) / 2 * len(self.fps_list)), len(self.fps_list) - 1)
        qp_index = min(int((action[2] + 1) / 2 * len(self.qp_list)), len(self.qp_list) - 1)

        self.latest_policy.update({'resolution': self.resolution_list[resolution_index],
                                   'fps': self.fps_list[fps_index],
                                   'qp': self.qp_list[qp_index],
                                   'buffer_size': math.ceil(self.fps_list[fps_index] * self.segment_length)
                                   })

        dag = self.latest_policy['dag']
        for service_name in dag:
            dag[service_name]['service']['execute_device'] = self.cloud_device

        self.latest_policy.update({'dag': dag})
        self.schedule_plan = self.latest_policy.copy()

        LOGGER.info(f'[CASVA Decision] (agent {self.agent_id}) Action: {action}   '
                    f'Decision: "resolution":{self.resolution_list[resolution_index]}, "fps":{self.fps_list[fps_index]}')

    def reset_drl_env(self):
        self.intermediate_decision = [0 for _ in range(self.action_dim)]

        state, _ = self.get_drl_state_buffer()

        return state

    def step_drl_env(self, action):

        self.map_drl_action_to_decision(action)

        time.sleep(self.drl_schedule_interval)

        state, evaluation_info = self.get_drl_state_buffer()
        reward = self.calculate_drl_reward(evaluation_info)
        done = False
        info = ''

        return state, reward, done, info

    def create_acc_estimator(self, service_name: str):
        gt_path_prefix = os.path.join(self.acc_gt_dir, service_name)
        gt_file_path = Context.get_file_path(os.path.join(gt_path_prefix, 'gt_file.txt'))
        self.acc_estimator = AccEstimator(gt_file_path)

    def calculate_drl_reward(self, evaluation_info):

        acc_list = []
        transmit_delay_list = []
        buffer_size_list = []

        for task in evaluation_info:
            meta_data = task.get_metadata()
            raw_metadata = task.get_raw_metadata()
            content = task.get_first_content()
            dag = task.get_dag()

            hash_data = task.get_hash_data()

            buffer_size = meta_data['buffer_size']
            buffer_size_list.append(buffer_size)

            raw_resolution = VideoOps.text2resolution(raw_metadata['resolution'])
            resolution = VideoOps.text2resolution(meta_data['resolution'])
            resolution_ratio = (resolution[0] / raw_resolution[0], resolution[1] / raw_resolution[1])

            fps_ratio = meta_data['fps'] / raw_metadata['fps']

            transmit_delay_list.append(task.calculate_cloud_edge_transmit_time())

            if not self.acc_estimator:
                self.create_acc_estimator(service_name=dag.get_next_nodes(TaskConstant.START.value)[0])
            acc = self.acc_estimator.calculate_accuracy(hash_data, content, resolution_ratio, fps_ratio)
            acc_list.append(acc)

        final_acc = np.mean(acc_list)
        final_transmit_delay = np.mean(transmit_delay_list)
        final_buffer_size = np.mean(buffer_size_list)

        if self.streaming_mode == 'latency_first':
            reward = (5 * final_acc - 1 * max(final_transmit_delay - self.segment_length, 0) / self.segment_length
                      - 1 * self.latest_skip_count)

            LOGGER.info(f'[CASVA Reward Computing] (latency first) '
                        f'acc:{final_acc:.4f}, delay:{final_transmit_delay:.4f} ,reward:{reward:.4f}')
        elif self.streaming_mode == 'delivery_first':
            reward = (2 * final_acc - 3 * max(final_transmit_delay - self.segment_length, 0) / self.segment_length
                      + ((final_buffer_size - self.past_buffer_size_value) / self.segment_length
                         if final_buffer_size < self.past_buffer_size_value else 0))

            self.past_buffer_size_value = final_buffer_size

            LOGGER.info(f'[CASVA Reward Computing] (delivery first) '
                        f'acc:{final_acc:.4f}, delay:{final_transmit_delay:.4f} ,reward:{reward:.4f}')
        else:
            raise ValueError('"streaming_mode" must be "latency_first" or "delivery_first"')

        with open(self.reward_file, 'a') as f:
            f.write(f'dacc:{final_acc:.4f}, delay:{final_transmit_delay:.4f} ,reward:{reward:.4f}\n')

        return reward

    def train_drl_agent(self):
        LOGGER.info(f'[CASVA DRL Train] (agent {self.agent_id}) Start train drl agent ..')
        state = self.reset_drl_env()
        for step in range(self.total_steps):
            with self.overhead_estimator:
                action = self.drl_agent.select_action(state, deterministic=False, with_logprob=False)

            next_state, reward, done, info = self.step_drl_env(action)
            done = self.adapter.done_adapter(done, step)
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            LOGGER.info(f'[CASVA DRL Train Data] (agent {self.agent_id}) Step:{step}  Reward:{reward}')

            if step >= self.update_after and step % self.update_interval == 0:
                for _ in range(self.update_interval):
                    LOGGER.info(f'[CASVA DRL Train] (agent {self.agent_id}) Train drl agent with replay buffer')
                    self.drl_agent.train(self.replay_buffer)

            if step % self.save_interval == 0:
                self.drl_agent.save(self.model_dir, step)

            if done:
                state = self.reset_drl_env()

        LOGGER.info(f'[CASVA DRL Train] (agent {self.agent_id}) End train drl agent ..')

    def inference_drl_agent(self):
        LOGGER.info(f'[CASVA DRL Inference] (agent {self.agent_id}) Start inference drl agent ..')
        state = self.reset_drl_env()
        cur_step = 0
        while True:
            time.sleep(self.drl_schedule_interval)
            cur_step += 1

            with self.overhead_estimator:
                action = self.drl_agent.select_action(state, deterministic=False, with_logprob=False)

            next_state, reward, done, info = self.step_drl_env(action)
            done = self.adapter.done_adapter(done, cur_step)
            state = next_state
            if done:
                state = self.reset_drl_env()
                cur_step = 0

    def update_scenario(self, scenario):
        try:
            task_delay = scenario['delay']
            buffer_size = scenario["buffer_size"]
            segment_size = scenario["segment_size"]
            content_dynamics = scenario["content_dynamics"]

            self.state_buffer.add_scenario_buffer(
                {'delay': task_delay,
                 'buffer_size': buffer_size,
                 'segment_size': segment_size,
                 'content_dynamics': content_dynamics})

        except Exception as e:
            LOGGER.warning(f'Wrong scenario from Distributor: {str(e)}')
            raise e

    def update_resource(self, device, resource):
        bandwidth = resource['available_bandwidth']
        if bandwidth != -1:
            self.state_buffer.add_resource_buffer([bandwidth])

    def update_policy(self, policy):
        self.set_latest_policy(policy)

        resolution_decision = self.system.resolution_list.index(policy['resolution'])
        fps_decision = self.system.fps_list.index(policy['fps'])
        qp_decision = self.system.qp_list.index(policy['qp'])

        self.state_buffer.add_decision_buffer([resolution_decision, fps_decision,
                                               qp_decision])

    def update_task(self, task):
        self.state_buffer.add_task_buffer(task)

    def set_latest_policy(self, policy):
        self.latest_policy = policy

    def get_schedule_plan(self, info):
        self.latest_skip_count = info['skip_count']
        return self.schedule_plan

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()

    def run(self):

        if self.mode == 'train':
            self.train_drl_agent()
        elif self.mode == 'inference':
            self.inference_drl_agent()
        else:
            assert None, f'(agent {self.agent_id}) Invalid execution mode: {self.mode}, ' \
                         f'only support ["train", "inference"]'
