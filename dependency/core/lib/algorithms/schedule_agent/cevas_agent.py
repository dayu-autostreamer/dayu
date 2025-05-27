import abc
import time
import numpy as np

from core.lib.common import ClassFactory, ClassType, Context, LOGGER
from core.lib.content import Task
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent

__all__ = ('CEVASAgent',)

"""
CEVAS Agent Class

Implementation of CEVAS

In this implementation, we make decision on one pipeline

Zhang M, Wang F, Zhu Y, et al. Towards cloud-edge collaborative online video analytics with fine-grained serverless pipelines[C]//Proceedings of the 12th ACM multimedia systems conference. 2021: 80-93.
"""


@ClassFactory.register(ClassType.SCH_AGENT, alias='cevas')
class CEVASAgent(BaseAgent, abc.ABC):

    def __init__(self, system, agent_id: int, fixed_policy: dict = None, time_slot: int = 3):
        super().__init__()

        import torch
        from .cevas.mlp import MLP
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.fixed_policy = fixed_policy

        self.pipe_seg = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logic_node_num = 1
        model_path = Context.get_file_path('model.pt')
        self.model = MLP(logic_node_num=logic_node_num).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.overhead_estimator = OverheadEstimator('CEVAS', 'scheduler/cevas')

        self.data_time_sequence = []
        self.time_index = 0
        self.time_slot = time_slot

    def get_schedule_plan(self, info):
        if self.fixed_policy is None:
            return self.fixed_policy

        policy = self.fixed_policy.copy()
        edge_device = info['device']
        cloud_device = self.cloud_device
        pipe_seg = self.pipe_seg
        pipeline = Task.extract_pipeline_deployment_from_dag_deployment(info['dag'])
        pipeline = [{**p, 'execute_device': edge_device} for p in pipeline[:pipe_seg]] + \
                   [{**p, 'execute_device': cloud_device} for p in pipeline[pipe_seg:]]

        policy.update({'dag': Task.extract_dag_deployment_from_pipeline_deployment(pipeline)})
        return policy

    # 最优化目标
    def optimize_target(self, strategy):
        a = 0.1
        b = 0.2
        return a * strategy[0] + b * strategy[1]

    # 获取流水线每个逻辑节点在t+1时隙边缘节点上的CPU/内存/输入数据大小/云端执行成本
    def get_pipeline_cpu_memory(self, index, time_slot=3):
        import torch
        # {
        #       edge_CPU:
        #       edge_memory:
        #       node_data:
        #       cloud_cost:
        # }

        # 需要输入节点个数

        data = []

        if index <= time_slot:
            # 冷启动
            for i in range(0, time_slot):
                data.append([self.data_time_sequence[index][0], self.data_time_sequence[index][1]])
        else:
            for i in range(index - time_slot, index):
                data.append([self.data_time_sequence[i][0], self.data_time_sequence[i][1]])
        # 展平
        previous_data = torch.tensor([item for sublist in data for item in sublist]).float().to(self.device)
        res = self.model.forward(previous_data)
        return res

    def run(self):
        while True:
            time.sleep(1)
            if len(self.data_time_sequence) == 0:
                LOGGER.info('[No Sequence data] Waiting...')
                continue

            with self.overhead_estimator:
                # 在t时隙获得t+1时隙单个流水线的最佳分割点
                # 获得t+1时隙相关输入信息
                schedule_info = self.get_pipeline_cpu_memory(self.time_index, self.time_slot)
                LOGGER.debug(f'[CEVAS schedule info] schedule info: {schedule_info}')

                # 信息顺序为 边缘节点CPU限制 / 内存限制 / 每个节点输入数据量大小 / 云开销
                # 解空间比较小,遍历获得结果即可
                # 遍历分割点
                # 优化目标 target=min (a * P * x + b * D * x)
                target = 0
                target_idx = -1
                P = schedule_info[3]
                D = schedule_info[2]
                edge_strategy = [D, 0]
                cloud_strategy = [0, P]

                target1 = self.optimize_target(edge_strategy)
                target2 = self.optimize_target(cloud_strategy)

                if target1 > target2:
                    target_idx = 0
                else:
                    target_idx = 1

            if target_idx != -1:
                raw_seg = self.pipe_seg
                self.pipe_seg = target_idx
                LOGGER.info(f'[CEVAS Update] update pipeline segment: {raw_seg} -> {self.pipe_seg}')
            else:
                LOGGER.warning('CEVAS comouting error!')

    def update_scenario(self, scenario):
        pass

    def update_resource(self, device, resource):
        pass

    def update_policy(self, policy):
        pass

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()

    def update_task(self, task):
        avg_num = np.mean(task.get_scenario_data()['obj_num'])
        file_size = task.get_tmp_data()['file_size']
        self.data_time_sequence.append([avg_num, file_size])
