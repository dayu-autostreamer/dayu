import abc
import copy

from core.lib.common import ClassFactory, ClassType, Context, LOGGER, TaskConstant
from core.lib.content import Task
from core.lib.estimation import OverheadEstimator

from .base_agent import BaseAgent
from .gecko import GeckoPolicySearch


__all__ = ('GeckoAgent',)


@ClassFactory.register(ClassType.SCH_AGENT, alias='gecko')
class GeckoAgent(BaseAgent, abc.ABC):
    def __init__(self, system, agent_id: int, sch_param: dict, baseline_param: dict):
        super().__init__(system, agent_id)
        self.agent_id = agent_id
        self.cloud_device = system.cloud_device
        self.edge_device = None
        self.cur_resource_table = {}
        self.cur_scenario = {}
        self.cur_policy = {}
        self.cur_task = None

        self.service_names = None
        self.edge_serv_num_list = None
        self.fps_list = system.fps_list
        self.resolution_list = system.resolution_list
        self.buffer_size_list = [value for value in system.buffer_size_list if value > 2]

        self.sch_param = copy.deepcopy(sch_param) if sch_param else {}
        self.baseline_param = copy.deepcopy(baseline_param) if baseline_param else {}
        self.baseline_param['kb_path'] = Context.get_file_path(self.baseline_param['kb_path'])
        self.search_engine = None
        self.overhead_estimator = OverheadEstimator('gecko', 'scheduler/gecko', agent_id=agent_id)

    def run(self):
        pass

    def update_scenario(self, scenario):
        self.cur_scenario = scenario

    def update_resource(self, device, resource):
        self.cur_resource_table[device] = resource

    def update_policy(self, policy):
        self.cur_policy = policy

    def update_task(self, task: Task):
        if task is None:
            LOGGER.debug('[Gecko] New task is None.')
            return

        self.cur_task = copy.deepcopy(task)
        if self.search_engine is None:
            return

        self.search_engine.update_feedback(
            context_info=self.get_context_info_from_task(self.cur_task),
            conf_info=self.get_conf_info_from_task(self.cur_task),
            task_info=self.get_task_info_from_task(self.cur_task),
        )

    def get_schedule_plan(self, info):
        self.ensure_runtime_shape(info)
        self.ensure_search_engine(info)

        task = copy.deepcopy(self.cur_task)
        task_id = task.get_task_id() if task is not None else None
        cur_policy = self.get_conf_info_from_task(task) if task is not None else None
        context_info = self.get_context_info_from_task(task) if task is not None else None

        with self.overhead_estimator:
            new_policy = self.search_engine.get_schedule_plan(
                cur_task_id=task_id,
                cur_policy=cur_policy,
                context_info=context_info,
            )
            new_dag = self.build_dag_with_policy(info['dag'], new_policy['edge_serv_num'])
            return {
                'fps': new_policy['fps'],
                'resolution': new_policy['resolution'],
                'buffer_size': new_policy['buffer_size'],
                'dag': new_dag,
                'encoding': self.get_encoding(info),
            }

    def get_schedule_overhead(self):
        return self.overhead_estimator.get_latest_overhead()

    def ensure_runtime_shape(self, info):
        if self.edge_device is None:
            self.edge_device = info['source_device']

        if self.edge_serv_num_list is not None and self.service_names is not None:
            return

        pipeline_dict = Task.extract_pipeline_deployment_from_dag_deployment(info['dag'])
        self.service_names = [
            service_info['service_name']
            for service_info in pipeline_dict
            if service_info['service_name'] not in (TaskConstant.START.value, TaskConstant.END.value)
        ]
        self.edge_serv_num_list = list(range(0, len(self.service_names) + 1))

    def ensure_search_engine(self, info):
        if self.search_engine is not None:
            return

        delay_cons = self.baseline_param['delay_cons'] * self.baseline_param.get('delay_cons_adjust', 1.0)
        acc_cons = self.baseline_param['acc_cons'] * self.baseline_param.get('acc_cons_adjust', 1.0)
        self.search_engine = GeckoPolicySearch(
            kb_path=self.baseline_param['kb_path'],
            service_name_pipeline=self.service_names,
            knob_value_range_dict={
                'fps': self.fps_list,
                'resolution': self.resolution_list,
                'buffer_size': self.buffer_size_list,
                'edge_serv_num': self.edge_serv_num_list,
            },
            delay_cons=delay_cons,
            acc_cons=acc_cons,
            delay_weight=self.baseline_param['delay_weight'],
            acc_weight=self.baseline_param['acc_weight'],
            default_policy=self.baseline_param['default_policy'],
            raw_meta_data=info.get('meta_data', {}),
            corrector_param=self.baseline_param['corrector_param'],
            queue_param=self.baseline_param['queue_param'],
            search_param=self.baseline_param.get('search_param', {}),
            gecko_param=self.baseline_param.get('gecko_param', {}),
            use_corrected_prediction=self.baseline_param.get('use_corrected_prediction', False),
            enable_feedback_update=self.baseline_param.get('enable_feedback_update', False),
        )

    def get_context_info_from_task(self, cur_task: Task):
        if cur_task is None:
            return None

        context_info = {}
        if self.edge_device in self.cur_resource_table:
            edge_resource = self.cur_resource_table[self.edge_device]
            if 'available_bandwidth' in edge_resource:
                context_info['band_Mbps'] = edge_resource['available_bandwidth']

        scenario_data = copy.deepcopy(cur_task.get_first_scenario_data() or {})
        tmp_data = copy.deepcopy(cur_task.get_tmp_data() or {})
        if 'file_size' in tmp_data:
            scenario_data['file_size'] = tmp_data['file_size']

        if 'obj_size' in scenario_data and 'obj_num' in scenario_data and 'obj_velocity' in scenario_data:
            context_info['obj_size_norm'] = self.mean_or_zero(scenario_data['obj_size'])
            context_info['obj_num'] = self.mean_or_zero(scenario_data['obj_num'])
            context_info['obj_speed'] = self.mean_or_zero(scenario_data['obj_velocity'])

        required_context = {'band_Mbps', 'obj_size_norm', 'obj_num', 'obj_speed'}
        if required_context.issubset(context_info):
            return context_info
        return None

    def get_conf_info_from_task(self, cur_task: Task):
        if cur_task is None:
            return None

        metadata = cur_task.get_metadata()
        pipeline_dict = cur_task.get_pipeline_deployment_info()
        return {
            'resolution': metadata['resolution'],
            'fps': metadata['fps'],
            'buffer_size': metadata['buffer_size'],
            'edge_serv_num': self.trans_pipeline_dict_to_edge_serv_num(pipeline_dict),
        }

    def get_task_info_from_task(self, cur_task: Task):
        if cur_task is None:
            return None

        task_info = {'task_id': cur_task.get_task_id()}
        pipeline_dict = cur_task.get_pipeline_deployment_info()
        metadata = cur_task.get_metadata()
        dag = cur_task.get_dag()
        service_num = 0
        found_partition = False
        edge_cloud_trans_delay = 0

        for service_info in pipeline_dict:
            service_name = service_info['service_name']
            if service_name in (TaskConstant.START.value, TaskConstant.END.value):
                continue

            service_num += 1
            service = dag.get_node(service_name).service
            execute_delay = service.get_real_execute_time() / metadata['buffer_size']
            wait_delay = (service.get_execute_time() / metadata['buffer_size']) - execute_delay

            if service_num == 1:
                task_info['real_exe_detect'] = execute_delay
                task_info['detect_wait_delay'] = wait_delay
            elif service_num == 2:
                task_info['real_exe_classify'] = execute_delay
                task_info['classify_wait_delay'] = wait_delay

            if service.get_execute_device() == self.cloud_device and not found_partition:
                edge_cloud_trans_delay = service.get_transmit_time() / metadata['buffer_size']
                found_partition = True

        task_info['real_trans'] = edge_cloud_trans_delay
        return task_info

    def build_dag_with_policy(self, dag, edge_serv_num):
        pipeline_dict = Task.extract_pipeline_deployment_from_dag_deployment(copy.deepcopy(dag))
        new_pipeline_dict = self.trans_edge_serv_num_to_pipeline_dict(
            edge_serv_num=edge_serv_num,
            pipeline_dict=pipeline_dict,
        )
        return Task.extract_dag_deployment_from_pipeline_deployment(new_pipeline_dict)

    def trans_edge_serv_num_to_pipeline_dict(self, edge_serv_num, pipeline_dict):
        edge_count = int(edge_serv_num)
        if pipeline_dict and pipeline_dict[0]['service_name'] == TaskConstant.START.value:
            edge_count += 1

        edge_part = [{**item, 'execute_device': self.edge_device} for item in pipeline_dict[:edge_count]]
        cloud_part = [{**item, 'execute_device': self.cloud_device} for item in pipeline_dict[edge_count:]]
        return edge_part + cloud_part

    def trans_pipeline_dict_to_edge_serv_num(self, pipeline_dict):
        edge_serv_num = 0
        for service_info in pipeline_dict:
            if service_info['service_name'] in (TaskConstant.START.value, TaskConstant.END.value):
                continue
            if service_info['execute_device'] == self.cloud_device:
                break
            edge_serv_num += 1
        return edge_serv_num

    @staticmethod
    def get_encoding(info):
        return (info.get('meta_data') or {}).get('encoding', 'mp4v')

    @staticmethod
    def mean_or_zero(value):
        if value is None:
            return 0
        if isinstance(value, (list, tuple)):
            if not value:
                return 0
            return sum(value) / len(value)
        return value
