import json

from core.lib.common import Context, LOGGER, SystemConstant
from core.lib.content import Task
from core.lib.network import merge_address
from core.lib.network import NodeInfo, PortInfo
from core.lib.network import NetworkAPIPath, NetworkAPIMethod
from core.lib.network import http_request
from core.lib.estimation import TimeEstimator


class Generator:
    def __init__(self, source_id: int, metadata: dict, task_dag: dict, ):
        """ Initialize the generator."""

        """task base information"""
        # source_id points to the corresponding source and is unique for generator
        self.source_id = source_id
        # task_dag contains offloading decisions
        self.task_dag = Task.extract_dag_from_dict(task_dag)
        # service_deployment contains service deployment situations in system
        self.service_deployment = None
        # raw_meta_data contains meta configuration of source
        self.raw_meta_data = metadata.copy()
        # meta_data contains data configuration decisions
        self.meta_data = metadata.copy()

        """distributed devices information"""
        self.local_device = NodeInfo.get_local_device()
        self.all_edge_devices = Context.get_parameter('ALL_EDGE_DEVICES', direct=False)
        self.task_dag = Task.set_execute_device(self.task_dag, self.local_device)

        """network communication base information"""
        self.scheduler_hostname = NodeInfo.get_cloud_node()
        self.scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.schedule_address = merge_address(NodeInfo.hostname2ip(self.scheduler_hostname),
                                              port=self.scheduler_port, path=NetworkAPIPath.SCHEDULER_SCHEDULE)

        """hook functions"""
        self.before_schedule_operation = Context.get_algorithm('GEN_BSO')
        self.after_schedule_operation = Context.get_algorithm('GEN_ASO')
        self.data_getter = Context.get_algorithm('GEN_GETTER')
        self.before_submit_task_operation = Context.get_algorithm('GEN_BSTO')

    def request_schedule_policy(self):
        params = self.before_schedule_operation(self)
        response = http_request(url=self.schedule_address,
                                method=NetworkAPIMethod.SCHEDULER_SCHEDULE,
                                data={'data': json.dumps(params)})
        self.after_schedule_operation(self, response)

    @staticmethod
    def record_total_start_ts(cur_task: Task):
        TimeEstimator.record_task_ts(cur_task,
                                     'total_start_time',
                                     is_end=False)

    @staticmethod
    def record_transmit_start_ts(cur_task: Task):
        TimeEstimator.record_dag_ts(cur_task,
                                    is_end=False,
                                    sub_tag='transmit')

    def generate_task(self, task_id, task_dag, service_deployment, meta_data, compressed_path, hash_codes):
        """generate a new task"""
        return Task(source_id=self.source_id,
                    task_id=task_id,
                    source_device=self.local_device,
                    all_edge_devices=self.all_edge_devices,
                    dag=task_dag,
                    deployment=service_deployment,
                    metadata=meta_data,
                    raw_metadata=self.raw_meta_data,
                    hash_data=hash_codes,
                    file_path=compressed_path)

    def submit_task_to_controller(self, cur_task):
        assert cur_task, 'Task is empty when submit to controller!'

        self.before_submit_task_operation(self, cur_task)

        dst_device = cur_task.get_current_stage_device()
        controller_ip = NodeInfo.hostname2ip(dst_device)
        controller_address = merge_address(controller_ip,
                                           port=self.controller_port,
                                           path=NetworkAPIPath.CONTROLLER_TASK)
        self.record_transmit_start_ts(cur_task)
        http_request(url=controller_address,
                     method=NetworkAPIMethod.CONTROLLER_TASK,
                     data={'data': cur_task.serialize()},
                     files={'file': (cur_task.get_file_path(),
                                     open(cur_task.get_file_path(), 'rb'),
                                     'multipart/form-data')}
                     )
        LOGGER.info(f'[To Controller {dst_device}] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()}  '
                    f'file: {cur_task.get_file_path()}')

    def run(self):
        assert None, 'Base Generator should not be invoked directly!'
