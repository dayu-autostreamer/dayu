import json

from core.lib.common import Context, LOGGER, SystemConstant
from core.lib.content import Task
from core.lib.network import get_merge_address
from core.lib.network import NodeInfo, PortInfo
from core.lib.network import NetworkAPIPath, NetworkAPIMethod
from core.lib.network import http_request
from core.lib.estimation import TimeEstimator


# TODO: update pipeline to dag
class Generator:
    def __init__(self, source_id: int, metadata: dict, task_dag: list, ):
        self.current_task = None

        self.source_id = source_id
        self.task_dag = Task.extract_dag_from_dict(task_dag)

        self.raw_meta_data = metadata.copy()
        self.meta_data = metadata.copy()

        self.task_content = None

        self.local_device = NodeInfo.get_local_device()
        self.task_dag = Task.set_execute_device(self.task_dag, self.local_device)

        self.scheduler_hostname = NodeInfo.get_cloud_node()
        self.scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.schedule_address = get_merge_address(NodeInfo.hostname2ip(self.scheduler_hostname),
                                                  port=self.scheduler_port, path=NetworkAPIPath.SCHEDULER_SCHEDULE)

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

    def record_total_start_ts(self):
        self.current_task, _ = TimeEstimator.record_task_ts(self.current_task,
                                                            'total_start_time',
                                                            is_end=False)

    def record_transmit_start_ts(self):
        self.current_task, _ = TimeEstimator.record_pipeline_ts(self.current_task,
                                                                is_end=False,
                                                                sub_tag='transmit')

    def submit_task_to_controller(self, compressed_file, hash_codes):
        assert self.current_task, 'Task is empty when submit to controller!'

        self.before_submit_task_operation(self, compressed_file, hash_codes)

        dst_device = self.current_task.get_current_stage_device()
        controller_ip = NodeInfo.hostname2ip(dst_device)
        controller_address = get_merge_address(controller_ip,
                                               port=self.controller_port,
                                               path=NetworkAPIPath.CONTROLLER_TASK)
        self.record_transmit_start_ts()
        http_request(url=controller_address,
                     method=NetworkAPIMethod.CONTROLLER_TASK,
                     data={'data': Task.serialize(self.current_task)},
                     files={'file': (self.current_task.get_file_path(),
                                     open(self.current_task.get_file_path(), 'rb'),
                                     'multipart/form-data')}
                     )
        LOGGER.info(f'[To Controller {dst_device}] source: {self.current_task.get_source_id()}  '
                    f'task: {self.current_task.get_task_id()}  '
                    f'file: {self.current_task.get_file_path()}')

    def run(self):
        assert None, 'Base Generator should not be invoked directly!'
