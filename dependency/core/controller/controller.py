import os

from core.lib.estimation import TimeEstimator
from core.lib.network import http_request
from core.lib.common import LOGGER
from core.lib.common import Context
from core.lib.common import SystemConstant
from core.lib.content import Task
from core.lib.network import get_merge_address
from core.lib.network import NodeInfo, PortInfo
from core.lib.network import NetworkAPIPath, NetworkAPIMethod

from .task_coordinator import TaskCoordinator


# TODO: check for time storage
class Controller:
    def __init__(self):
        self.task_coordinator = TaskCoordinator()

        self.cur_task = None

        self.is_display = Context.get_parameter('DISPLAY', direct=False)

        self.service_ports_dict = PortInfo.get_service_ports_dict()
        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.distributor_port = PortInfo.get_component_port(SystemConstant.DISTRIBUTOR.value)
        self.distributor_hostname = NodeInfo.get_cloud_node()
        self.distribute_address = get_merge_address(NodeInfo.hostname2ip(self.distributor_hostname),
                                                    port=self.distributor_port,
                                                    path=NetworkAPIPath.DISTRIBUTOR_DISTRIBUTE)

        self.local_device = NodeInfo.get_local_device()

    def set_current_task(self, task_data: str):
        self.cur_task = Task.deserialize(task_data)

    def send_task_to_other_device(self, device):
        controller_address = get_merge_address(NodeInfo.hostname2ip(device),
                                               port=self.controller_port,
                                               path=NetworkAPIPath.CONTROLLER_TASK)

        self.record_transmit_ts(is_end=False)

        http_request(url=controller_address,
                     method=NetworkAPIMethod.CONTROLLER_TASK,
                     data={'data': Task.serialize(self.cur_task)},
                     files={'file': (self.cur_task.get_file_path(),
                                     open(self.cur_task.get_file_path(), 'rb'),
                                     'multipart/form-data')}
                     )

        LOGGER.info(f'[To Device {device}] source: {self.cur_task.get_source_id()}  '
                    f'task: {self.cur_task.get_task_id()}')

    def send_task_to_service(self, service):
        service_address = get_merge_address(NodeInfo.hostname2ip(self.local_device),
                                            port=self.service_ports_dict[service],
                                            path=NetworkAPIPath.PROCESSOR_PROCESS
                                            )

        self.record_execute_ts(is_end=False)

        http_request(url=service_address,
                     method=NetworkAPIMethod.PROCESSOR_PROCESS,
                     data={'data': Task.serialize(self.cur_task)},
                     files={'file': (self.cur_task.get_file_path(),
                                     open(self.cur_task.get_file_path(), 'rb'),
                                     'multipart/form-data')}
                     )

        LOGGER.info(f'[To Service {service}] source: {self.cur_task.get_source_id()}  '
                    f'task: {self.cur_task.get_task_id()}')

    def send_task_to_distributor(self):
        self.record_transmit_ts(is_end=False)

        if not os.path.exists(self.cur_task.get_file_path()):
            LOGGER.warning(f'[Task File Lost] source: {self.cur_task.get_source_id()}  '
                           f'task: {self.cur_task.get_task_id()} file: {self.cur_task.get_file_path()}')
            return
        file_content = open(self.cur_task.get_file_path(), 'rb') if self.is_display else b''

        http_request(url=self.distribute_address,
                     method=NetworkAPIMethod.DISTRIBUTOR_DISTRIBUTE,
                     files={'file': (self.cur_task.get_file_path(), file_content, 'multipart/form-data')},
                     data={'data': Task.serialize(self.cur_task)}
                     )

        LOGGER.info(f'[To Distributor] source: {self.cur_task.get_source_id()}  task: {self.cur_task.get_task_id()}')

    def submit_task(self):

        if not self.cur_task:
            LOGGER.warning('Current task of controller is NOT set!')
            return

        LOGGER.info(f'[Submit Task] source: {self.cur_task.get_source_id()}  task: {self.cur_task.get_task_id()}')

        service_name, _ = self.cur_task.get_current_service_info()
        dst_device = self.cur_task.get_current_stage_device()

        if service_name == 'end':
            self.send_task_to_distributor()
            action = 'transmit'
        elif dst_device != self.local_device:
            self.send_task_to_other_device(dst_device)
            action = 'transmit'
        else:
            self.send_task_to_service(service_name)
            action = 'execute'

        return action

    def process_return(self):
        assert self.cur_task, 'Current task of controller is NOT set!'

        LOGGER.info(f'[Process Return] source: {self.cur_task.get_source_id()}  task: {self.cur_task.get_task_id()}')

        required_parallel_task_count = len(self.cur_task.get_parallel_services())

        # node with parallel nodes should merge before step to next stage
        if required_parallel_task_count > 1:
            # current node have parallel nodes
            joint_service_name = self.cur_task.get_joint_service()
            parallel_count = self.task_coordinator.store_task_data(self.cur_task, joint_service_name)

            # wait when some duplicated tasks (with parallel nodes) not arrive
            if parallel_count != required_parallel_task_count:
                self.cur_task = None
                return
            tasks = self.task_coordinator.retrieve_task_data(self.cur_task.get_root_uuid(),
                                                             joint_service_name,
                                                             required_parallel_task_count)
            # something wrong causes invalid task retrieving
            if not tasks:
                self.cur_task = None
                return

            # merge parallel tasks
            for task in tasks:
                self.cur_task.merge_task(task)

        self.cur_task.step_to_next_stage()

    def record_transmit_ts(self, is_end: bool):
        assert self.cur_task, 'Current task of controller is NOT set!'

        try:
            task, duration = TimeEstimator.record_pipeline_ts(self.cur_task, is_end=is_end, sub_tag='transmit')
            self.cur_task = task
        except Exception as e:
            LOGGER.warning(f'Time record failed: {str(e)}')
            duration = 0

        if is_end:
            self.cur_task.save_transmit_time(duration)
            LOGGER.info(f'[Source {self.cur_task.get_source_id()} / Task {self.cur_task.get_task_id()}] '
                        f'record transmit time of stage {self.cur_task.get_flow_index()}: {duration:.3f}s')

    def record_execute_ts(self, is_end: bool):
        assert self.cur_task, 'Current task of controller is NOT set!'

        try:
            task, duration = TimeEstimator.record_pipeline_ts(self.cur_task, is_end=is_end, sub_tag='execute')
            self.cur_task = task
        except Exception as e:
            LOGGER.warning(f'Time record failed: {str(e)}')
            duration = 0

        if is_end:
            self.cur_task.save_execute_time(duration)
            LOGGER.info(f'[Source {self.cur_task.get_source_id()} / Task {self.cur_task.get_task_id()}] '
                        f'record execute time of stage {self.cur_task.get_flow_index()}: {duration:.3f}s')
