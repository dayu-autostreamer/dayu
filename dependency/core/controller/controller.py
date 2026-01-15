import os

from core.lib.estimation import TimeEstimator
from core.lib.network import http_request, merge_address, NodeInfo, PortInfo, NetworkAPIPath, NetworkAPIMethod
from core.lib.common import LOGGER, Context, SystemConstant, KubeConfig, TaskConstant
from core.lib.content import Task

from .task_coordinator import TaskCoordinator


class Controller:
    def __init__(self):
        self.task_coordinator = TaskCoordinator()

        self.is_display = Context.get_parameter('DISPLAY', direct=False)

        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.distributor_port = PortInfo.get_component_port(SystemConstant.DISTRIBUTOR.value)
        self.distributor_hostname = NodeInfo.get_cloud_node()
        self.distribute_address = merge_address(NodeInfo.hostname2ip(self.distributor_hostname),
                                                port=self.distributor_port,
                                                path=NetworkAPIPath.DISTRIBUTOR_DISTRIBUTE)

        self.local_device = NodeInfo.get_local_device()
        self.cloud_device = NodeInfo.get_cloud_node()

    def check_processor_health(self):
        PortInfo.force_refresh()
        service_ports_dict = PortInfo.get_service_ports_dict(self.local_device)
        LOGGER.debug(f'[HEALTH CHECK] health checking services: {service_ports_dict.keys()}')
        for service, service_port in service_ports_dict.items():
            processor_health_address = merge_address(NodeInfo.hostname2ip(self.local_device),
                                                     port=service_port,
                                                     path=NetworkAPIPath.PROCESSOR_HEALTH)
            response = http_request(url=processor_health_address, method=NetworkAPIMethod.PROCESSOR_HEALTH)
            if not response or response.get('status') != 'ok':
                LOGGER.debug(f'[HEALTH CHECK] service {service} processor health check failed.')
                return False
            LOGGER.debug(f'[HEALTH CHECK] service {service} processor health check succeed.')
        return True

    def send_task_to_other_device(self, cur_task: Task, device: str = ''):
        self.record_transmit_ts(cur_task=cur_task, is_end=False)
        controller_address = merge_address(NodeInfo.hostname2ip(device),
                                           port=self.controller_port,
                                           path=NetworkAPIPath.CONTROLLER_TASK)

        http_request(url=controller_address,
                     method=NetworkAPIMethod.CONTROLLER_TASK,
                     data={'data': cur_task.serialize()},
                     files={'file': (cur_task.get_file_path(),
                                     open(Context.get_temporary_file_path(cur_task.get_file_path()), 'rb'),
                                     'multipart/form-data')})

        LOGGER.info(f'[To Device {device}] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()} current service: {cur_task.get_flow_index()}')

    def send_task_to_service(self, cur_task: Task, service: str = ''):
        self.record_execute_ts(cur_task=cur_task, is_end=False)

        service_ports_dict = PortInfo.get_service_ports_dict(self.local_device)

        # Force refresh port cache and retry once
        if service not in service_ports_dict:
            PortInfo.force_refresh()
            service_ports_dict = PortInfo.get_service_ports_dict(self.local_device)

        if service not in service_ports_dict:
            service_deployment = KubeConfig.get_service_nodes_dict()
            if service not in service_deployment or not service_deployment[service] or \
                    self.cloud_device not in service_deployment[service]:
                LOGGER.warning(f'[Service Not Exist] Service {service} does not exist in {self.local_device} '
                               f'({self.local_device} has service: {service_ports_dict.keys()}).')
                return 'error'

            LOGGER.warning(f'[Service Not In Current Device] Service {service} does not exist in {self.local_device} '
                           f'({self.local_device} has service: {service_ports_dict.keys()}, '
                           f'Service {service} running on {service_deployment[service]})). '
                           f'Transmit to cloud device {self.cloud_device} as default.')

            cur_task.set_current_stage_device(self.cloud_device)
            self.erase_execute_ts(cur_task)
            self.erase_transmit_ts(cur_task)
            self.submit_task(cur_task=cur_task)
            return 'transmit'

        service_address = merge_address(NodeInfo.hostname2ip(self.local_device),
                                        port=service_ports_dict[service],
                                        path=NetworkAPIPath.PROCESSOR_PROCESS_LOCAL)

        if not os.path.exists(Context.get_temporary_file_path(cur_task.get_file_path())):
            LOGGER.warning(f'[Task File Lost] source: {cur_task.get_source_id()}  '
                           f'task: {cur_task.get_task_id()} '
                           f'file: {Context.get_temporary_file_path(cur_task.get_file_path())}')
            return 'error'

        # Local fast path: only send metadata
        http_request(url=service_address,
                     method=NetworkAPIMethod.PROCESSOR_PROCESS_LOCAL,
                     data={'data': cur_task.serialize()})

        LOGGER.info(f'[To Service {service} Local] source: {cur_task.get_source_id()}  '
                    f'task: {cur_task.get_task_id()} current service: {cur_task.get_flow_index()}')

        return 'execute'

    def send_task_to_distributor(self, cur_task: Task):
        self.record_transmit_ts(cur_task=cur_task, is_end=False)
        if not os.path.exists(Context.get_temporary_file_path(cur_task.get_file_path())):
            LOGGER.warning(f'[Task File Lost] source: {cur_task.get_source_id()}  '
                           f'task: {cur_task.get_task_id()} '
                           f'file: {Context.get_temporary_file_path(cur_task.get_file_path())}')
            return
        file_content = open(Context.get_temporary_file_path(cur_task.get_file_path()), 'rb') if self.is_display else b''

        http_request(url=self.distribute_address,
                     method=NetworkAPIMethod.DISTRIBUTOR_DISTRIBUTE,
                     files={'file': (cur_task.get_file_path(), file_content, 'multipart/form-data')},
                     data={'data': cur_task.serialize()})

        LOGGER.info(f'[To Distributor] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()} '
                    f'current service: {cur_task.get_flow_index()}')

    def submit_task(self, cur_task: Task):
        if not cur_task:
            LOGGER.warning('Current task is None')
            return 'error'

        LOGGER.info(f'[Submit Task] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()} '
                    f'current service: {cur_task.get_flow_index()}')

        service_name, _ = cur_task.get_current_service_info()
        dst_device = cur_task.get_current_stage_device()

        if service_name == TaskConstant.START.value:
            next_tasks = cur_task.step_to_next_stage()
            actions = [self.submit_task(new_task) for new_task in next_tasks]
            action = 'execute' if 'execute' in actions else 'transmit'
        elif service_name == TaskConstant.END.value:
            self.send_task_to_distributor(cur_task)
            action = 'transmit'
        elif dst_device != self.local_device:
            self.send_task_to_other_device(cur_task, dst_device)
            action = 'transmit'
        else:
            action = self.send_task_to_service(cur_task, service_name)

        return action

    def process_return(self, cur_task):
        """step to next step and submit task"""
        assert cur_task, 'Current task is None'

        LOGGER.info(f'[Process Return] source: {cur_task.get_source_id()}  task: {cur_task.get_task_id()}')

        actions = []
        parallel_joints = cur_task.get_parallel_info_for_merge()
        for parallel_joint in parallel_joints:
            joint_service_name = parallel_joint['joint_service']
            parallel_service_names = parallel_joint['parallel_services']
            required_parallel_task_count = len(parallel_service_names)
            new_task = cur_task.fork_task(joint_service_name)

            # node with parallel nodes should merge before step to next stage
            if required_parallel_task_count > 1:
                parallel_count = self.task_coordinator.store_task_data(new_task, joint_service_name)
                # wait when some duplicated tasks (with parallel nodes) not arrive
                if parallel_count != required_parallel_task_count:
                    actions.append('wait')
                    continue
                # retrieve parallel nodes in redis
                tasks = self.task_coordinator.retrieve_task_data(new_task.get_root_uuid(),
                                                                 joint_service_name,
                                                                 required_parallel_task_count)
                # something wrong causes invalid task retrieving
                if not tasks:
                    LOGGER.warning('Invalid task retrieving from task coordinator!')
                    actions.append('wait')
                    continue

                # merge parallel tasks
                for task in tasks:
                    new_task.merge_task(task)
                LOGGER.debug(f"Merge task with services {[task.get_past_flow_index() for task in tasks]} "
                             f"into task with service '{joint_service_name}'")

            actions.append(self.submit_task(new_task))

        return actions

    @staticmethod
    def record_transmit_ts(cur_task: Task, is_end: bool = False):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            duration = TimeEstimator.record_dag_ts(cur_task, is_end=is_end, sub_tag='transmit')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                           f'transmit time record failed of stage {cur_task.get_flow_index()}: {str(e)}')
            duration = 0

        if is_end:
            cur_task.save_transmit_time(duration)
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'record transmit time of stage {cur_task.get_flow_index()}: {duration:.3f}s')

    @staticmethod
    def record_execute_ts(cur_task: Task, is_end: bool = False):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            duration = TimeEstimator.record_dag_ts(cur_task, is_end=is_end, sub_tag='execute')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                           f'execute time record failed of stage {cur_task.get_flow_index()}: {str(e)}')
            duration = 0

        if is_end:
            cur_task.save_execute_time(duration)
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'record execute time of stage {cur_task.get_flow_index()}: {duration:.3f}s')

    @staticmethod
    def erase_transmit_ts(cur_task: Task):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            TimeEstimator.erase_dag_ts(cur_task, is_end=False, sub_tag=f'transmit')
            TimeEstimator.erase_dag_ts(cur_task, is_end=True, sub_tag=f'transmit')
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'erase transmit time of stage {cur_task.get_flow_index()}')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}]'
                           f' Erase transmit time failed for stage {cur_task.get_flow_index()}: {str(e)}')

    @staticmethod
    def erase_execute_ts(cur_task: Task):
        assert cur_task, 'Current task of controller is NOT set!'

        try:
            TimeEstimator.erase_dag_ts(cur_task, is_end=False, sub_tag=f'execute')
            LOGGER.info(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}] '
                        f'erase execute time of stage {cur_task.get_flow_index()}')
        except Exception as e:
            LOGGER.warning(f'[Source {cur_task.get_source_id()} / Task {cur_task.get_task_id()}]'
                           f' Erase execute time failed for stage {cur_task.get_flow_index()}: {str(e)}')
