import copy
import json
import uuid

from .service import Service
from .dag import DAG

from core.lib.solver import LCASolver, IntermediateNodeSolver, PathSolver
from core.lib.common import NameMaintainer


class Task:
    def __init__(self,
                 source_id: int,
                 task_id: int,
                 source_device: str,
                 all_edge_devices: list,
                 dag: DAG = None,
                 flow_index: str = 'start',
                 past_flow_index: str = None,
                 metadata: dict = None,
                 raw_metadata: dict = None,
                 scenario: dict = None,
                 temp: dict = None,
                 hash_data: list = None,
                 file_path: str = None,
                 task_uuid: str = '',
                 parent_uuid: str = '',
                 root_uuid: str = ''):

        # unique uuid for each duplicated task
        self.__task_uuid = task_uuid or str(uuid.uuid4())
        # parent uuid for duplicated task (currently unused)
        self.__parent_uuid = parent_uuid
        # unique uuid for each task
        self.__root_uuid = root_uuid or self.__task_uuid

        # sequential id for each source
        self.__source_id = source_id
        # sequential id for each task
        self.__task_id = task_id
        # hostname of source binding device (position of generator)
        self.__source_device = source_device
        # hostname list of all offloading edge devices
        self.__all_edge_devices = all_edge_devices

        # metadata of task
        self.__metadata = metadata

        # raw metadata of source
        self.__raw_metadata = raw_metadata

        # dag info of task
        self.__dag_flow = dag

        # current service name in dag (work as pointer)
        self.__cur_flow_index = flow_index
        # past service name in dag
        self.__past_flow_index = past_flow_index

        # scenario data extracted from processing
        self.__scenario_data = scenario if scenario else {}

        # temporary data (main for time tickets)
        self.__tmp_data = temp if temp else {}

        # hash data for stream data in task
        self.hash_data = hash_data if hash_data else []

        # file path to store stream data
        self.__file_path = file_path

    @staticmethod
    def extract_dag_from_dict(dag_dict: dict, start_node_name='start', end_node_name='end'):
        """transfer DAG dict in to DAG class"""
        dag_flow = DAG.from_dict(dag_dict)
        if start_node_name not in dag_dict:
            dag_flow.add_start_node(Service(start_node_name))
        if end_node_name not in dag_dict:
            dag_flow.add_end_node(Service(end_node_name))
        dag_flow.validate_dag()

        return dag_flow

    @staticmethod
    def extract_dict_from_dag(dag_flow: DAG):
        """transfer DAG class in to DAG dict"""
        return dag_flow.to_dict()

    @staticmethod
    def extract_dag_deployment_from_dag(dag_flow: DAG):
        """
        get deployment info from dag class
        service_name/execute device for each node in DAG
        """
        dag_dict = dag_flow.to_dict()
        deployment_info = {}
        for node_name in dag_dict:
            node = dag_dict[node_name]
            deployment_info[node_name] = {'service': {'service_name': node_name,
                                                      'execute_device': node['service']['execute_device']},
                                          'next_nodes': node['next_nodes'],
                                          'prev_nodes': node['prev_nodes']}
        return deployment_info

    @staticmethod
    def extract_dag_from_dag_deployment(dag_deployment: dict):
        return Task.extract_dag_from_dict(dag_deployment)

    @staticmethod
    def extract_pipeline_deployment_from_dag(dag_flow: DAG):
        """get pipeline deployment info if dag has a linear structure"""
        if not dag_flow.check_is_pipeline():
            raise ValueError('Current DAG is not a pipeline structure.')

        pipeline_deployment_info = []
        start = 'start'
        end = 'end'
        current = dag_flow.get_next_nodes(start)[0]
        while current != end:
            pipeline_deployment_info.append(
                {'service_name': dag_flow.get_node(current).service.get_service_name(),
                 'execute_device': dag_flow.get_node(current).service.get_execute_device()}
            )
        pipeline_deployment_info.append(
            {'service_name': dag_flow.get_node(current).service.get_service_name(),
             'execute_device': dag_flow.get_node(current).service.get_execute_device()}
        )

        return pipeline_deployment_info

    @staticmethod
    def extract_dag_from_pipeline_deployment(pipeline_deployment: list):
        dag_dict = {}
        for service in pipeline_deployment[:-1]:
            dag_dict[service['service_name']] = {
                'service': service,
                'next_nodes': [],
            }
        prev_node = None
        for service in pipeline_deployment[-2::-1]:
            if prev_node:
                dag_dict[service['service_name']]['next_nodes'].append(prev_node)
            prev_node = service['service_name']
        dag_flow = Task.extract_dag_from_dict(dag_dict)
        return dag_flow

    @staticmethod
    def extract_pipeline_deployment_from_dag_deployment(dag_dict: dict):
        dag_flow = Task.extract_dag_from_dict(dag_dict)
        return Task.extract_pipeline_deployment_from_dag(dag_flow)

    @staticmethod
    def extract_dag_deployment_from_pipeline_deployment(pipeline_deployment: list):
        dag_workflow = Task.extract_dag_from_pipeline_deployment(pipeline_deployment)
        return Task.extract_dag_deployment_from_dag(dag_workflow)

    def get_pipeline_deployment_info(self):
        return Task.extract_pipeline_deployment_from_dag(self.__dag_flow)

    def get_dag_deployment_info(self):
        return Task.extract_dag_deployment_from_dag(self.__dag_flow)

    def get_source_id(self):
        return self.__source_id

    def get_task_id(self):
        return self.__task_id

    def get_source_device(self):
        return self.__source_device

    def get_all_edge_devices(self):
        return self.__all_edge_devices

    def get_dag(self):
        return self.__dag_flow

    def set_dag(self, dag):
        self.__dag_flow = dag

    def get_flow_index(self):
        return self.__cur_flow_index

    def set_flow_index(self, flow_index):
        self.__cur_flow_index = flow_index

    def get_past_flow_index(self):
        return self.__past_flow_index

    def set_past_flow_index(self, past_flow_index):
        self.__past_flow_index = past_flow_index

    def get_metadata(self):
        return self.__metadata

    def set_metadata(self, data: dict):
        self.__metadata = data

    def get_raw_metadata(self):
        return self.__raw_metadata

    def set_raw_metadata(self, data: dict):
        self.__raw_metadata = data

    def get_scenario_data(self):
        return self.__scenario_data

    def set_scenario_data(self, data: dict):
        self.__scenario_data = data

    def add_scenario(self, data: dict):
        self.__scenario_data.update(data)

    def get_tmp_data(self):
        return self.__tmp_data

    def set_tmp_data(self, data: dict):
        self.__tmp_data = data

    def get_file_path(self):
        return self.__file_path

    def set_file_path(self, path: str):
        self.__file_path = path

    def get_hash_data(self):
        return self.hash_data

    def set_hash_data(self, hash_data: list):
        self.hash_data = hash_data

    def add_hash_data(self, hash_code):
        self.hash_data.append(hash_code)

    def get_task_uuid(self):
        return self.__task_uuid

    def set_task_uuid(self, task_uuid: str):
        self.__task_uuid = task_uuid

    def get_parent_uuid(self):
        return self.__parent_uuid

    def set_parent_uuid(self, parent_uuid: str):
        self.__parent_uuid = parent_uuid

    def get_root_uuid(self):
        return self.__root_uuid

    def set_root_uuid(self, root_uuid: str):
        self.__root_uuid = root_uuid

    def get_current_content(self):
        return self.__dag_flow.get_node(self.__cur_flow_index).service.get_content_data()

    def get_prev_content(self):
        prev_service_names = self.__dag_flow.get_prev_nodes(self.__cur_flow_index)
        prev_contents = [self.__dag_flow.get_node(service_name).service.get_content_data()
                         for service_name in prev_service_names]
        # return one of prev non-empty content
        return next((content for content in prev_contents if content is not None), None)

    def get_first_content(self):
        first_service_names = self.__dag_flow.get_next_nodes('start')
        first_contents = [self.__dag_flow.get_node(service_name).service.get_content_data()
                          for service_name in first_service_names]
        # return one of first non-empty content
        return next((content for content in first_contents if content is not None), None)

    def get_last_content(self):
        last_service_names = self.__dag_flow.get_prev_nodes('end')
        last_contents = [self.__dag_flow.get_node(service_name).service.get_content_data()
                          for service_name in last_service_names]
        # return one of first non-empty content
        return next((content for content in last_contents if content is not None), None)

    def set_current_content(self, content):
        self.__dag_flow.get_node(self.__cur_flow_index).service.set_content_data(content)

    def get_current_service_info(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        service = self.__dag_flow.get_node(self.__cur_flow_index).service
        return service.get_service_name(), service.get_execute_device()

    def save_transmit_time(self, transmit_time):
        assert self.__dag_flow, 'Task DAG is empty!'
        service = self.__dag_flow.get_node(self.__cur_flow_index).service
        service.set_transmit_time(transmit_time=transmit_time)

    def save_execute_time(self, execute_time):
        assert self.__dag_flow, 'Task DAG is empty!'
        service = self.__dag_flow.get_node(self.__cur_flow_index).service
        service.set_execute_time(execute_time=execute_time)

    def save_real_execute_time(self, real_execute_time):
        assert self.__dag_flow, 'Task DAG is empty!'
        service = self.__dag_flow.get_node(self.__cur_flow_index).service
        service.set_real_execute_time(real_execute_time=real_execute_time)

    def get_real_end_to_end_time(self):
        """get real end to end time of task: from generator to distributor by estimation"""
        tag_prefix = NameMaintainer.get_time_ticket_tag_prefix(self)
        if f'{tag_prefix}:total_start_time' not in self.__tmp_data:
            raise ValueError(f'Timestamp of task starting lacks: "{tag_prefix}:total_start_time"')
        if f'{tag_prefix}:total_start_time' not in self.__tmp_data:
            raise ValueError(f'Timestamp of task ending lacks: "{tag_prefix}:total_end_time"')

        return (self.__tmp_data[f'{tag_prefix}:total_end_time'] -
                self.__tmp_data[f'{tag_prefix}:total_start_time'])

    def calculate_total_time(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'

        total_time, _ = PathSolver(self.__dag_flow).get_weighted_longest_path('start', 'end',
                                                                               lambda x: x.get_service_total_time())
        return total_time

    def calculate_cloud_edge_transmit_time(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'

        # get the longest transmitting time as cloud-edge transmitting time
        transmit_time = 0
        for service_name in self.__dag_flow:
            service = self.__dag_flow.get_node(service_name).service
            transmit_time = max(transmit_time, service.get_transmit_time())
        return transmit_time

    def get_delay_info(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'

        delay_info = ''
        total_time = self.calculate_total_time()
        real_total_time = self.get_real_end_to_end_time()
        delay_info += f'[Delay Info] Source:{self.get_source_id()}  Task:{self.get_task_id()}\n'
        for service_name in self.__dag_flow.nodes:
            service = self.__dag_flow.get_node(service_name).service
            delay_info += f'stage[{service.get_service_name()}] -> (device:{service.get_execute_device()})    ' \
                          f'execute delay:{service.get_execute_time():.4f}s    ' \
                          f'transmit delay:{service.get_transmit_time():.4f}s\n'
        delay_info += (f'total delay:{total_time:.4f}s  '
                       f'average delay: {total_time / self.get_metadata()["buffer_size"]:.4f}s\n')
        delay_info += (f'real end-to-end delay:{real_total_time:.4f}s  '
                       f'average delay: {real_total_time / self.get_metadata()["buffer_size"]:.4f}s\n')
        return delay_info

    def get_parallel_info_for_merge(self):
        """
        Obtain nodes parallel to the current node and the corresponding joint nodes

        output:
        [
            {joint_service: joint_service_name1, parallel_services:[...]},
            {joint_service: joint_service_name2, parallel_services:[...]},
            ...
        ]
        """
        next_node_names = self.__dag_flow.get_node(self.__cur_flow_index).next_nodes

        parallel_info = [
            {
                'joint_service': next_node_name,
                'parallel_services': self.__dag_flow.get_prev_nodes(next_node_name)
            }
            for next_node_name in next_node_names
        ]

        return parallel_info

    def step_to_next_stage(self):
        next_services = self.__dag_flow.get_next_nodes(self.__cur_flow_index)
        return [self.fork_task(service) for service in next_services]

    def get_current_stage_device(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        return self.__dag_flow.get_node(self.__cur_flow_index).service.get_execute_device()

    def set_initial_execute_device(self, device):
        Task.set_execute_device(self.__dag_flow, device)

    @staticmethod
    def set_execute_device(dag, device):
        assert dag, 'DAG is empty!'

        for node_name in dag.nodes:
            node = dag.nodes[node_name]
            node.service.set_execute_device(device)
        return dag

    def fork_task(self, new_flow_index: str = None) -> 'Task':
        new_task = copy.deepcopy(self)
        if new_flow_index and new_flow_index != self.__cur_flow_index:
            new_task.set_past_flow_index(self.__cur_flow_index)
            new_task.set_flow_index(new_flow_index)
        new_task.set_task_uuid(str(uuid.uuid4()))
        new_task.set_parent_uuid(self.__task_uuid)
        return new_task

    def merge_task(self, other_task: 'Task'):
        lca_service_name = LCASolver(self.__dag_flow).find_lca(self.get_past_flow_index(), other_task.get_past_flow_index())

        merged_dag = self.get_dag()
        other_dag = other_task.get_dag()

        # Complete missing part of merged_task with other_task
        # missing part contains intermediate nodes between "LCA" and "current node of other_task" (including latter)
        nodes_for_merge = IntermediateNodeSolver(merged_dag).get_intermediate_nodes(lca_service_name,
                                                                                    other_task.get_past_flow_index())
        nodes_for_merge.add(other_task.get_past_flow_index())

        for node in nodes_for_merge:
            merged_dag.set_node_service(node, other_dag.get_node(node).service)

        self.set_dag(merged_dag)

    def to_dict(self):
        return {
            'source_id': self.get_source_id(),
            'task_id': self.get_task_id(),
            'source_device': self.get_source_device(),
            'all_edge_devices': self.get_all_edge_devices(),
            'dag': self.get_dag().to_dict() if self.get_dag() else None,
            'cur_flow_index': self.get_flow_index(),
            'past_flow_index': self.get_past_flow_index(),
            'meta_data': self.get_metadata(),
            'raw_meta_data': self.get_raw_metadata(),
            'scenario_data': self.get_scenario_data(),
            'tmp_data': self.get_tmp_data(),
            'hash_data': self.get_hash_data(),
            'file_path': self.get_file_path(),
            'task_uuid': self.get_task_uuid(),
            'parent_uuid': self.get_parent_uuid(),
            'root_uuid': self.get_root_uuid(),
        }

    @classmethod
    def from_dict(cls, dag_dict):
        task = cls(source_id=dag_dict['source_id'],
                   task_id=dag_dict['task_id'],
                   source_device=dag_dict['source_device'],
                   all_edge_devices=dag_dict['all_edge_devices'])

        task.set_dag(DAG.from_dict(dag_dict['dag'])) if 'dag' in dag_dict and dag_dict['dag'] else None
        task.set_flow_index(dag_dict['cur_flow_index']) if 'cur_flow_index' in dag_dict else None
        task.set_past_flow_index(dag_dict['past_flow_index']) if 'past_flow_index' in dag_dict else None
        task.set_metadata(dag_dict['meta_data']) if 'meta_data' in dag_dict else None
        task.set_raw_metadata(dag_dict['raw_meta_data']) if 'raw_meta_data' in dag_dict else None
        task.set_scenario_data(dag_dict['scenario_data']) if 'scenario_data' in dag_dict else None
        task.set_tmp_data(dag_dict['tmp_data']) if 'tmp_data' in dag_dict else None
        task.set_hash_data(dag_dict['hash_data']) if 'hash_data' in dag_dict else None
        task.set_file_path(dag_dict['file_path']) if 'file_path' in dag_dict else None
        task.set_task_uuid(dag_dict['task_uuid']) if 'task_uuid' in dag_dict else None
        task.set_parent_uuid(dag_dict['parent_uuid']) if 'parent_uuid' in dag_dict else None
        task.set_root_uuid(dag_dict['root_uuid']) if 'root_uuid' in dag_dict else None

        return task

    def serialize(self):
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, data: str):
        data = json.loads(data)
        return cls.from_dict(data)
