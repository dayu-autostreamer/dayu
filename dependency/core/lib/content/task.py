import copy
import json
import uuid

from .service import Service
from .dag import DAG

from core.lib.solver import LCASolver, IntermediateNodeSolver, PathSolver


class Task:
    def __init__(self,
                 source_id: int,
                 task_id: int,
                 source_device: str,
                 dag: DAG = None,
                 flow_index: str = 'start',
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

        # metadata of task
        self.__metadata = metadata

        # raw metadata of source
        self.__raw_metadata = raw_metadata

        # dag info of task
        self.__dag_flow = dag

        # current service name in dag (work as pointer)
        self.__cur_flow_index = flow_index

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
        dag_flow = DAG.deserialize(dag_dict)
        if start_node_name not in dag_dict:
            dag_flow.add_start_node(Service(start_node_name))
        if end_node_name not in dag_dict:
            dag_flow.add_end_node(Service(end_node_name))

        return dag_flow

    @staticmethod
    def extract_dict_from_dag(dag_flow: DAG):
        """transfer DAG class in to DAG dict"""
        return DAG.serialize(dag_flow)

    @staticmethod
    def extract_deployment_info_from_dag(dag_flow: DAG):
        """
        get deployment info from dag class
        service_name/execute device for each node in DAG
        """
        dag_dict = DAG.serialize(dag_flow)
        deployment_info = {}
        for node_name in dag_dict:
            node = dag_dict[node_name]
            deployment_info[node_name] = {'service': {'service_name': node_name,
                                                      'execute_device': node['service']['execute_device']},
                                          'next_nodes': node['next_nodes'],
                                          'prev_nodes': node['prev_nodes']}
        return deployment_info

    def get_dag_deployment_info(self):
        return Task.extract_deployment_info_from_dag(self.__dag_flow)

    def get_source_id(self):
        return self.__source_id

    def get_task_id(self):
        return self.__task_id

    def get_source_device(self):
        return self.__source_device

    def get_dag(self):
        return self.__dag_flow

    def set_dag(self, dag):
        self.__dag_flow = dag

    def get_flow_index(self):
        return self.__cur_flow_index

    def set_flow_index(self, flow_index):
        self.__cur_flow_index = flow_index

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
        assert uuid.UUID(task_uuid).version == 4, \
            f'Invalid version for input UUID, need 4 give {uuid.UUID(task_uuid).version}'
        self.__task_uuid = task_uuid

    def get_parent_uuid(self):
        return self.__parent_uuid

    def set_parent_uuid(self, parent_uuid: str):
        assert not parent_uuid or uuid.UUID(parent_uuid).version == 4, \
            f'Invalid version for input UUID, need 4 give {uuid.UUID(parent_uuid).version}'
        self.__parent_uuid = parent_uuid

    def get_root_uuid(self):
        return self.__root_uuid

    def set_root_uuid(self, root_uuid: str):
        assert uuid.UUID(root_uuid).version == 4, \
            f'Invalid version for input UUID, need 4 give {uuid.UUID(root_uuid).version}'
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
        first_service_names = self.__dag_flow.get_prev_nodes('start')
        first_contents = [self.__dag_flow.get_node(service_name).service.get_content_data()
                          for service_name in first_service_names]
        # return one of first non-empty content
        return next((content for content in first_contents if content is not None), None)

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
        if f'dayu:{self.__root_uuid}:total_start_time' not in self.__tmp_data:
            raise ValueError(f'Timestamp of task starting lacks: dayu:{self.__root_uuid}:total_start_time')
        if f'dayu:{self.__root_uuid}:total_start_time' not in self.__tmp_data:
            raise ValueError(f'Timestamp of task ending lacks: dayu:{self.__root_uuid}:total_end_time')

        return (self.__tmp_data[f'dayu:{self.__root_uuid}:total_end_time'] -
                self.__tmp_data[f'dayu:{self.__root_uuid}:total_start_time'])

    def calculate_total_time(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'

        total_time, _ = PathSolver(self.__dag_flow).get_weighted_shortest_path('start', 'end',
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
        for service in self.__dag_flow:
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

        for node_name in dag:
            node = dag[node_name]
            node.service.set_execute_device(device)
        return dag

    def fork_task(self, new_flow_index: str = None) -> 'Task':
        new_task = copy.deepcopy(self)
        if new_flow_index:
            new_task.set_flow_index(new_flow_index)
        new_task.set_task_uuid(uuid.uuid4())
        new_task.set_parent_uuid(self.__task_uuid)
        return new_task

    def merge_task(self, other_task: 'Task'):
        lca_service_name = LCASolver(self.__dag_flow).find_lca(self.get_flow_index(), other_task.get_flow_index())

        merged_task = copy.deepcopy(self)
        merged_task.set_flow_index(lca_service_name)
        merged_task.set_task_uuid(uuid.uuid4())

        merged_dag = merged_task.get_dag()
        other_dag = other_task.get_dag()

        # Complete missing part of merged_task with other_task
        # missing part contains intermediate nodes between "LCA" and "current node of other_task" (including latter)
        nodes_for_merge = IntermediateNodeSolver(merged_dag).get_intermediate_nodes(lca_service_name,
                                                                                    other_task.get_flow_index())
        nodes_for_merge.add(other_task.get_flow_index())

        for node in nodes_for_merge:
            merged_dag.set_node_service(node, other_dag.get_node(node).service)

        merged_task.set_dag(merged_dag)

    @staticmethod
    def serialize(task: 'Task'):
        return json.dumps({
            'source_id': task.get_source_id(),
            'task_id': task.get_task_id(),
            'source_device': task.get_source_device(),
            'dag': DAG.serialize(task.get_dag()),
            'cur_flow_index': task.get_flow_index(),
            'meta_data': task.get_metadata(),
            'raw_meta_data': task.get_raw_metadata(),
            'scenario_data': task.get_scenario_data(),
            'tmp_data': task.get_tmp_data(),
            'hash_data': task.get_hash_data(),
            'file_path': task.get_file_path(),
            'task_uuid': task.get_task_uuid(),
            'parent_uuid': task.get_parent_uuid(),
            'root_uuid': task.get_root_uuid(),
        })

    @staticmethod
    def deserialize(data: str):
        data = json.loads(data)
        task = Task(source_id=data['source_id'],
                    task_id=data['task_id'],
                    source_device=data['source_device'])

        task.set_dag(DAG.deserialize(data['dag'])) if 'dag' in data else None
        task.set_flow_index(data['cur_flow_index']) if 'cur_flow_index' in data else None
        task.set_metadata(data['meta_data']) if 'meta_data' in data else None
        task.set_raw_metadata(data['raw_meta_data']) if 'raw_meta_data' in data else None
        task.set_scenario_data(data['scenario_data']) if 'scenario_data' in data else None
        task.set_tmp_data(data['tmp_data']) if 'tmp_data' in data else None
        task.set_hash_data(data['hash_data']) if 'hash_data' in data else None
        task.set_file_path(data['file_path']) if 'file_path' in data else None
        task.set_task_uuid(data['task_uuid']) if 'task_uuid' in data else None
        task.set_parent_uuid(data['parent_uuid']) if 'parent_uuid' in data else None
        task.set_root_uuid(data['root_uuid']) if 'root_uuid' in data else None

        return task
