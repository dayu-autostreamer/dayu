import copy
import json

from .service import Service
from .dag import DAG


class Task:
    def __init__(self,
                 source_id: int,
                 task_id: int,
                 source_device: str,
                 dag: DAG = None,
                 flow_index: str = '',
                 metadata: dict = None,
                 raw_metadata: dict = None,
                 content: object = None,
                 scenario: dict = None,
                 temp: dict = None,
                 hash_data: list = None,
                 file_path: str = None):
        self.__source_id = source_id
        self.__task_id = task_id
        self.__source_device = source_device

        self.__metadata = metadata

        self.__raw_metadata = raw_metadata

        self.__dag_flow = dag

        self.__cur_flow_index = flow_index

        self.__content_data = content

        self.__scenario_data = scenario if scenario else {}

        self.__tmp_data = temp if temp else {}

        self.hash_data = hash_data if hash_data else []

        self.__file_path = file_path

    @staticmethod
    def extract_dag_from_dict(dag_dict: dict, start_node_name='start', end_node_name='end'):
        dag_flow = DAG.deserialize(dag_dict)
        if start_node_name not in dag_dict:
            dag_flow.add_start_node(Service(start_node_name))
            dag_flow.add_end_node(Service(end_node_name))

        return dag_flow

    @staticmethod
    def extract_dict_from_dag(dag_flow: DAG):
        return DAG.serialize(dag_flow)

    @staticmethod
    def extract_deployment_info_from_dag(dag_flow: DAG):
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

    def get_content(self):
        return self.__content_data

    def set_content(self, content):
        self.__content_data = content

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

    # TODO: calculate total delay
    def calculate_total_time(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'
        total_time = 0
        for service in self.__dag_flow:
            total_time += service.get_service_total_time()

        return total_time

    # TODO: calculate transmit time
    def calculate_cloud_edge_transmit_time(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'
        transmit_time = 0
        for service in self.__dag_flow:
            transmit_time = max(transmit_time, service.get_transmit_time())

        return transmit_time

    # TODO: get delay info
    def get_delay_info(self):
        assert self.__dag_flow, 'Task DAG is empty!'
        assert self.__cur_flow_index == 'end', f'DAG is not completed, current service: {self.__cur_flow_index}'

        delay_info = ''
        total_time = 0
        delay_info += f'[Delay Info] Source:{self.get_source_id()}  Task:{self.get_task_id()}\n'
        for service in self.__dag_flow:
            delay_info += f'stage[{service.get_service_name()}] -> (device:{service.get_execute_device()})    ' \
                          f'execute delay:{service.get_execute_time():.4f}s    ' \
                          f'transmit delay:{service.get_transmit_time():.4f}s\n'
        delay_info += f'total delay:{total_time:.4f}s average delay: {total_time / self.get_metadata()["buffer_size"]:.4f}s'
        return delay_info

    def step_to_next_stage(self):
        next_services = self.__dag_flow.get_next_nodes(self.__cur_flow_index)
        return [self.duplicate_task(service) for service in next_services]

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

    def duplicate_task(self, new_flow_index):
        new_task = copy.deepcopy(self)
        new_task.set_flow_index(new_flow_index)
        return new_task

    # TODO: merge task
    def merge_task(self, other_task):
        pass

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
            'content_data': task.get_content(),
            'scenario_data': task.get_scenario_data(),
            'tmp_data': task.get_tmp_data(),
            'hash_data': task.get_hash_data(),
            'file_path': task.get_file_path()
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
        task.set_content(data['content_data']) if 'content_data' in data else None
        task.set_scenario_data(data['scenario_data']) if 'scenario_data' in data else None
        task.set_tmp_data(data['tmp_data']) if 'tmp_data' in data else None
        task.set_hash_data(data['hash_data']) if 'hash_data' in data else None
        task.set_file_path(data['file_path']) if 'file_path' in data else None

        return task
