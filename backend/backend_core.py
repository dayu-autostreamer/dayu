import copy
import re
from collections import deque
from func_timeout import func_set_timeout as timeout

import os
import time
from core.lib.content import Task
from core.lib.common import LOGGER, Context, YamlOps, FileOps, Counter, SystemConstant
from core.lib.network import http_request, NodeInfo, PortInfo, merge_address, NetworkAPIPath, NetworkAPIMethod

from kube_helper import KubeHelper
from template_helper import TemplateHelper


class BackendCore:
    def __init__(self):

        self.template_helper = TemplateHelper(Context.get_file_path(0))

        self.namespace = ''
        self.image_meta = None
        self.schedulers = None
        self.services = None
        self.visualizations = None

        self.source_configs = []

        self.dags = []

        self.time_ticket = 0

        self.result_url = None
        self.result_file_url = None
        self.resource_url = None
        self.log_fetch_url = None
        self.log_clear_url = None

        self.inner_datasource = self.check_simulation_datasource()
        self.source_open = False
        self.source_label = ''

        self.task_results = {}

        self.is_get_result = False

        self.cur_yaml_docs = None
        self.save_yaml_path = 'resources.yaml'
        self.log_file_path = 'log.json'

        self.default_visualization_image = 'default_visualization.png'

        self.parse_base_info()

    def parse_base_info(self):
        try:
            base_info = self.template_helper.load_base_info()
            self.namespace = base_info['namespace']
            self.image_meta = base_info['default-image-meta']
            self.schedulers = base_info['scheduler-policies']
            self.services = base_info['services']
            self.visualizations = base_info['visualizations']
        except KeyError as e:
            LOGGER.warning(f'Parse base info failed: {str(e)}')

    def get_log_file_name(self):
        base_info = self.template_helper.load_base_info()
        load_file_name = base_info['log-file-name']
        if not load_file_name:
            return None
        return load_file_name.split('.')[0]

    @timeout(60)
    def parse_and_apply_templates(self, policy, source_deploy):
        yaml_dict = {}

        yaml_dict.update(self.template_helper.load_policy_apply_yaml(policy))

        service_dict, source_deploy = self.extract_service_from_source_deployment(source_deploy)
        yaml_dict.update({'processor': self.template_helper.load_application_apply_yaml(service_dict)})

        first_stage_components = ['scheduler', 'distributor', 'monitor', 'controller']
        second_stage_components = ['generator', 'processor']

        LOGGER.info(f'[First Deployment Stage] deploy components:{first_stage_components}')
        first_docs_list = self.template_helper.finetune_yaml_parameters(yaml_dict, source_deploy,
                                                                        scopes=first_stage_components)
        result, msg = self.install_yaml_templates(first_docs_list)
        if not result:
            return False, msg

        LOGGER.info(f'[Second Deployment Stage] deploy components:{second_stage_components}')
        second_docs_list = self.template_helper.finetune_yaml_parameters(yaml_dict, source_deploy,
                                                                         scopes=second_stage_components)
        result, msg = self.install_yaml_templates(second_docs_list)
        if not result:
            return False, msg

        self.save_component_yaml(first_docs_list + second_docs_list)

        return True, 'Install services successfully'

    def install_yaml_templates(self, yaml_docs):
        if not yaml_docs:
            return False, 'components yaml data is empty'
        _result = KubeHelper.apply_custom_resources(yaml_docs)
        while not KubeHelper.check_pods_running(self.namespace):
            time.sleep(1)
        return _result, '' if _result else 'kubernetes api error'

    def save_component_yaml(self, docs_list):
        self.cur_yaml_docs = docs_list
        YamlOps.write_all_yaml(docs_list, self.save_yaml_path)

    def extract_service_from_source_deployment(self, source_deploy):
        service_dict = {}

        for s in source_deploy:
            dag = s['dag']
            node_set = s['node_set']
            extracted_dag = copy.deepcopy(dag)
            del extracted_dag['_start']

            def get_service_callback(node_item):
                service_id = node_item['id']
                service = self.find_service_by_id(service_id)
                service_name = service['service']
                service_yaml = service['yaml']
                if service_id in service_dict:
                    pre_node_list = service_dict[service_id]['node']
                    service_dict[service_id]['node'] = list(set(pre_node_list + node_set))
                else:
                    service_dict[service_id] = {'service_name': service_name, 'yaml': service_yaml, 'node': node_set}

                extracted_dag[node_item['id']]['service'] = service

            self.bfs_dag(dag, get_service_callback)
            s['dag'] = extracted_dag

        return service_dict, source_deploy

    def get_yaml_docs(self):
        if self.cur_yaml_docs:
            return self.cur_yaml_docs
        elif os.path.exists(self.save_yaml_path):
            return YamlOps.read_all_yaml(self.save_yaml_path)
        else:
            return None

    def clear_yaml_docs(self):
        self.cur_yaml_docs = None
        FileOps.remove_file(self.save_yaml_path)

    def find_service_by_id(self, service_id):
        for service in self.services:
            if service['id'] == service_id:
                return service
        return None

    def find_dag_by_id(self, dag_id):
        for dag in self.dags:
            if dag['dag_id'] == dag_id:
                return dag['dag']
        return None

    def find_scheduler_policy_by_id(self, policy_id):
        for policy in self.schedulers:
            if policy['id'] == policy_id:
                return policy
        return None

    def find_datasource_configuration_by_label(self, source_label):
        for source_config in self.source_configs:
            if source_config['source_label'] == source_label:
                return source_config
        return None

    def fill_datasource_config(self, config):
        config['source_label'] = f'source_config_{Counter().get_count("source_label")}'
        source_list = config['source_list']
        for index, source in enumerate(source_list):
            source['id'] = index
            source['url'] = self.fill_datasource_url(source['url'], config['source_type'], config['source_mode'], index)

        config['source_list'] = source_list
        return config

    def fill_datasource_url(self, url, source_type, source_mode, source_id):
        if not self.inner_datasource:
            return url
        source_hostname = KubeHelper.get_pod_node(SystemConstant.DATASOURCE.value, self.namespace)
        if not source_hostname:
            assert None, 'Datasource pod not exists.'
        source_protocol = source_mode.split('_')[0]
        source_ip = NodeInfo.hostname2ip(source_hostname)
        source_port = PortInfo.get_component_port(SystemConstant.DATASOURCE.value)
        url = f'{source_protocol}://{source_ip}:{source_port}/{source_type}{source_id}'

        return url

    @staticmethod
    def bfs_dag(dag_graph, dag_callback):
        source_list = dag_graph['_start']
        queue = deque(source_list)
        visited = set(source_list)
        while queue:
            current_node_item = dag_graph[queue.popleft()]
            dag_callback(current_node_item)
            for child_id in current_node_item['succ']:
                if child_id not in visited:
                    queue.append(child_id)
                    visited.add(child_id)

    @staticmethod
    def check_node_exist(node):
        return node in NodeInfo.get_node_info()

    @staticmethod
    def get_edge_nodes():
        def sort_key(item):
            name = item['name']
            patterns = [
                (r'^edge(\d+)$', 0),
                (r'^edgexn(\d+)$', 1),
                (r'^edgex(\d+)$', 2),
                (r'^edgen(\d+)$', 3),
            ]
            for pattern, group in patterns:
                match = re.match(pattern, name)
                if match:
                    num = int(match.group(1))
                    return group, num
            return len(patterns), 0

        node_role = NodeInfo.get_node_info_role()
        edge_nodes = [{'name': node_name} for node_name in node_role if node_role[node_name] == 'edge']
        edge_nodes.sort(key=sort_key)
        return edge_nodes

    def check_simulation_datasource(self):
        return KubeHelper.check_pod_name('datasource', namespace=self.namespace)

    def check_dag(self, dag, modal='frame'):

        def topo_sort(graph, modal):
            in_degree = {}
            for node in graph.keys():
                if node != '_start':
                    in_degree[node] = len(graph[node]['prev'])
            queue = copy.deepcopy(graph['_start'])
            topo_order = []

            for node in queue:
                node_service = self.find_service_by_id(node)
                if node_service['input'] != modal:
                    error_msg = f"Node '{node}' has invalid input mode, expected '{modal}' got '{node_service['input']}'"
                    LOGGER.warning(f"DAG Validation Error: {error_msg}")
                    return False, error_msg

            while queue:
                parent = queue.pop(0)
                topo_order.append(parent)
                for child in graph[parent]['succ']:
                    parent_service = self.find_service_by_id(parent)
                    child_service = self.find_service_by_id(child)
                    if not parent_service or not child_service:
                        error_msg = f"Missing service definition for node {parent if not parent_service else child}"
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg
                    if child_service['input'] != parent_service['output']:
                        error_msg = (
                            f"Node connection mismatch, '{parent}' output '{parent_service['output']}', '{child}' input '{child_service['input']}' "
                        )
                        LOGGER.error(f"DAG Validation Error: {error_msg}")
                        return False, error_msg

                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

            if len(topo_order) != len(in_degree):
                error_msg = "DAG contains cycles or unreachable nodes"
                LOGGER.warning(f"DAG Validation Error: {error_msg}")
                return False, error_msg

            return True, "DAG validation passed"

        return topo_sort(dag.copy(), modal)

    def get_source_ids(self):
        source_ids = []
        source_config = self.find_datasource_configuration_by_label(self.source_label)
        if not source_config:
            return []
        for source in source_config['source_list']:
            source_ids.append(source['id'])

        return source_ids

    def run_get_result(self):
        time_ticket = 0
        while self.is_get_result:
            time.sleep(1)
            self.get_result_url()
            if not self.result_url:
                LOGGER.debug('[NO RESULT] Fetch result url failed.')
                continue
            response = http_request(self.result_url,
                                    method=NetworkAPIMethod.DISTRIBUTOR_RESULT,
                                    json={'time_ticket': time_ticket, 'size': 0})

            if not response:
                self.result_url = None
                self.result_file_url = None
                LOGGER.debug('[NO RESULT] Request result url failed.')
                continue

            time_ticket = response["time_ticket"]
            LOGGER.debug(f'time ticket: {time_ticket}')
            results = response['result']
            for result in results:
                if result is None or result == '':
                    continue

                task = Task.deserialize(result)

                source_id = task.get_source_id()
                task_id = task.get_task_id()
                file_path = self.get_file_result(task.get_file_path())
                LOGGER.debug(task.get_delay_info())
                visualization_data = {}

                try:
                    for idx, vf in enumerate(self.visualizations):
                        al_name = vf['hook_name']
                        al_params = eval(vf['params']) if 'params' in vf else {}
                        vf_func = Context.get_algorithm('VISUALIZER', al_name=al_name, **al_params)
                        visualization_data[idx] = vf_func(task)
                except Exception as e:
                    LOGGER.warning(f'Prepare visualization data failed: {str(e)}')
                    LOGGER.exception(e)
                    continue

                if os.path.exists(file_path):
                    os.remove(file_path)

                if not self.source_open:
                    break

                self.task_results[source_id].put_all([{
                    'task_id': task_id,
                    'data': visualization_data,
                }])

    def check_datasource_config(self, config_path):
        if not YamlOps.is_yaml_file(config_path):
            return None

        config = YamlOps.read_yaml(config_path)
        try:
            source_name = config['source_name']
            source_type = config['source_type']
            source_mode = config['source_mode']
            for camera in config['source_list']:
                name = camera['name']
                if self.inner_datasource:
                    directory = camera['dir']
                else:
                    url = camera['url']
                metadata = camera['metadata']

        except KeyError:
            return None

        return config

    def get_resource_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        except AssertionError:
            return
        self.resource_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_GET_RESOURCE)

    def get_result_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            distributor_port = PortInfo.get_component_port(SystemConstant.DISTRIBUTOR.value)
        except AssertionError:
            return
        self.result_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                        port=distributor_port,
                                        path=NetworkAPIPath.DISTRIBUTOR_RESULT)
        self.result_file_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                             port=distributor_port,
                                             path=NetworkAPIPath.DISTRIBUTOR_FILE)

    def get_log_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            distributor_port = PortInfo.get_component_port(SystemConstant.DISTRIBUTOR.value)
        except AssertionError:
            return
        self.log_fetch_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                           port=distributor_port,
                                           path=NetworkAPIPath.DISTRIBUTOR_ALL_RESULT)
        self.log_clear_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                           port=distributor_port,
                                           path=NetworkAPIPath.DISTRIBUTOR_CLEAR_DATABASE)

    def get_file_result(self, file_name):
        if not self.result_file_url:
            return ''
        response = http_request(self.result_file_url,
                                method=NetworkAPIMethod.DISTRIBUTOR_FILE,
                                no_decode=True,
                                json={'file': file_name},
                                stream=True)
        if response is None:
            self.result_file_url = None
            return ''
        with open(file_name, 'wb') as file_out:
            for chunk in response.iter_content(chunk_size=8192):
                file_out.write(chunk)
        return file_name

    def download_log_file(self):
        self.parse_base_info()
        self.get_log_url()
        if not self.log_fetch_url:
            return ''

        response = http_request(self.log_fetch_url, method=NetworkAPIMethod.DISTRIBUTOR_ALL_RESULT, )
        if response is None:
            self.log_fetch_url = None
            return ''
        results = response['result']

        http_request(self.log_clear_url, method=NetworkAPIMethod.DISTRIBUTOR_CLEAR_DATABASE)

        return results

    def get_visualization_config(self):
        self.parse_base_info()
        return [{'id': idx, **vf} for idx, vf in enumerate(self.visualizations)]
