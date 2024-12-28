import cv2
import numpy as np
import os
import time
from core.lib.content import Task
from core.lib.common import LOGGER, Context, YamlOps, FileOps, Counter, SystemConstant, EncodeOps
from core.lib.network import http_request, NodeInfo, PortInfo, get_merge_address, NetworkAPIPath, NetworkAPIMethod

from kube_helper import KubeHelper
from template_helper import TemplateHelper
from utils import get_first_frame_from_video, draw_bboxes


class BackendCore:
    def __init__(self):

        self.template_helper = TemplateHelper(Context.get_file_path(0))

        self.namespace = ''
        self.image_meta = None
        self.schedulers = None
        self.services = None

        self.source_configs = []
        self.pipelines = []

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
        except KeyError as e:
            LOGGER.warning(f'Parse base info failed: {str(e)}')

    def get_log_file_name(self):
        base_info = self.template_helper.load_base_info()
        load_file_name = base_info['log-file-name']
        if not load_file_name:
            return None
        return load_file_name.split('.')[0]

    def parse_apply_templates(self, policy, source_deploy):
        yaml_dict = {}

        yaml_dict.update(self.template_helper.load_policy_apply_yaml(policy))

        service_dict, source_deploy = self.extract_service_from_source_deployment(source_deploy)
        yaml_dict.update({'processor': self.template_helper.load_application_apply_yaml(service_dict)})

        docs_list = self.template_helper.finetune_yaml_parameters(yaml_dict, source_deploy)

        self.cur_yaml_docs = docs_list
        YamlOps.write_all_yaml(docs_list, self.save_yaml_path)

        return docs_list

    def extract_service_from_source_deployment(self, source_deploy):
        service_dict = {}

        for s in source_deploy:
            pipeline = s['pipeline']
            node = s['node']
            extracted_pipeline = []
            for service_id in pipeline:
                service = self.find_service_by_id(service_id)
                service_name = service['service']
                service_yaml = service['yaml']
                extracted_pipeline.append(service)
                if service_id in service_dict:
                    service_dict[service_id]['node'].append(node)
                else:
                    service_dict[service_id] = {'service_name': service_name, 'yaml': service_yaml, 'node': [node]}
            s['pipeline'] = extracted_pipeline
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

    def find_pipeline_by_id(self, dag_id):
        for pipeline in self.pipelines:
            if pipeline['dag_id'] == dag_id:
                return pipeline['dag']
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
    def check_node_exist(node):
        return node in NodeInfo.get_node_info()

    @staticmethod
    def get_edge_nodes():
        node_role = NodeInfo.get_node_info_role()
        edge_nodes = [{'name': node_name} for node_name in node_role if node_role[node_name] == 'edge']
        return edge_nodes

    def check_simulation_datasource(self):
        return KubeHelper.check_pod_name('datasource', namespace=self.namespace)

    def check_pipeline(self, pipeline, initial_form='frame'):
        last_data_form = initial_form
        for custom_service_id in pipeline:
            custom_service = self.find_service_by_id(custom_service_id)
            if custom_service and custom_service['input'] == last_data_form:
                last_data_form = custom_service['output']
            else:
                return False
        return True

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
                delay = task.calculate_total_time()
                LOGGER.debug(task.get_delay_info())

                try:
                    task_result = float(np.mean(task.get_scenario_data()['obj_num']))
                except Exception as e:
                    LOGGER.debug(f'scenario: {task.get_scenario_data()}')
                    LOGGER.warning(f'Scenario fetch failed: {str(e)}')
                    continue

                content = task.get_content()
                file_path = self.get_file_result(task.get_file_path())

                try:
                    image = get_first_frame_from_video(file_path)
                    image = draw_bboxes(image, content[0][0])

                    base64_data = EncodeOps.encode_image(image)
                except Exception as e:
                    base64_data = EncodeOps.encode_image(
                        cv2.imread(self.default_visualization_image)
                    )
                    LOGGER.warning(f'Video visualization fetch failed: {str(e)}')
                if os.path.exists(file_path):
                    os.remove(file_path)

                if not self.source_open:
                    break

                self.task_results[source_id].put_all([{
                    'taskId': task_id,
                    'result': task_result,
                    'delay': delay,
                    'visualize': base64_data
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
        self.resource_url = get_merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                              port=scheduler_port,
                                              path=NetworkAPIPath.SCHEDULER_GET_RESOURCE)

    def get_result_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            distributor_port = PortInfo.get_component_port(SystemConstant.DISTRIBUTOR.value)
        except AssertionError:
            return
        self.result_url = get_merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                            port=distributor_port,
                                            path=NetworkAPIPath.DISTRIBUTOR_RESULT)
        self.result_file_url = get_merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                                 port=distributor_port,
                                                 path=NetworkAPIPath.DISTRIBUTOR_FILE)

    def get_log_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            distributor_port = PortInfo.get_component_port(SystemConstant.DISTRIBUTOR.value)
        except AssertionError:
            return
        self.log_fetch_url = get_merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                               port=distributor_port,
                                               path=NetworkAPIPath.DISTRIBUTOR_ALL_RESULT)
        self.log_clear_url = get_merge_address(NodeInfo.hostname2ip(cloud_hostname),
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
