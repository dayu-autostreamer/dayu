import copy
import re
import json
import threading
from collections import deque
from func_timeout import func_set_timeout as timeout
import func_timeout.exceptions as timeout_exceptions

import os
import time
from core.lib.content import Task
from core.lib.common import LOGGER, Context, YamlOps, FileOps, Counter, SystemConstant, TaskConstant, \
    ConfigBoundInstanceCache
from core.lib.network import http_request, NodeInfo, PortInfo, merge_address, NetworkAPIPath, NetworkAPIMethod
from core.lib.estimation import Timer

from kube_helper import KubeHelper
from template_helper import TemplateHelper


class BackendCore:
    def __init__(self):

        self.template_helper = TemplateHelper(Context.get_file_path(0))

        self.namespace = ''
        self.image_meta = None
        self.schedulers = None
        self.services = None

        self.result_visualization_configs = None
        self.system_visualization_configs = None
        self.customized_source_result_visualization_configs = {}
        self.result_visualization_cache = ConfigBoundInstanceCache(
            factory=lambda vf: Context.get_algorithm(
                'RESULT_VISUALIZER',
                al_name=vf['hook_name'],
                **(dict(eval(vf['hook_params'])) if 'hook_params' in vf else {}),
                variables=vf['variables']
            )
        )
        self.system_visualization_cache = ConfigBoundInstanceCache(
            factory=lambda vf: Context.get_algorithm(
                'SYSTEM_VISUALIZER',
                al_name=vf['hook_name'],
                **(dict(eval(vf['hook_params'])) if 'hook_params' in vf else {}),
                variables=vf['variables']
            )
        )

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
        self.is_cycle_deploy = False

        self.yaml_dict = None
        self.source_deploy = None

        self.cur_yaml_docs = None
        self.save_yaml_path = 'resources.yaml'
        self.log_file_path = 'log.json'
        # File path for system visualization logs (ephemeral download file)
        self.system_log_file_path = 'system_log.json'
        # In-memory system logs storage
        self.system_logs = []
        # Persistent store for system logs (append-only JSON Lines)
        self.system_log_store_path = 'system_log_store.jsonl'
        # Threshold for flushing in-memory logs to file
        self.system_log_threshold = 1000

        self.default_visualization_image = 'default_visualization.png'

        self.system_support_components = ['backend', 'frontend', 'datasource', 'redis']
        self.function_components = ['generator', 'scheduler', 'controller', 'distributor', 'monitor']

        self.parse_base_info()

    def parse_base_info(self):
        try:
            base_info = self.template_helper.load_base_info()
            self.namespace = base_info['namespace']
            self.image_meta = base_info['default-image-meta']
            self.schedulers = base_info['scheduler-policies']
            self.services = base_info['services']
            self.result_visualization_configs = base_info['result-visualizations']
            self.system_visualization_configs = base_info['system-visualizations']
        except KeyError as e:
            LOGGER.warning(f'Parse base info failed: {str(e)}')

    def get_log_file_name(self):
        base_info = self.template_helper.load_base_info()
        load_file_name = base_info['log-file-name']
        if not load_file_name:
            return None
        return load_file_name.split('.')[0]

    def parse_and_apply_templates(self, policy, source_deploy):
        yaml_dict = {}

        yaml_dict.update(self.template_helper.load_policy_apply_yaml(policy))

        service_dict = self.extract_service_from_source_deployment(source_deploy)
        yaml_dict.update({'processor': self.template_helper.load_application_apply_yaml(service_dict)})

        self.yaml_dict = yaml_dict
        self.source_deploy = source_deploy

        first_stage_components = ['scheduler', 'distributor', 'monitor', 'controller']
        second_stage_components = ['generator', 'processor']

        LOGGER.info(f'[First Deployment Stage] deploy components:{first_stage_components}')
        first_docs_list = self.template_helper.finetune_yaml_parameters(copy.deepcopy(yaml_dict),
                                                                        copy.deepcopy(source_deploy),
                                                                        scopes=first_stage_components)
        try:
            result, msg = self.install_yaml_templates(first_docs_list)
        except timeout_exceptions.FunctionTimedOut as e:
            LOGGER.warning(f'Parse and apply templates failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = 'first-stage install timeout after 100 seconds'
        except Exception as e:
            LOGGER.warning(f'Parse and apply templates failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = 'unexpected system error, please refer to logs in backend'
        finally:
            self.save_component_yaml(first_docs_list)
        if not result:
            return False, msg

        LOGGER.info(f'[Second Deployment Stage] deploy components:{second_stage_components}')
        second_docs_list = self.template_helper.finetune_yaml_parameters(copy.deepcopy(yaml_dict),
                                                                         copy.deepcopy(source_deploy),
                                                                         scopes=second_stage_components)
        try:
            result, msg = self.install_yaml_templates(second_docs_list)
        except timeout_exceptions.FunctionTimedOut as e:
            LOGGER.warning(f'Parse and apply templates failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = 'second-stage install timeout after 100 seconds'
        except Exception as e:
            LOGGER.warning(f'Parse and apply templates failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = 'unexpected system error, please refer to logs in backend'
        finally:
            self.save_component_yaml(first_docs_list + second_docs_list)

        if not result:
            return False, msg

        # Start cycle deployment
        self.is_cycle_deploy = True
        threading.Thread(target=self.run_cycle_deploy).start()

        return True, 'Install services successfully'

    def parse_and_delete_templates(self):

        # End cycle deployment
        self.is_cycle_deploy = False

        docs = self.read_component_yaml()
        try:
            result, msg = self.uninstall_yaml_templates(docs)
        except timeout_exceptions.FunctionTimedOut as e:
            msg = 'timeout after 200 seconds'
            result = False
            LOGGER.warning(f'Uninstall services failed: {msg}')
        except Exception as e:
            LOGGER.warning(f'Uninstall services failed: {str(e)}')
            LOGGER.exception(e)
            result = False
            msg = f'unexpected system error, please refer to logs in backend'

        return result, msg

    def parse_and_redeploy_services(self, update_docs):
        original_docs = self.read_component_yaml()
        if not original_docs:
            msg = 'no valid components yaml docs found.'
            LOGGER.warning(msg)
            return False, ''

        _, docs_to_add, docs_to_update, docs_to_delete = self.check_and_update_docs_list(original_docs, update_docs)

        if docs_to_update:
            res, msg = self.update_processors(docs_to_update)
            if not res:
                return False, msg

        if docs_to_add:
            res, msg = self.install_processors(docs_to_add)
            if not res:
                return False, msg

        if docs_to_delete:
            res, msg = self.uninstall_processors(docs_to_delete)
            if not res:
                return False, msg

        return True, ''

    @timeout(200)
    def update_processors(self, yaml_docs):
        yaml_docs = [doc for doc in (yaml_docs or []) if doc['metadata']['name']
                     not in (self.system_support_components + self.function_components)]
        if not yaml_docs:
            return True, 'no processors need to be installed.'

        processors = [doc['metadata']['name'] for doc in yaml_docs]
        LOGGER.info(f'[Redeployment] update processors:{processors}')

        _result = KubeHelper.delete_custom_resources(yaml_docs)
        if not _result:
            return False, 'kubernetes api error.'
        while KubeHelper.check_pods_with_string_exists(self.namespace, include_str_list=processors):
            time.sleep(1)

        _result = KubeHelper.apply_custom_resources(yaml_docs)
        if not _result:
            return False, 'kubernetes api error.'
        while not KubeHelper.check_specific_pods_running(self.namespace, processors):
            time.sleep(1)
        return _result, '' if _result else 'kubernetes api error.'

    @timeout(100)
    def install_processors(self, yaml_docs):
        yaml_docs = [doc for doc in (yaml_docs or []) if doc['metadata']['name']
                     not in (self.system_support_components + self.function_components)]
        if not yaml_docs:
            return True, 'no processors need to be installed.'

        processors = [doc['metadata']['name'] for doc in yaml_docs]
        LOGGER.info(f'[Redeployment] install processors: {processors}')
        _result = KubeHelper.apply_custom_resources(yaml_docs)
        if not _result:
            return False, 'kubernetes api error.'
        while not KubeHelper.check_specific_pods_running(self.namespace, processors):
            time.sleep(1)
        return _result, '' if _result else 'kubernetes api error.'

    @timeout(200)
    def uninstall_processors(self, yaml_docs):
        yaml_docs = [doc for doc in (yaml_docs or []) if doc['metadata']['name']
                     not in (self.system_support_components + self.function_components)]
        if not yaml_docs:
            return True, 'no processors need to be installed.'

        processors = [doc['metadata']['name'] for doc in yaml_docs]
        LOGGER.info(f'[Redeployment] uninstall processors: {processors}')
        _result = KubeHelper.delete_custom_resources(yaml_docs)
        if not _result:
            return False, 'kubernetes api error.'
        while KubeHelper.check_pods_with_string_exists(self.namespace, include_str_list=processors):
            time.sleep(1)
        return _result, '' if _result else 'kubernetes api error'

    @timeout(100)
    def install_yaml_templates(self, yaml_docs):
        if not yaml_docs:
            return False, 'yaml data is lost, fail to install resources'
        _result = KubeHelper.apply_custom_resources(yaml_docs)
        if not _result:
            return False, 'kubernetes api error.'
        while not KubeHelper.check_pods_running(self.namespace):
            time.sleep(1)
        return _result, '' if _result else 'kubernetes api error'

    @timeout(200)
    def uninstall_yaml_templates(self, yaml_docs):
        if not yaml_docs:
            return False, 'yaml docs is lost, fail to delete resources'
        _result = KubeHelper.delete_custom_resources(yaml_docs)
        if not _result:
            return False, 'kubernetes api error.'
        while KubeHelper.check_pods_without_string_exists(self.namespace,
                                                          exclude_str_list=self.system_support_components):
            time.sleep(1)
        return _result, '' if _result else 'kubernetes api error'

    @staticmethod
    def check_and_update_docs_list(original_docs, update_docs):
        """
        Intelligently compares and categorizes Kubernetes resource configurations
        :param original_docs: List of existing resource configurations
        :param update_docs: List of new resource configurations
        :return: Tuple containing:
            - total_docs: Complete merged configuration
            - resources_to_add: Resources to be created
            - resources_to_update: Resources needing updates
            - resources_to_delete: Resources to be deleted
        """
        # Create name-based dictionaries for efficient lookup
        original_dict = {doc['metadata']['name']: doc for doc in original_docs}
        update_dict = {doc['metadata']['name']: doc for doc in update_docs}

        # Initialize change sets
        resources_to_add = []
        resources_to_update = []
        resources_to_delete = []

        # Detect resources to delete (present in original but missing in update)
        for name in list(original_dict.keys()):
            if name not in update_dict:
                resources_to_delete.append(original_dict[name])

        # Detect resources to add or update
        for name, new_doc in update_dict.items():
            if name not in original_dict:
                # New resource found
                resources_to_add.append(new_doc)
                original_dict[name] = new_doc
            else:
                # Compare configuration changes
                old_doc = original_dict[name]
                if BackendCore.has_significant_changes(old_doc, new_doc):
                    resources_to_update.append(new_doc)
                    original_dict[name] = new_doc

        # Generate merged configuration (updated state)
        total_docs = list(original_dict.values())

        return total_docs, resources_to_add, resources_to_update, resources_to_delete

    @staticmethod
    def has_significant_changes(old_doc, new_doc):
        """
        Detects if resource configurations have meaningful differences
        Ignores metadata, status, and non-critical fields
        """
        # Basic type checks
        if old_doc['kind'] != new_doc['kind']:
            LOGGER.debug(f"Kind changed from {old_doc['kind']} to {new_doc['kind']}")
            return True
        if old_doc['apiVersion'] != new_doc['apiVersion']:
            LOGGER.debug(f"API version changed from {old_doc['apiVersion']} to {new_doc['apiVersion']}")
            return True

        # Prepare comparison objects (deepcopy to avoid mutation)
        old_spec = copy.deepcopy(old_doc.get('spec', {}))
        new_spec = copy.deepcopy(new_doc.get('spec', {}))

        # Remove fields that don't trigger redeployment
        for spec in [old_spec, new_spec]:
            # Remove log level fields
            spec.pop('logLevel', None)

            # Process worker configurations
            for worker_type in ['cloudWorker', 'edgeWorker']:
                if worker_type in spec:
                    worker = spec[worker_type]

                    worker_list = worker if isinstance(worker, list) else [worker]

                    for worker_item in worker_list:
                        if not isinstance(worker_item, dict):
                            continue

                        worker_item.pop('file', None)

                        if 'template' in worker and 'spec' in worker['template']:
                            template_spec = worker['template']['spec']
                            # Remove fields that don't require pod recreation
                            template_spec.pop('dnsPolicy', None)
                            template_spec.pop('serviceAccountName', None)
                            template_spec.pop('restartPolicy', None)

                            # Process container configurations
                            for container in template_spec.get('containers', []):
                                # Keep only deployment-critical fields
                                retained_fields = {'image', 'ports', 'nodeName', 'command', 'args'}
                                container_keys = list(container.keys())
                                for key in container_keys:
                                    if key not in retained_fields:
                                        container.pop(key, None)

        # Normalize for comparison
        def normalize_spec(spec):
            """Standardize spec for reliable comparison"""
            # Sort edgeWorker list by nodeName
            if 'edgeWorker' in spec and isinstance(spec['edgeWorker'], list):
                spec['edgeWorker'] = sorted(
                    spec['edgeWorker'],
                    key=lambda x: x.get('template', {}).get('spec', {}).get('nodeName', '')
                )
            return json.dumps(spec, sort_keys=True, default=str)

        # Perform comparison
        old_normalized = normalize_spec(old_spec)
        new_normalized = normalize_spec(new_spec)

        has_changes = old_normalized != new_normalized
        return has_changes

    def save_component_yaml(self, docs_list):
        self.cur_yaml_docs = copy.deepcopy(docs_list)
        YamlOps.write_all_yaml(docs_list, self.save_yaml_path)

    def read_component_yaml(self):
        if self.cur_yaml_docs:
            return copy.deepcopy(self.cur_yaml_docs)
        elif os.path.exists(self.save_yaml_path):
            return YamlOps.read_all_yaml(self.save_yaml_path)
        else:
            return None

    def update_component_yaml(self, update_docs_list):
        original_docs_list = self.read_component_yaml()
        if not original_docs_list:
            raise Exception('No valid components yaml docs found.')
        total_docs, _, _, _ = self.check_and_update_docs_list(original_docs_list, update_docs_list)
        self.save_component_yaml(total_docs)

    def extract_service_from_source_deployment(self, source_deploy):

        def bfs_dag(dag_graph, id_to_name, node_set, extracted_dag, service_dict):
            source_list = dag_graph[TaskConstant.START.value]
            queue = deque(source_list)
            visited = set(source_list)
            while queue:
                current_node = queue.popleft()
                current_node_item = dag_graph[current_node]

                service_id = current_node_item['id']
                service = self.find_service_by_id(service_id)
                service_name = service['service']
                service_yaml = service['yaml']
                id_to_name[service_id] = service_name

                if service_id in service_dict:
                    pre_node_list = service_dict[service_id]['node']
                    service_dict[service_id]['node'] = list(set(pre_node_list + node_set))
                else:
                    service_dict[service_id] = {'service_name': service_name, 'yaml': service_yaml, 'node': node_set}
                extracted_dag[current_node_item['id']]['service'] = service

                for child_id in current_node_item['succ']:
                    if child_id not in visited:
                        queue.append(child_id)
                        visited.add(child_id)

        service_dict = {}

        for s in source_deploy:
            dag = s['dag']
            node_set = s['node_set']
            extracted_dag = copy.deepcopy(dag)
            del extracted_dag[TaskConstant.START.value]

            id_to_name = {}
            bfs_dag(dag, id_to_name, node_set, extracted_dag, service_dict)

            renamed_dag = {}
            for old_key, node in extracted_dag.items():
                old_id = node.get('id', old_key)
                new_key = id_to_name.get(old_id, old_id)

                node_new = copy.deepcopy(node)
                node_new['id'] = new_key
                if 'prev' in node_new:
                    node_new['prev'] = [id_to_name.get(x, x) for x in node_new['prev']]
                if 'succ' in node_new:
                    node_new['succ'] = [id_to_name.get(x, x) for x in node_new['succ']]

                renamed_dag[new_key] = node_new
            s['dag'] = renamed_dag

        return service_dict

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
        config['source_label'] = f'source_config_{Counter.get_count("source_label")}'
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

    def check_install_state(self):
        return 'install' if KubeHelper.check_pods_without_string_exists(
            self.namespace,
            exclude_str_list=self.system_support_components
        ) else 'uninstall'

    def check_simulation_datasource(self):
        return KubeHelper.check_pod_name('datasource', namespace=self.namespace)

    def check_dag(self, dag):

        def topo_sort(graph):
            in_degree = {}
            for node in graph.keys():
                if node != TaskConstant.START.value:
                    in_degree[node] = len(graph[node]['prev'])
            queue = copy.deepcopy(graph[TaskConstant.START.value])
            topo_order = []

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

        return topo_sort(dag.copy())

    def get_source_ids(self):
        source_ids = []
        source_config = self.find_datasource_configuration_by_label(self.source_label)
        if not source_config:
            return []
        for source in source_config['source_list']:
            source_ids.append(source['id'])

        return source_ids

    def prepare_result_visualization_data(self, task, is_last=False):
        source_id = task.get_source_id()
        viz_configs = self.customized_source_result_visualization_configs[source_id] \
            if source_id in self.customized_source_result_visualization_configs else self.result_visualization_configs
        viz_functions = self.result_visualization_cache.sync_and_get(viz_configs, namespace='result_visualizer')

        visualization_data = []
        for idx, (viz_config, viz_func) in enumerate(zip(viz_configs, viz_functions)):
            try:
                if 'save_expense' in viz_config and viz_config['save_expense'] and not is_last:
                    visualization_data.append({"id": idx, "data": {v: None for v in viz_config['variables']}})
                else:
                    visualization_data.append({"id": idx, "data": viz_func(task)})
            except Exception as e:
                LOGGER.warning(f'Failed to load result visualization data: {str(e)}')
                LOGGER.exception(e)

        return visualization_data

    def prepare_system_visualizations_data(self):
        viz_configs = self.system_visualization_configs
        viz_functions = self.system_visualization_cache.sync_and_get(viz_configs, namespace='system_visualizer')

        # Fetch scheduler resources once to avoid duplicate requests in visualizers
        self.get_resource_url()
        resource_snapshot = None
        try:
            if self.resource_url:
                resource_snapshot = http_request(self.resource_url, method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE)
        except Exception as e:
            LOGGER.warning(f'Failed to fetch scheduler resource for system viz: {str(e)}')
            LOGGER.exception(e)

        visualization_data = []
        for idx, viz_func in enumerate(viz_functions):
            try:
                # Prefer passing shared resource snapshot when supported, fallback otherwise
                try:
                    data = viz_func(resource=resource_snapshot)
                except TypeError:
                    data = viz_func()
                visualization_data.append({"id": idx, "data": data})
            except Exception as e:
                LOGGER.warning(f'Failed to load result visualization data: {str(e)}')
                LOGGER.exception(e)

        return visualization_data

    def parse_task_result(self, results):
        for result in results:
            if result is None or result == '':
                continue

            task = Task.deserialize(result)

            source_id = task.get_source_id()
            LOGGER.debug(task.get_delay_info())

            if not self.source_open:
                break

            self.task_results[source_id].put(copy.deepcopy(task))

    def fetch_visualization_data(self, source_id):
        assert source_id in self.task_results, f'Source_id {source_id} not found in task results!'
        tasks = self.task_results[source_id].get_all()
        vis_results = []

        with Timer(f'Visualization preparation for {len(tasks)} tasks'):
            for idx, task in enumerate(tasks):
                file_path = self.get_file_result(task.get_file_path())
                try:
                    visualization_data = self.prepare_result_visualization_data(task, idx == len(tasks) - 1)
                except Exception as e:
                    LOGGER.warning(f'Prepare visualization data failed: {str(e)}')
                    LOGGER.exception(e)
                    continue

                if os.path.exists(file_path):
                    os.remove(file_path)

                vis_results.append({
                    'task_id': task.get_task_id(),
                    'data': visualization_data,
                })

        return vis_results

    def run_get_result(self):
        time_ticket = 0
        while self.is_get_result:
            try:
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
                results = response['result']
                LOGGER.debug(f'Fetch {len(results)} tasks from time ticket: {time_ticket}')
                self.parse_task_result(results)

            except Exception as e:
                LOGGER.warning(f'Unexpected error occurred in getting task result: {str(e)}')
                LOGGER.exception(e)

    def run_cycle_deploy(self):
        time.sleep(5)
        while self.is_cycle_deploy:
            try:
                time.sleep(1)
                if not self.yaml_dict or not self.source_deploy:
                    LOGGER.debug('[Redeployment] Configuration is lacked, cancel redeployment request..')
                    time.sleep(5)
                    continue
                if not KubeHelper.check_pods_running(self.namespace):
                    LOGGER.debug('[Redeployment] Pods is in error state, cancel redeployment request..')
                    time.sleep(5)
                    continue

                redeploy_docs_list = self.template_helper.finetune_yaml_parameters(copy.deepcopy(self.yaml_dict),
                                                                                   copy.deepcopy(self.source_deploy),
                                                                                   scopes=['processor'])

                res, msg = self.parse_and_redeploy_services(redeploy_docs_list)

                if res:
                    self.update_component_yaml(redeploy_docs_list)
                    LOGGER.info(f'[Redeployment] Redeployment succeeded.')
                else:
                    LOGGER.warning(f'[Redeployment] Redeployment failed, {msg}')

            except Exception as e:
                LOGGER.warning(f'[Redeployment] Unexpected error occurred in redeployment: {str(e)}')
                LOGGER.exception(e)

    def _flush_system_logs_to_file(self):
        """Append in-memory logs to persistent JSONL file and clear memory buffer."""
        if not self.system_logs:
            return
        try:
            with open(self.system_log_store_path, 'a', encoding='utf-8') as f:
                for item in self.system_logs:
                    f.write(json.dumps(item, ensure_ascii=False))
                    f.write('\n')
            # Clear in-memory after successful flush
            self.system_logs.clear()
        except Exception as e:
            LOGGER.warning(f'Flush system logs to file failed: {str(e)}')
            LOGGER.exception(e)

    def _load_system_logs_from_file(self):
        """Load all persisted logs from JSONL file into a list."""
        logs = []
        if not os.path.exists(self.system_log_store_path):
            return logs
        try:
            with open(self.system_log_store_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        logs.append(json.loads(line))
                    except Exception:
                        # Skip malformed line
                        continue
        except Exception as e:
            LOGGER.warning(f'Read system log store failed: {str(e)}')
            LOGGER.exception(e)
        return logs

    def get_system_parameters(self):
        # Backend-controlled timestamp and single resource fetch per request
        timestamp = time.strftime('%H:%M:%S', time.localtime())

        data = self.prepare_system_visualizations_data()
        snapshot = {"timestamp": timestamp, "data": data}

        # Append to in-memory system logs; flush to file when threshold is reached
        try:
            self.system_logs.append(copy.deepcopy(snapshot))
            if len(self.system_logs) >= self.system_log_threshold:
                self._flush_system_logs_to_file()
        except Exception as e:
            LOGGER.warning(f'Append system log failed: {str(e)}')
            LOGGER.exception(e)

        return [snapshot]

    def download_system_log_content(self):
        """Return current system logs (file + memory) and clear them afterwards."""
        # Ensure any pending in-memory logs are flushed so we don't lose data
        try:
            self._flush_system_logs_to_file()
        except Exception:
            pass

        # Load all logs from file
        file_logs = self._load_system_logs_from_file()
        # Combine with any remaining in-memory logs (should be empty after flush)
        combined = file_logs + list(self.system_logs)
        # Clear in-memory buffer and delete the store file
        self.system_logs.clear()
        FileOps.remove_file(self.system_log_store_path)
        return combined

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

        except Exception as e:
            LOGGER.warning(f'Datasource config file format error: {str(e)}')
            LOGGER.exception(e)
            return None

        return config

    def check_visualization_config(self, config_path):
        if not YamlOps.is_yaml_file(config_path):
            return None

        config = YamlOps.read_yaml(config_path)

        try:
            for visualization in config:
                viz_name = visualization['name']
                assert isinstance(viz_name, str), '"name" is not a string'
                viz_type = visualization['type']
                assert isinstance(viz_type, str), '"type" is not a string'
                viz_var = visualization['variables']
                assert isinstance(viz_var, list), '"variables" is not a list'
                viz_size = visualization['size']
                assert isinstance(viz_size, int), '"size" is not an integer'
                if 'hook_name' in visualization:
                    assert isinstance(visualization['hook_name'], str), '"hook_name" is not a string'
                if 'hook_params' in visualization:
                    assert isinstance(visualization['hook_params'], str), '"hook_params" is not a string(dict)'
                    assert isinstance(eval(visualization['hook_params']), dict), '"hook_params" is not a string(dict)'
                if 'x_axis' in visualization:
                    assert isinstance(visualization['x_axis'], str), '"x_axis" is not a string'
                if 'y_axis' in visualization:
                    assert isinstance(visualization['y_axis'], str), '"y_axis" is not a string'
            return config
        except Exception as e:
            LOGGER.warning(f'Visualization config file format error: {str(e)}')
            LOGGER.exception(e)
            return None

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

    def get_result_visualization_config(self, source_id):
        self.parse_base_info()
        visualizations = self.customized_source_result_visualization_configs[
            source_id] if source_id in self.customized_source_result_visualization_configs else self.result_visualization_configs
        return [{'id': idx, **vf} for idx, vf in enumerate(visualizations)]

    def get_system_visualization_config(self):
        self.parse_base_info()
        return [{'id': idx, **vf} for idx, vf in enumerate(self.system_visualization_configs)]
