import copy
import json
import os
import re
from kube_helper import KubeHelper

from core.lib.common import YamlOps, LOGGER, SystemConstant
from core.lib.network import NodeInfo, PortInfo, merge_address, NetworkAPIPath, NetworkAPIMethod, http_request


class TemplateHelper:
    def __init__(self, templates_dir):
        self.templates_dir = templates_dir

    def load_base_info(self):
        base_template_path = os.path.join(self.templates_dir, 'base.yaml')
        return YamlOps.read_yaml(base_template_path)

    def load_policy_apply_yaml(self, policy):
        yaml_dict = {'scheduler': YamlOps.read_yaml(
            os.path.join(self.templates_dir, 'scheduler', policy['yaml'])
        )}
        for component in policy['dependency']:
            yaml_dict.update({
                component: YamlOps.read_yaml(
                    os.path.join(self.templates_dir, component, policy['dependency'][component])
                )
            })
        return yaml_dict

    def load_application_apply_yaml(self, service_dict):
        for service_id in service_dict:
            service_dict[service_id]['service'] = YamlOps.read_yaml(
                os.path.join(self.templates_dir, 'processor', service_dict[service_id]['yaml'])
            )
        return service_dict

    def fill_template(self, yaml_doc, component_name):
        base_info = self.load_base_info()
        namespace = base_info['namespace']
        log_level = base_info['log-level']
        service_account = base_info['pod-permission']['service-account']
        file_prefix = base_info['default-file-mount-prefix']
        crd_meta_info = base_info['crd-meta']
        pos = yaml_doc['position']
        template = yaml_doc['pod-template']
        node_port = yaml_doc.get('port-open')
        file_mount = yaml_doc.get('file-mount')
        k8s_endpoint = KubeHelper.get_kubernetes_endpoint()

        template_doc = {
            'apiVersion': crd_meta_info['api-version'],
            'kind': crd_meta_info['kind'],
            'metadata': {'name': component_name, 'namespace': namespace},
            'spec': {}
        }

        if 'env' not in template or template['env'] is None:
            template['env'] = []
        template['env'].extend([
            {'name': 'NAMESPACE', 'value': str(namespace)},
            {'name': 'KUBERNETES_SERVICE_HOST', 'value': str(k8s_endpoint['address'])},
            {'name': 'KUBERNETES_SERVICE_PORT', 'value': str(k8s_endpoint['port'])},

        ])
        template['name'] = component_name
        template['image'] = self.process_image(template['image'])

        if node_port:
            template_doc['spec'].update(
                {'serviceConfig': {'pos': node_port['pos'],
                                   'port': node_port['port'],
                                   'targetPort': node_port['port']}}
            )
            template['env'].extend([
                {'name': 'GUNICORN_PORT', 'value': str(node_port['port'])},
                {'name': 'FILE_PREFIX', 'value': str(file_prefix)}
            ])
            template['ports'] = [{'containerPort': node_port['port']}]

        template = {
            'serviceAccountName': service_account,
            'nodeName': '',
            'dnsPolicy': 'ClusterFirstWithHostNet',
            'containers': [template]
        }

        if pos == 'cloud':
            files_cloud = [self.prepare_file_path(file['path'])
                           for file in file_mount if file['pos'] in ('cloud', 'both')] \
                if file_mount else None

            template_doc['spec'].update({
                'cloudWorker': {
                    'template': {'spec': copy.deepcopy(template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_cloud}} if files_cloud else {}),
                }
            })
        elif pos == 'edge':
            files_edge = [self.prepare_file_path(file['path'])
                          for file in file_mount if file['pos'] in ('edge', 'both')] \
                if file_mount else None

            template_doc['spec'].update({
                'edgeWorker': [{
                    'template': {'spec': copy.deepcopy(template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_edge}} if files_edge else {}),
                }]
            })
        elif pos == 'both':
            files_cloud = [self.prepare_file_path(file['path'])
                           for file in file_mount if file['pos'] in ('cloud', 'both')] \
                if file_mount else None
            files_edge = [self.prepare_file_path(file['path'])
                          for file in file_mount if file['pos'] in ('edge', 'both')] \
                if file_mount else None

            template_doc['spec'].update({
                'edgeWorker': [{
                    'template': {'spec': copy.deepcopy(template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_edge}} if files_edge else {}),
                }],
                'cloudWorker': {
                    'template': {'spec': copy.deepcopy(template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_cloud}} if files_cloud else {}),
                }
            })
        else:
            assert None, f'Unknown position of {pos} (position in [cloud, edge, both]).'

        return template_doc

    def finetune_yaml_parameters(self, yaml_dict, source_deploy, scopes=None):
        edge_nodes = self.get_all_selected_edge_nodes(yaml_dict)
        cloud_node = NodeInfo.get_cloud_node()

        docs_list = []
        if not scopes or 'generator' in scopes:
            docs_list.append(self.finetune_generator_yaml(yaml_dict['generator'], source_deploy))
        if not scopes or 'controller' in scopes:
            docs_list.append(self.finetune_controller_yaml(yaml_dict['controller'], edge_nodes, cloud_node))
        if not scopes or 'distributor' in scopes:
            docs_list.append(self.finetune_distributor_yaml(yaml_dict['distributor'], cloud_node))
        if not scopes or 'scheduler' in scopes:
            docs_list.append(self.finetune_scheduler_yaml(yaml_dict['scheduler'], cloud_node))
        if not scopes or 'monitor' in scopes:
            docs_list.append(self.finetune_monitor_yaml(yaml_dict['monitor'], edge_nodes, cloud_node))
        if not scopes or 'processor' in scopes:
            docs_list.extend(self.finetune_processor_yaml(yaml_dict['processor'], cloud_node, source_deploy))

        return docs_list

    def finetune_generator_yaml(self, yaml_doc, source_deploy):
        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        scheduler_address = merge_address(NodeInfo.hostname2ip(scheduler_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODE)

        params = []

        for source_info in source_deploy:
            SOURCE_ENV = source_info['source']
            NODE_SET_ENV = source_info['node_set']
            DAG_ENV = {}
            dag = source_info['dag']

            for key in dag.keys():
                temp_node = {}
                if key != 'begin':
                    temp_node['service'] = {'service_name': key}
                    temp_node['next_nodes'] = dag[key]['succ']
                    DAG_ENV[key] = temp_node
            params.append({"source": SOURCE_ENV, "node_set": NODE_SET_ENV, "dag": DAG_ENV})

        response = http_request(url=scheduler_address,
                                method=NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODE,
                                data={'data': json.dumps(params)},
                                )

        if response is None:
            LOGGER.warning('[Source Node Selection] No response from scheduler.')
            selection_plan = None
        else:
            selection_plan = response['plan']
            selection_plan = {int(k):v for k, v in selection_plan.items()}

        yaml_doc = self.fill_template(yaml_doc, 'generator')

        edge_worker_template = yaml_doc['spec']['edgeWorker'][0]
        edge_workers_dict = {}
        for source_info in source_deploy:
            new_edge_worker = copy.deepcopy(edge_worker_template)
            source = source_info['source']
            node_set = source_info['node_set']

            if selection_plan is not None and selection_plan[source['id']] is not None:
                node = selection_plan[source['id']]
            else:
                LOGGER.warning("Using default selection plan.")
                node = node_set[0]

            source_info['source'].update({'deploy_node': node})

            dag = source_info['dag']

            new_edge_worker['template']['spec']['nodeName'] = node

            container = new_edge_worker['template']['spec']['containers'][0]

            DAG_ENV = {}
            for key in dag.keys():
                temp_node = {}
                if key != 'begin':
                    temp_node['service'] = {'service_name': key}
                    temp_node['next_nodes'] = dag[key]['succ']
                    DAG_ENV[key] = temp_node

            container['env'].extend(
                [
                    {'name': 'GEN_GETTER_NAME', 'value': str(source['source_mode'])},
                    {'name': 'SOURCE_URL', 'value': str(source['url'])},
                    {'name': 'SOURCE_TYPE', 'value': str(source['source_type'])},
                    {'name': 'SOURCE_ID', 'value': str(source['id'])},
                    {'name': 'SOURCE_METADATA', 'value': str(source['metadata'])},
                    {'name': 'ALL_EDGE_DEVICES', 'value': str(node_set)},
                    {'name': 'DAG', 'value': str(DAG_ENV)},
                ])

            if node in edge_workers_dict:
                edge_workers_dict[node]['template']['spec']['containers'].append(container)
            else:
                new_edge_worker['template']['spec']['containers'] = [container]
                edge_workers_dict[node] = new_edge_worker

        yaml_doc['spec']['edgeWorker'] = list(edge_workers_dict.values())

        return yaml_doc

    def finetune_controller_yaml(self, yaml_doc, edge_nodes, cloud_node):
        yaml_doc = self.fill_template(yaml_doc, 'controller')

        edge_worker_template = yaml_doc['spec']['edgeWorker'][0]
        cloud_worker_template = yaml_doc['spec']['cloudWorker']

        edge_workers = []
        for edge_node in edge_nodes:
            new_edge_worker = copy.deepcopy(edge_worker_template)
            new_edge_worker['template']['spec']['nodeName'] = edge_node
            edge_workers.append(new_edge_worker)

        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node

        yaml_doc['spec']['edgeWorker'] = edge_workers
        yaml_doc['spec']['cloudWorker'] = new_cloud_worker

        return yaml_doc

    def finetune_distributor_yaml(self, yaml_doc, cloud_node):
        yaml_doc = self.fill_template(yaml_doc, 'distributor')

        cloud_worker_template = yaml_doc['spec']['cloudWorker']
        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node

        yaml_doc['spec']['cloudWorker'] = new_cloud_worker

        return yaml_doc

    def finetune_scheduler_yaml(self, yaml_doc, cloud_node):
        yaml_doc = self.fill_template(yaml_doc, 'scheduler')

        cloud_worker_template = yaml_doc['spec']['cloudWorker']
        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node

        yaml_doc['spec']['cloudWorker'] = new_cloud_worker

        return yaml_doc

    def finetune_monitor_yaml(self, yaml_doc, edge_nodes, cloud_node):
        yaml_doc = self.fill_template(yaml_doc, 'monitor')

        edge_worker_template = yaml_doc['spec']['edgeWorker'][0]
        cloud_worker_template = yaml_doc['spec']['cloudWorker']

        edge_workers = []
        for index, edge_node in enumerate(edge_nodes):
            new_edge_worker = copy.deepcopy(edge_worker_template)

            new_edge_worker['template']['spec']['nodeName'] = edge_node

            # only test bandwidth for one edge
            for parameter in new_edge_worker['template']['spec']['containers'][0]['env']:
                if parameter['name'] == 'MONITORS':
                    parameter_list = eval(parameter['value'])
                    if index != 0 and 'bandwidth' in parameter_list:
                        parameter_list.remove('bandwidth')
                        parameter['value'] = str(parameter_list)

            edge_workers.append(new_edge_worker)

        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node

        yaml_doc['spec']['edgeWorker'] = edge_workers
        yaml_doc['spec']['cloudWorker'] = new_cloud_worker

        return yaml_doc

    def finetune_processor_yaml(self, service_dict, cloud_node, source_deploy):
        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        scheduler_address = merge_address(NodeInfo.hostname2ip(scheduler_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_INITIAL_DEPLOYMENT)

        params = []
        for source_info in source_deploy:
            SOURCE_ENV = source_info['source']
            NODE_SET_ENV = source_info['node_set']
            DAG_ENV = {}
            dag = source_info['dag']

            for key in dag.keys():
                temp_node = {}
                if key != 'begin':
                    temp_node['service'] = {'service_name': key}
                    temp_node['next_nodes'] = dag[key]['succ']
                    DAG_ENV[key] = temp_node
            params.append({"source": SOURCE_ENV, "node_set": NODE_SET_ENV, "dag": DAG_ENV})

        response = http_request(url=scheduler_address,
                                method=NetworkAPIMethod.SCHEDULER_INITIAL_DEPLOYMENT,
                                data={'data': json.dumps(params)},
                                )
        if response is None:
            LOGGER.warning('[Service Deployment] No response from scheduler.')
            invert_deployment_plan = {}
        else:
            deployment_plan = response['plan']
            invert_deployment_plan = {}
            for node in deployment_plan:
                if deployment_plan[node]:
                    for service in deployment_plan[node]:
                        if invert_deployment_plan.get(service) is None:
                            invert_deployment_plan[service] = []
                        else:
                            invert_deployment_plan[service].append(node)
        print("invert_deployment_plan:", invert_deployment_plan)
        yaml_docs = []
        for index, service_id in enumerate(service_dict):
            yaml_doc = service_dict[service_id]['yaml']
            service_name = service_dict[service_id]['service_name']
            yaml_doc = self.fill_template(yaml_doc, f'processor-{service_name}')

            edge_nodes = service_dict[service_id]['node']
            if (invert_deployment_plan.get(service_id) is not None and
                    (set(invert_deployment_plan[service_id]) - set(edge_nodes) is not None)):
                edge_worker_template = list(set(invert_deployment_plan[service_id]) & set(edge_nodes))
            else:
                LOGGER.warning("Using default service plan.")
                edge_worker_template = service_dict[service_id]['node']

            cloud_worker_template = yaml_doc['spec']['cloudWorker']

            edge_workers = []
            for edge_node in edge_nodes:
                new_edge_worker = copy.deepcopy(edge_worker_template)
                new_edge_worker['template']['spec']['nodeName'] = edge_node
                edge_workers.append(new_edge_worker)

            new_cloud_worker = copy.deepcopy(cloud_worker_template)
            new_cloud_worker['template']['spec']['nodeName'] = cloud_node

            yaml_doc['spec']['edgeWorker'] = edge_workers
            yaml_doc['spec']['cloudWorker'] = new_cloud_worker

            yaml_docs.append(yaml_doc)

        return yaml_docs

    def process_image(self, image: str) -> str:
        """
            legal input:
                - registry/repository/image:tag
                - registry/repository/image
                - repository/image:tag
                - repository/image
                - image:tag
                - image
            output: complete the full image
        """
        image_meta = self.load_base_info()['default-image-meta']
        default_registry = image_meta['registry']
        default_repository = image_meta['repository']
        default_tag = image_meta['tag']

        pattern = re.compile(
            r"^(?:(?P<registry>[^/]+)/"  # match registry with '/'
            r"(?=.*/)"  # forward pre-check to make sure there is a '/' followed
            r")?"  # registry is optional
            r"(?:(?P<repository>[^/:]+)/)?"  # match repository
            r"(?P<image>[^:]+)"  # match image
            r"(?::(?P<tag>[^:]+))?$"  # match tag
        )

        match = pattern.match(image)
        if not match:
            raise ValueError(f'Format of input image "{image}" is illegal')

        registry = match.group("registry") or default_registry
        repository = match.group("repository") or default_repository
        image_name = match.group("image")
        tag = match.group("tag") or default_tag

        full_image = f"{registry}/{repository}/{image_name}:{tag}"
        return full_image

    def prepare_file_path(self, file_path: str) -> str:
        file_prefix = self.load_base_info()['default-file-mount-prefix']
        return os.path.join(file_prefix, file_path, "")

    @staticmethod
    def get_all_selected_edge_nodes(yaml_dict):
        service_dict = yaml_dict['processor']
        edge_nodes = set()
        for service_id in service_dict:
            edge_nodes.update(service_dict[service_id]['node'])
        return list(edge_nodes)
