import copy
import json
import os
import re
import uuid

from kube_helper import KubeHelper

from core.lib.common import YamlOps, LOGGER, SystemConstant, deep_merge, TaskConstant
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
            {'name': 'KUBE_CACHE_TTL', 'value': str(base_info['kube-cache-ttl'])},
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

        cloud_template = deep_merge(copy.deepcopy(template), yaml_doc['cloud-pod-template']) \
            if 'cloud-pod-template' in yaml_doc else copy.deepcopy(template)
        edge_template = deep_merge(copy.deepcopy(template), yaml_doc['edge-pod-template']) \
            if 'edge-pod-template' in yaml_doc else copy.deepcopy(template)

        cloud_template = {
            'serviceAccountName': service_account,
            'nodeName': '',
            'dnsPolicy': 'ClusterFirstWithHostNet',
            'containers': [cloud_template]
        }
        edge_template = {
            'serviceAccountName': service_account,
            'nodeName': '',
            'dnsPolicy': 'ClusterFirstWithHostNet',
            'containers': [edge_template]
        }

        if pos == 'cloud':
            files_cloud = [self.prepare_file_path(file['path'])
                           for file in file_mount if file['pos'] in ('cloud', 'both')] \
                if file_mount else []
            files_cloud.append(self.prepare_file_path('temp/'))

            template_doc['spec'].update({
                'cloudWorker': {
                    'template': {'spec': copy.deepcopy(cloud_template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_cloud}}),
                }
            })
        elif pos == 'edge':
            files_edge = [self.prepare_file_path(file['path'])
                          for file in file_mount if file['pos'] in ('edge', 'both')] \
                if file_mount else []
            files_edge.append(self.prepare_file_path('temp/'))

            template_doc['spec'].update({
                'edgeWorker': [{
                    'template': {'spec': copy.deepcopy(edge_template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_edge}}),
                }]
            })
        elif pos == 'both':
            files_cloud = [self.prepare_file_path(file['path'])
                           for file in file_mount if file['pos'] in ('cloud', 'both')] \
                if file_mount else []
            files_cloud.append(self.prepare_file_path('temp/'))
            files_edge = [self.prepare_file_path(file['path'])
                          for file in file_mount if file['pos'] in ('edge', 'both')] \
                if file_mount else []
            files_edge.append(self.prepare_file_path('temp/'))

            template_doc['spec'].update({
                'edgeWorker': [{
                    'template': {'spec': copy.deepcopy(edge_template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_edge}}),
                }],
                'cloudWorker': {
                    'template': {'spec': copy.deepcopy(cloud_template)},
                    'logLevel': {'level': log_level},
                    **({'file': {'paths': files_cloud}}),
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
        selection_plan = self.request_source_selection_decision(source_deploy)

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

            source_info['source'].update({'source_device': node})

            dag = source_info['dag']

            new_edge_worker['template']['spec']['nodeName'] = node

            container = new_edge_worker['template']['spec']['containers'][0]

            container['name'] += str(uuid.uuid4())

            DAG_ENV = {}
            for key in dag.keys():
                temp_node = {}
                if key != TaskConstant.START.value:
                    temp_node['service'] = {'service_name': key}
                    temp_node['prev_nodes'] = dag[key]['prev']
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
            edge_workers.append(new_edge_worker)

        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node

        yaml_doc['spec']['edgeWorker'] = edge_workers
        yaml_doc['spec']['cloudWorker'] = new_cloud_worker

        return yaml_doc

    def finetune_processor_yaml(self, service_dict, cloud_node, source_deploy):
        """Generate processor CRs with fine-grained units.

        For each logical processor service we generate:
        - One cloud-only CR (if cloudWorker template exists) with name
          `processor-{service_name}-cloud`.
        - One edge-only CR per edge node with name
          `processor-{service_name}-{edge_node}`.

        This enables independent redeployment for cloud and each edge node
        while keeping the `processor-{service_name}` prefix so that
        service-level queries by prefix still work.
        """
        deployment_plan = self.request_deployment_decision(source_deploy)
        yaml_docs = []

        for service_id, service_info in service_dict.items():
            base_yaml_doc = service_info['service']
            service_name = service_info['service_name']

            # Original candidate edge nodes from DAG / services config
            edge_nodes = service_info['node']

            # Apply scheduler's deployment decision if available
            if service_name in deployment_plan:
                # Intersect with original edge_nodes to avoid unexpected nodes
                edge_nodes = list(set(deployment_plan[service_name]) & set(edge_nodes))
            else:
                LOGGER.warning(f"Using default service plan for service '{service_id}'.")

            # Cloud-only CR
            # Create cloudWorker CR (processor must be deployed on cloud node)
            cloud_component_name = f"processor-{service_name}-cloud"
            cloud_yaml_doc = copy.deepcopy(base_yaml_doc)
            cloud_yaml_doc = self.fill_template(cloud_yaml_doc, cloud_component_name)

            # Configure cloudWorker bound to cloud_node
            if 'cloudWorker' in cloud_yaml_doc['spec'] and cloud_yaml_doc['spec']['cloudWorker']:
                cloud_worker_template = cloud_yaml_doc['spec']['cloudWorker']
                new_cloud_worker = copy.deepcopy(cloud_worker_template)
                new_cloud_worker['template']['spec']['nodeName'] = cloud_node
                new_cloud_worker['template']['spec']['containers'][0]['env'].extend({
                    'name': 'PROCESSOR_SERVICE_NAME', 'value': f"processor-{service_name}"})
                cloud_yaml_doc['spec']['cloudWorker'] = new_cloud_worker
            else:
                LOGGER.warning(f"Processor service '{service_name}' has no cloudWorker template; skip cloud CR.")

            # Remove edgeWorker from cloud-only CR if present to avoid
            # accidentally scheduling edge workloads here.
            if 'edgeWorker' in cloud_yaml_doc['spec']:
                cloud_yaml_doc['spec'].pop('edgeWorker', None)

            yaml_docs.append(cloud_yaml_doc)

            # If no edge nodes are selected, skip edge CR generation
            if not edge_nodes:
                continue

            # Edge-only CRs per node
            for edge_node in edge_nodes:
                edge_component_name = f"processor-{service_name}-{edge_node}"

                edge_yaml_doc = copy.deepcopy(base_yaml_doc)
                edge_yaml_doc = self.fill_template(edge_yaml_doc, edge_component_name)

                # Configure edgeWorker for this specific node (single worker per CR)
                if 'edgeWorker' in edge_yaml_doc['spec'] and edge_yaml_doc['spec']['edgeWorker']:
                    edge_worker_template = edge_yaml_doc['spec']['edgeWorker'][0]

                    new_edge_worker = copy.deepcopy(edge_worker_template)
                    new_edge_worker['template']['spec']['nodeName'] = edge_node

                    edge_yaml_doc['spec']['edgeWorker'] = [new_edge_worker]
                else:
                    LOGGER.warning(
                        f"Processor service '{service_name}' has no edgeWorker template; skip node {edge_node}."
                    )
                    continue

                # Remove cloudWorker from edge-only CR if present to avoid
                # duplicating cloud workloads into each edge CR.
                if 'cloudWorker' in edge_yaml_doc['spec']:
                    edge_yaml_doc['spec'].pop('cloudWorker', None)

                yaml_docs.append(edge_yaml_doc)

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

    def request_source_selection_decision(self, source_deploy):
        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        scheduler_address = merge_address(NodeInfo.hostname2ip(scheduler_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_SELECT_SOURCE_NODES)

        params = []

        for source_info in source_deploy:
            SOURCE_ENV = source_info['source']
            NODE_SET_ENV = source_info['node_set']
            DAG_ENV = {}
            dag = source_info['dag']

            for key in dag.keys():
                temp_node = {}
                if key != TaskConstant.START.value:
                    temp_node['service'] = {'service_name': key}
                    temp_node['next_nodes'] = dag[key]['succ']
                    DAG_ENV[key] = temp_node
            params.append({"source": SOURCE_ENV, "node_set": NODE_SET_ENV, "dag": DAG_ENV})

        response = http_request(url=scheduler_address,
                                method=NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODES,
                                data={'data': json.dumps(params)},
                                )

        if response is None:
            LOGGER.warning('[Source Node Selection] No response from scheduler.')
            selection_plan = None
        else:
            selection_plan = response['plan']
            selection_plan = {int(k): v for k, v in selection_plan.items()}

        return selection_plan

    def request_deployment_decision(self, source_deploy):
        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        initial_deployment_address = merge_address(NodeInfo.hostname2ip(scheduler_hostname),
                                                   port=scheduler_port,
                                                   path=NetworkAPIPath.SCHEDULER_INITIAL_DEPLOYMENT)
        redeployment_address = merge_address(NodeInfo.hostname2ip(scheduler_hostname),
                                             port=scheduler_port,
                                             path=NetworkAPIPath.SCHEDULER_REDEPLOYMENT)

        params = []
        for source_info in source_deploy:
            SOURCE_ENV = source_info['source']
            NODE_SET_ENV = source_info['node_set']
            DAG_ENV = {}
            dag = source_info['dag']

            for key in dag.keys():
                temp_node = {}
                if key != TaskConstant.START.value:
                    temp_node['service'] = {'service_name': key}
                    temp_node['next_nodes'] = dag[key]['succ']
                    DAG_ENV[key] = temp_node
            params.append({
                "source": SOURCE_ENV,
                "node_set": NODE_SET_ENV,
                "dag": DAG_ENV})

        if not self.check_is_redeployment():
            # initial deployment
            response = http_request(url=initial_deployment_address,
                                    method=NetworkAPIMethod.SCHEDULER_INITIAL_DEPLOYMENT,
                                    data={'data': json.dumps(params)})
        else:
            # redeployment
            response = http_request(url=redeployment_address,
                                    method=NetworkAPIMethod.SCHEDULER_REDEPLOYMENT,
                                    data={'data': json.dumps(params)})

        if response is None:
            LOGGER.warning('[Service Deployment] No response from scheduler.')
            deployment_plan = {}
        else:
            deployment_plan = response['plan']

        return deployment_plan

    def check_is_redeployment(self):
        base_info = self.load_base_info()
        return KubeHelper.check_pods_with_string_exists(base_info['namespace'], include_str_list=['processor'])

    @staticmethod
    def get_all_selected_edge_nodes(yaml_dict):
        service_dict = yaml_dict['processor']
        edge_nodes = set()
        for service_id in service_dict:
            edge_nodes.update(service_dict[service_id]['node'])
        return list(edge_nodes)
