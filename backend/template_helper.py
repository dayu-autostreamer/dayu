import copy
import json
import os
import re
import uuid

from kube_helper import KubeHelper

from core.lib.common import YamlOps, LOGGER, SystemConstant, deep_merge, TaskConstant, NameMaintainer
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
        dependency = policy.get('dependency')
        if dependency is None:
            dependency = {
                component: policy[component]
                for component in ('generator', 'controller', 'distributor', 'monitor')
                if component in policy
            }
        for component in dependency:
            yaml_dict.update({
                component: YamlOps.read_yaml(
                    os.path.join(self.templates_dir, component, dependency[component])
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
                {'name': 'GUNICORN_PORT', 'value': str(node_port['port'])}
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

        cloud_mount_files, edge_mount_files = self.resolve_file_mount(file_mount, file_prefix)
        if pos == 'cloud':
            template_doc['spec'].update({
                'cloudWorker': {
                    'template': {'spec': copy.deepcopy(cloud_template)},
                    'logLevel': {'level': log_level},
                    'mounts': cloud_mount_files,
                }
            })
        elif pos == 'edge':
            template_doc['spec'].update({
                'edgeWorker': [{
                    'template': {'spec': copy.deepcopy(edge_template)},
                    'logLevel': {'level': log_level},
                    'mounts': edge_mount_files,
                }]
            })
        elif pos == 'both':
            template_doc['spec'].update({
                'edgeWorker': [{
                    'template': {'spec': copy.deepcopy(edge_template)},
                    'logLevel': {'level': log_level},
                    'mounts': edge_mount_files,
                }],
                'cloudWorker': {
                    'template': {'spec': copy.deepcopy(cloud_template)},
                    'logLevel': {'level': log_level},
                    'mounts': cloud_mount_files,
                }
            })
        else:
            raise ValueError(f"position of {pos} is illegal, only support cloud/edge/both.")

        return template_doc

    def finetune_yaml_parameters(self, yaml_dict, source_deploy, scopes=None, current_docs=None):
        cloud_node = NodeInfo.get_cloud_node()

        docs_list = []
        if not scopes or 'generator' in scopes:
            docs_list.append(self.finetune_generator_yaml(yaml_dict['generator'], source_deploy))

        edge_nodes = self.get_all_selected_edge_nodes(yaml_dict)
        controller_edge_nodes = self.get_controller_target_edge_nodes(source_deploy, edge_nodes, cloud_node)

        if not scopes or 'controller' in scopes:
            docs_list.append(self.finetune_controller_yaml(yaml_dict['controller'], controller_edge_nodes, cloud_node))
        if not scopes or 'distributor' in scopes:
            docs_list.append(self.finetune_distributor_yaml(yaml_dict['distributor'], cloud_node))
        if not scopes or 'scheduler' in scopes:
            docs_list.append(self.finetune_scheduler_yaml(yaml_dict['scheduler'], cloud_node))
        if not scopes or 'monitor' in scopes:
            docs_list.append(self.finetune_monitor_yaml(yaml_dict['monitor'], edge_nodes, cloud_node))
        if not scopes or 'processor' in scopes:
            if current_docs is None:
                docs_list.extend(self.finetune_processor_yaml(yaml_dict['processor'], cloud_node, source_deploy))
            else:
                docs_list.extend(self.finetune_processor_yaml(yaml_dict['processor'], cloud_node, source_deploy,
                                                              current_docs=current_docs, ))

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

        base_info = self.load_base_info()
        log_export = base_info.get('log-export', {})
        result_log_export = log_export.get('result', {})

        cloud_worker_template = yaml_doc['spec']['cloudWorker']
        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node
        new_cloud_worker['template']['spec']['containers'][0]['env'].extend([
            {
                'name': 'RESULT_LOG_RETENTION_RECORDS',
                'value': str(result_log_export.get('retention-records', 0))
            },
            {
                'name': 'RESULT_LOG_RETENTION_PRUNE_INTERVAL',
                'value': str(result_log_export.get('retention-prune-interval', 200))
            },
            {
                'name': 'RESULT_LOG_EXPORT_BATCH_SIZE',
                'value': str(result_log_export.get('batch-size', 500))
            }, ])

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

            image_name = new_edge_worker['template']['spec']['containers'][0]['image']
            jetpack_major = self.get_device_jetpack_major_version(edge_node)
            image_name = self.specify_jetpack_image(image_name, jetpack_major)
            new_edge_worker['template']['spec']['containers'][0]['image'] = image_name
            new_edge_worker['template']['spec']['containers'][0]['env'].append(
                {'name': 'JETPACK', 'value': str(jetpack_major)})

            new_edge_worker['template']['spec']['nodeName'] = edge_node
            edge_workers.append(new_edge_worker)

        new_cloud_worker = copy.deepcopy(cloud_worker_template)
        new_cloud_worker['template']['spec']['nodeName'] = cloud_node

        yaml_doc['spec']['edgeWorker'] = edge_workers
        yaml_doc['spec']['cloudWorker'] = new_cloud_worker

        return yaml_doc

    def get_controller_target_edge_nodes(self, source_deploy, base_edge_nodes, cloud_node):
        controller_edge_nodes = list(dict.fromkeys(base_edge_nodes or []))
        for source_info in source_deploy or []:
            source = source_info.get('source') or {}
            selected_source_node = source.get('source_device')
            if not selected_source_node or selected_source_node == cloud_node:
                continue
            if selected_source_node not in controller_edge_nodes:
                controller_edge_nodes.append(selected_source_node)

        return controller_edge_nodes

    @staticmethod
    def get_processor_service_name(doc):
        if not isinstance(doc, dict):
            return None

        spec = doc.get('spec', {})
        edge_workers = spec.get('edgeWorker') or []
        worker_templates = list(edge_workers)
        if spec.get('cloudWorker'):
            worker_templates.append(spec['cloudWorker'])

        for worker in worker_templates:
            containers = worker.get('template', {}).get('spec', {}).get('containers', [])
            for container in containers:
                for env_item in container.get('env', []):
                    if env_item.get('name') != 'PROCESSOR_SERVICE_NAME':
                        continue
                    service_name = env_item.get('value')
                    if not service_name:
                        continue
                    return service_name[len('processor-'):] if service_name.startswith('processor-') else service_name

        metadata_name = doc.get('metadata', {}).get('name', '')
        if not metadata_name.startswith('processor-'):
            return None

        service_name = metadata_name[len('processor-'):]
        edge_nodes = []
        for worker in edge_workers:
            node_name = worker.get('template', {}).get('spec', {}).get('nodeName')
            if node_name:
                edge_nodes.append(node_name)

        if len(edge_nodes) == 1:
            node_suffix = f"-{NameMaintainer.standardize_device_name(edge_nodes[0])}"
            if service_name.endswith(node_suffix):
                service_name = service_name[:-len(node_suffix)]

        return service_name

    def extract_current_processor_deployment(self, docs_list):
        deployment_plan = {}
        if not docs_list:
            return deployment_plan

        for doc in docs_list:
            if not isinstance(doc, dict):
                continue

            edge_workers = doc.get('spec', {}).get('edgeWorker') or []
            if not edge_workers:
                continue

            service_name = self.get_processor_service_name(doc)
            if not service_name:
                continue

            selected_nodes = deployment_plan.setdefault(service_name, [])
            for worker in edge_workers:
                node_name = worker.get('template', {}).get('spec', {}).get('nodeName')
                if node_name and node_name not in selected_nodes:
                    selected_nodes.append(node_name)

        return deployment_plan

    @staticmethod
    def normalize_deployment_plan(deployment_plan, source_deploy):
        """Normalize scheduler plans to {service_name: [edge_node, ...]}."""
        if not isinstance(deployment_plan, dict):
            return None

        service_names = set()
        edge_nodes = set()
        for source_info in source_deploy or []:
            edge_nodes.update(source_info.get('node_set') or [])
            for service_name in (source_info.get('dag') or {}).keys():
                if service_name != TaskConstant.START.value:
                    service_names.add(service_name)

        normalized_plan = {}
        for key, value in deployment_plan.items():
            values = list(value) if isinstance(value, (list, tuple, set)) else [value]
            if key in service_names:
                selected_nodes = normalized_plan.setdefault(key, [])
                for node_name in values:
                    if node_name in edge_nodes and node_name not in selected_nodes:
                        selected_nodes.append(node_name)
            elif key in edge_nodes:
                for service_name in values:
                    if service_name not in service_names:
                        continue
                    selected_nodes = normalized_plan.setdefault(service_name, [])
                    if key not in selected_nodes:
                        selected_nodes.append(key)

        return normalized_plan

    def finetune_processor_yaml(self, service_dict, cloud_node, source_deploy, current_docs=None):
        """Generate processor CRs with fine-grained units.

        For each logical processor service we generate:
        - One cloud-only CR  with name
          `processor-{service_name}-cloud`.
        - One edge-only CR per edge node with name
          `processor-{service_name}-{edge_node}`.

        This enables independent redeployment for cloud and each edge node
        while keeping the `processor-{service_name}` prefix so that
        service-level queries by prefix still work.
        """
        deployment_plan = self.request_deployment_decision(source_deploy)
        current_deployment_plan = self.extract_current_processor_deployment(
            current_docs) if current_docs is not None else None
        yaml_docs = []

        for service_id, service_info in service_dict.items():
            base_yaml_doc = service_info['service']
            service_name = service_info['service_name']

            # Original candidate edge nodes from DAG / services config
            edge_nodes = service_info['node']

            # Apply scheduler's deployment decision if available.
            if deployment_plan is not None and service_name in deployment_plan:
                # Intersect with original edge_nodes to avoid unexpected nodes.
                selected_nodes = set(deployment_plan[service_name])
                edge_nodes = [node for node in edge_nodes if node in selected_nodes]
            else:
                if current_deployment_plan is not None:
                    edge_nodes = current_deployment_plan.get(service_name, [])
                    LOGGER.warning(
                        f"Scheduler redeployment plan unavailable or missed service '{service_name}', "
                        f"keep current deployment: {edge_nodes}."
                    )
                else:
                    edge_nodes = []
                    LOGGER.warning(
                        f"Scheduler initial deployment plan unavailable or missed service '{service_name}', "
                        "deploy processor on cloud only."
                    )

            # Cloud-only CR
            # Create cloudWorker CR (processor must be deployed on cloud node)
            cloud_component_name = f"processor-{service_name}-{NameMaintainer.standardize_device_name(cloud_node)}"
            cloud_yaml_doc = copy.deepcopy(base_yaml_doc)
            cloud_yaml_doc = self.fill_template(cloud_yaml_doc, cloud_component_name)

            # Configure cloudWorker bound to cloud_node
            if 'cloudWorker' in cloud_yaml_doc['spec'] and cloud_yaml_doc['spec']['cloudWorker']:
                cloud_worker_template = cloud_yaml_doc['spec']['cloudWorker']
                new_cloud_worker = copy.deepcopy(cloud_worker_template)
                new_cloud_worker['template']['spec']['nodeName'] = cloud_node
                new_cloud_worker['template']['spec']['containers'][0]['env'].append({
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
                edge_component_name = f"processor-{service_name}-{NameMaintainer.standardize_device_name(edge_node)}"

                edge_yaml_doc = copy.deepcopy(base_yaml_doc)
                edge_yaml_doc = self.fill_template(edge_yaml_doc, edge_component_name)

                # Configure edgeWorker for this specific node (single worker per CR)
                if 'edgeWorker' in edge_yaml_doc['spec'] and edge_yaml_doc['spec']['edgeWorker']:
                    edge_worker_template = edge_yaml_doc['spec']['edgeWorker'][0]

                    new_edge_worker = copy.deepcopy(edge_worker_template)
                    new_edge_worker['template']['spec']['nodeName'] = edge_node

                    image_name = new_edge_worker['template']['spec']['containers'][0]['image']
                    jetpack_major = self.get_device_jetpack_major_version(edge_node)
                    image_name = self.specify_jetpack_image(image_name, jetpack_major)
                    new_edge_worker['template']['spec']['containers'][0]['image'] = image_name
                    new_edge_worker['template']['spec']['containers'][0]['env'].extend(
                        [{'name': 'PROCESSOR_SERVICE_NAME', 'value': f"processor-{service_name}"},
                         {'name': 'JETPACK', 'value': str(jetpack_major)}])

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

    @staticmethod
    def specify_jetpack_image(image: str, jetpack_major: int) -> str:
        if not jetpack_major or not isinstance(jetpack_major, int) or jetpack_major < 0:
            # return original image if jetpack version is unknown
            return image
        return f'{image}-jp{jetpack_major}'

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

    def resolve_file_mount(self, mount_files, file_prefix=''):
        cloud_mount_files = []
        edge_mount_files = []

        for file_config in mount_files or []:
            mount_config = {}

            if 'pos' not in file_config:
                raise ValueError(f"pos is required for file-mount config, but not found in {file_config}.")
            if 'path' not in file_config:
                raise ValueError(f"path is required for file-mount config, but not found in {file_config}.")

            if 'name' in file_config:
                mount_config['name'] = file_config['name']
            source_path_type = file_config.get('type', 'Directory')
            if source_path_type not in ['Directory', 'DirectoryOrCreate', 'File', 'FileOrCreate',
                                        'Socket', 'CharDevice', 'BlockDevice']:
                raise ValueError(f'Type "{source_path_type}" for file-mount config is illegal, only support Directory '
                                 f'| DirectoryOrCreate | File | FileOrCreate | Socket | CharDevice | BlockDevice')
            mount_config['source'] = {
                'type': 'hostPath',
                'hostPath': {
                    'path': file_config['path'],
                    'pathType': source_path_type,
                    'prefix': file_prefix,
                }
            }

            target = {}
            if 'target_path' in file_config:
                if not os.path.isabs(file_config['target_path']):
                    raise ValueError(f"target_path '{file_config['target_path']}' in "
                                     f"file-mount config should be absolute.")
                target['path'] = file_config['target_path']
            if 'read_only' in file_config:
                target['readOnly'] = file_config['read_only']
            if 'sub_path' in file_config:
                target['subPath'] = file_config['sub_path']
            if 'mount_propagation' in file_config:
                target['mountPropagation'] = file_config['mount_propagation']
            mount_config['target'] = target

            if 'containers' in file_config:
                mount_config['containers'] = file_config['containers']
            if 'env_name' in file_config:
                mount_config['envName'] = file_config['env_name']

            if file_config['pos'] == 'cloud':
                cloud_mount_files.append(copy.deepcopy(mount_config))
            elif file_config['pos'] == 'edge':
                edge_mount_files.append(copy.deepcopy(mount_config))
            elif file_config['pos'] == 'both':
                cloud_mount_files.append(copy.deepcopy(mount_config))
                edge_mount_files.append(copy.deepcopy(mount_config))
            else:
                raise ValueError(f"pos of file-mount should be cloud/edge/both, not {file_config['pos']}.")

        if cloud_mount_files and 'path' not in cloud_mount_files[0]['target'] and 'envName' not in cloud_mount_files[0]:
            cloud_mount_files[0]['envName'] = 'DEFAULT_MOUNT_PATH'
        if edge_mount_files and 'path' not in edge_mount_files[0]['target'] and 'envName' not in edge_mount_files[0]:
            edge_mount_files[0]['envName'] = 'DEFAULT_MOUNT_PATH'

        temp_config = self.resolve_temporary_file_mount(file_prefix)
        cloud_mount_files.append(copy.deepcopy(temp_config))
        edge_mount_files.append(copy.deepcopy(temp_config))

        return cloud_mount_files, edge_mount_files

    def resolve_temporary_file_mount(self, file_prefix=''):
        return {
            'name': 'temporary-directory',
            'source': {
                'type': 'hostPath',
                'hostPath': {
                    'path': 'temp/',
                    'pathType': 'DirectoryOrCreate',
                    'prefix': file_prefix,
                }
            },
            'target': {
                'path': '/temp'
            },
            'envName': 'TEMP_PATH'
        }

    def request_source_selection_decision(self, source_deploy):
        scheduler_hostname = NodeInfo.get_cloud_node()
        all_edge_nodes = NodeInfo.get_all_edge_nodes()
        PortInfo.force_refresh()
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
            params.append({
                "source": SOURCE_ENV,
                "node_set": NODE_SET_ENV,
                "all_edge_nodes": all_edge_nodes,
                "dag": DAG_ENV
            })

        try:
            response = http_request(url=scheduler_address,
                                    method=NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODES,
                                    data={'data': json.dumps(params)},
                                    retry=3)
        except Exception as e:
            LOGGER.warning(f'[Source Node Selection] Error occurred while requesting scheduler: {str(e)}')
            response = None

        if response is None:
            LOGGER.warning('[Source Node Selection] No response from scheduler.')
            selection_plan = None
        elif not isinstance(response, dict) or 'plan' not in response or response['plan'] is None:
            LOGGER.warning('[Source Node Selection] Scheduler response missed selection plan.')
            selection_plan = None
        elif not isinstance(response['plan'], dict):
            LOGGER.warning('[Source Node Selection] Scheduler response contained invalid selection plan.')
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
            scheduler_address = initial_deployment_address
            request_method = NetworkAPIMethod.SCHEDULER_INITIAL_DEPLOYMENT
        else:
            # redeployment
            scheduler_address = redeployment_address
            request_method = NetworkAPIMethod.SCHEDULER_REDEPLOYMENT

        try:
            response = http_request(url=scheduler_address,
                                    method=request_method,
                                    data={'data': json.dumps(params)},
                                    retry=3)
        except Exception as e:
            LOGGER.warning(f'[Service Deployment] Error occurred while requesting scheduler: {str(e)}')
            response = None

        if response is None:
            LOGGER.warning('[Service Deployment] No response from scheduler.')
            deployment_plan = None
        elif not isinstance(response, dict) or 'plan' not in response or response['plan'] is None:
            LOGGER.warning('[Service Deployment] Scheduler response missed deployment plan.')
            deployment_plan = None
        elif not isinstance(response['plan'], dict):
            LOGGER.warning('[Service Deployment] Scheduler response contained invalid deployment plan.')
            deployment_plan = None
        else:
            deployment_plan = self.normalize_deployment_plan(response['plan'], source_deploy)

        return deployment_plan

    def check_is_redeployment(self):
        base_info = self.load_base_info()
        return KubeHelper.check_pods_with_string_exists(base_info['namespace'], include_str_list=['processor'])

    @staticmethod
    def get_device_jetpack_major_version(node_name: str) -> int:
        jetpack_labels = KubeHelper.get_node_jetpack_labels(node_name)
        try:
            if jetpack_labels.get('jetpack_major'):
                return int(jetpack_labels.get('jetpack_major'))
            else:
                return -1
        except Exception as e:
            LOGGER.warning(f'Get Jetpack major version error: {e}')
            LOGGER.exception(e)
            return -1

    @staticmethod
    def get_all_selected_edge_nodes(yaml_dict):
        service_dict = yaml_dict.get('processor', {})
        edge_nodes = set()
        for service_id in service_dict:
            edge_nodes.update(service_dict[service_id]['node'])
        return list(edge_nodes)
