from kubernetes import client, config
import psutil
import pytz

from core.lib.common import LOGGER, YamlOps


class KubeHelper:
    @staticmethod
    def apply_custom_resources(docs):
        config.load_incluster_config()
        api_instance = client.CustomObjectsApi()

        for doc in docs:
            if doc is None:
                continue
            group = doc['apiVersion'].split('/')[0]
            version = doc['apiVersion'].split('/')[-1]
            namespace = doc['metadata']['namespace']
            plural = KubeHelper.get_crd_plural(doc['kind'])

            try:
                api_response = api_instance.create_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    body=doc
                )

                LOGGER.info(f"Created {doc['kind']} named {doc['metadata']['name']} in {namespace} namespace.")
            except Exception as e:
                LOGGER.exception(e)
                return False
        return True

    @staticmethod
    def apply_custom_resources_by_file(yaml_file_path):
        docs = YamlOps.read_all_yaml(yaml_file_path)
        return KubeHelper.apply_custom_resources(docs)

    @staticmethod
    def delete_custom_resources(docs):
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        apps_v1 = client.AppsV1Api()

        for doc in docs:
            if doc is None:
                continue
            group = doc['apiVersion'].split('/')[0]
            version = doc['apiVersion'].split('/')[-1]
            kind = doc['kind']
            namespace = doc['metadata']['namespace']
            name = doc['metadata']['name']
            node_name_config = doc['spec']['serviceConfig'] if 'serviceConfig' in doc['spec'] else None

            try:
                # Delete Service
                if node_name_config:
                    svc_name = f"{name}-{node_name_config['pos']}"
                    LOGGER.info(f"Deleting Service: {name} in namespace {namespace}")
                    v1.delete_namespaced_service(name=svc_name, namespace=namespace)

                # Delete Deployments
                deployments = apps_v1.list_namespaced_deployment(namespace)
                for dep in deployments.items:
                    if dep.metadata.name.startswith(name):
                        LOGGER.info(f"Deleting Deployment: {dep.metadata.name} in namespace {namespace}")
                        apps_v1.delete_namespaced_deployment(name=dep.metadata.name, namespace=namespace)

                # Delete Custom Resource
                custom_api = client.CustomObjectsApi()
                custom_api.delete_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=KubeHelper.get_crd_plural(kind),
                    name=name
                )
                LOGGER.info(f"Deleting Custom Resource: {name} in namespace {namespace}")

            except Exception as e:
                LOGGER.exception(e)
                return False
        return True

    @staticmethod
    def delete_custom_resources_by_file(yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            docs = YamlOps.read_all_yaml(file)

        return KubeHelper.delete_custom_resources(docs)

    @staticmethod
    def update_custom_resources(docs):
        config.load_incluster_config()
        api_instance = client.CustomObjectsApi()

        for doc in docs:
            if not doc:
                continue
            group, version = doc['apiVersion'].split('/')
            namespace = doc['metadata']['namespace']
            plural = KubeHelper.get_crd_plural(doc['kind'])
            name = doc['metadata']['name']

            try:
                # try to get existing resource
                existing = api_instance.get_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name
                )

                # keep version data
                doc['metadata']['resourceVersion'] = existing['metadata']['resourceVersion']

                api_instance.replace_namespaced_custom_object(
                    group=group,
                    version=version,
                    namespace=namespace,
                    plural=plural,
                    name=name,
                    body=doc
                )
                LOGGER.info(f"Updated {doc['kind']}/{name} in {namespace}")

            except client.rest.ApiException as e:
                if e.status == 404:
                    # create resource directly if resource does not exist
                    api_instance.create_namespaced_custom_object(
                        group=group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                        body=doc
                    )
                    LOGGER.info(f"Created {doc['kind']}/{name} in {namespace}")
                else:
                    LOGGER.exception(f"API error: {e}")
                    return False
            except Exception as e:
                LOGGER.exception(f"Unexpected error: {e}")
                return False
        return True

    @staticmethod
    def update_custom_resources_by_file(yaml_file_path):
        with open(yaml_file_path, 'r') as file:
            docs = YamlOps.read_all_yaml(file)

        return KubeHelper.update_custom_resources(docs)

    @staticmethod
    def check_pods_running(namespace):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        pods = v1.list_namespaced_pod(namespace)

        all_running = True
        for pod in pods.items:
            if pod.status.phase != "Running" or not all([c.ready for c in pod.status.container_statuses]):
                all_running = False
                break

        return all_running

    @staticmethod
    def check_specific_pods_running(namespace, specific_pods):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        pods = v1.list_namespaced_pod(namespace)
        all_running = True
        for pod in pods.items:
            if any(specific_pod in pod.metadata.name for specific_pod in specific_pods):
                if pod.status.phase != "Running" or not all([c.ready for c in pod.status.container_statuses]):
                    all_running = False
                    break

        return all_running

    @staticmethod
    def check_specific_pods_exist(namespace: str, include_pods: list = None, exclude_pods: list = None):
        if include_pods is None:
            include_pods = []
        if exclude_pods is None:
            exclude_pods = []
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace)
        for pod in pods.items:
            if any(include_pod in pod.metadata.name for include_pod in include_pods) and \
                    not any(exclude_pod in pod.metadata.name for exclude_pod in exclude_pods):
                return True
        return False

    @staticmethod
    def check_pods_exist(namespace):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        pods = v1.list_namespaced_pod(namespace)
        return len(pods.items) > 0

    @staticmethod
    def check_pod_name(name, namespace):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        pods = v1.list_namespaced_pod(namespace)
        for pod in pods.items:
            if name in pod.metadata.name:
                return True
        return False

    @staticmethod
    def get_pod_node(name, namespace):
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        pods = v1.list_namespaced_pod(namespace)
        for pod in pods.items:
            if name in pod.metadata.name:
                return pod.spec.node_name
        return None

    @staticmethod
    def get_service_info(service_name, namespace):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        api = client.CustomObjectsApi()
        cpu_usage = api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods"
        )
        cpu_dict = {}
        mem_dict = {}

        for pod in cpu_usage.get('items', []):
            pod_name = pod['metadata']['name']
            if service_name in pod_name:
                container = pod.get('containers')[0]
                cpu_dict[pod_name] = int(container['usage']['cpu'][:-1]) / 1000000 / 1000
                mem_dict[pod_name] = int(container['usage']['memory'][:-2]) * 1024

        info = []

        pods = v1.list_namespaced_pod(namespace)
        for pod in pods.items:
            if service_name in pod.metadata.name:
                cpu_usage = f'{cpu_dict[pod.metadata.name] / KubeHelper.get_node_cpu(pod.spec.node_name) * 100:.2f}%' if pod.metadata.name in cpu_dict else ''
                mem_usage = f'{mem_dict[pod.metadata.name] / psutil.virtual_memory().total * 100:.2f}%' if pod.metadata.name in mem_dict else ''

                info_dict = {'age': pod.metadata.creation_timestamp.astimezone(pytz.timezone('Asia/Shanghai')).strftime(
                    '%Y-%m-%d %H:%M:%S'),
                    'hostname': pod.spec.node_name,
                    'ip': KubeHelper.get_node_ip(pod.spec.node_name),
                    'cpu': cpu_usage,
                    'memory': mem_usage,
                    'bandwidth': ''}
                info.append(info_dict)

        return info

    @staticmethod
    def get_node_ip(hostname):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        nodes = v1.list_node()
        for node in nodes.items:
            if node.metadata.name == hostname:
                for address in node.status.addresses:
                    if address.type == "InternalIP":
                        return address.address
        return ''

    @staticmethod
    def get_node_cpu(hostname):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        nodes = v1.list_node()
        for node in nodes.items:
            if node.metadata.name == hostname:
                return int(node.status.capacity['cpu'][-1])

        raise Exception(f'hostname of {hostname} not exists')

    @staticmethod
    def create_namespace(namespace_name):
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        namespace = client.V1Namespace(
            metadata=client.V1ObjectMeta(name=namespace_name)
        )

        try:
            api_response = v1.create_namespace(body=namespace)
            LOGGER.info(f"Namespace {namespace_name} created. Status: {api_response.status}")
        except Exception as e:
            LOGGER.exception(f"Exception when calling CoreV1Api->create_namespace: {e}")

    @staticmethod
    def delete_namespace(namespace_name):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        try:
            api_response = v1.delete_namespace(name=namespace_name)
            LOGGER.info(f"Namespace {namespace_name} deleted. Status: {api_response.status}")
        except Exception as e:
            LOGGER.exception(f"Exception when calling CoreV1Api->delete_namespace: {e}")

    @staticmethod
    def list_namespaces(filter_name=None):
        config.load_incluster_config()
        v1 = client.CoreV1Api()

        try:
            namespace_list = []
            namespaces = v1.list_namespace()
            for ns in namespaces.items:
                if filter_name is None or filter_name in ns.metadata.name:
                    namespace_list.append(ns.metadata.name)
            return namespace_list
        except client.exceptions.ApiException as e:
            LOGGER.exception(f"Exception when calling CoreV1Api->list_namespace: {e}")
            return []

    @staticmethod
    def get_crd_plural(crd_kind):
        config.load_incluster_config()
        api_instance = client.ApiextensionsV1Api()
        crds = api_instance.list_custom_resource_definition()
        for crd in crds.items:
            if crd.spec.names.kind == crd_kind:
                return crd.spec.names.plural
        assert None, f'Crd kind {crd_kind} not exists.'

    @staticmethod
    def get_kubernetes_endpoint():
        config.load_incluster_config()
        v1 = client.CoreV1Api()
        endpoints = v1.read_namespaced_endpoints(name='kubernetes', namespace='default')
        subset = endpoints.subsets[0]
        address = subset.addresses[0].ip
        port = subset.ports[0].port
        return {
            'address': address,
            'port': port
        }
