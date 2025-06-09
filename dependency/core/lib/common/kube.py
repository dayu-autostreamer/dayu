from kubernetes import client, config
from collections import defaultdict

from core.lib.common import Context


class KubeConfig:
    _api = None
    NAMESPACE = Context.get_parameter('NAMESPACE')
    SERVICE_PREFIX = 'processor-'

    @classmethod
    def _get_api(cls):
        if not cls._api:
            config.load_incluster_config()
            cls._api = client.CoreV1Api()
        return cls._api

    @classmethod
    def get_service_nodes_dict(cls):
        """
        Get nodes for each service based on pod name pattern
        Returns:
            {
                service1: [node1, node2],
                service2: [node3],
                ...,
            }
        """
        api = cls._get_api()

        pods = api.list_namespaced_pod(cls.NAMESPACE).items
        service_nodes = defaultdict(set)

        for pod in pods:
            pod_name = pod.metadata.name
            node_name = pod.spec.node_name

            if not node_name or not pod_name.startswith(cls.SERVICE_PREFIX):
                continue

            parts = pod_name.split('-')
            if len(parts) < 1:
                continue
            service_name = '-'.join(parts[1:3])

            service_nodes[service_name].add(node_name)

        return {svc: list(nodes) for svc, nodes in service_nodes.items()}

    @classmethod
    def get_node_services_dict(cls):
        """
        Get services on each node by reversing service-node mapping
        Returns:
            {
                node1: [service1, service2],
                node2: [service1],
                ...,
            }
        """
        service_nodes = cls.get_service_nodes_dict()
        node_services = defaultdict(set)

        for service, nodes in service_nodes.items():
            for node in nodes:
                node_services[node].add(service)

        return {node: list(svcs) for node, svcs in node_services.items()}

    @classmethod
    def get_services_on_node(cls, node_name):
        """
        Get services on specified node
        Args:
            node_name: target node name
        Returns:
            List of service names
        """
        return cls.get_node_services_dict().get(node_name, [])

    @classmethod
    def get_nodes_for_service(cls, service_name):
        """
        Get nodes running specified service
        Args:
            service_name: target service name
        Returns:
            List of node names
        """
        return cls.get_service_nodes_dict().get(service_name, [])

    @classmethod
    def check_services_running(cls):
        """
        Check if all services are running
        Returns:
            state: bool
        """
        api = cls._get_api()

        pods = api.list_namespaced_pod(cls.NAMESPACE).items
        for pod in pods:
            pod_name = pod.metadata.name
            node_name = pod.spec.node_name
            if not node_name or not pod_name.startswith(cls.SERVICE_PREFIX):
                continue

            if pod.status.phase != "Running" or not all([c.ready for c in pod.status.container_statuses]):
                return False

        return True

