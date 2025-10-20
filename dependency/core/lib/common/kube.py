from kubernetes import client, config
from collections import defaultdict
import re
import time
import threading

from core.lib.common import Context


class KubeConfig:
    _api = None
    NAMESPACE = Context.get_parameter('NAMESPACE')
    SERVICE_PATTERN = pattern = re.compile(r"^processor-(.+?)-(?:cloudworker|edgeworker)-")

    # Caching controls
    _cache_lock = threading.Lock()
    _last_refresh_monotonic = 0.0
    _service_nodes_cache = None
    _node_services_cache = None

    # Cache TTL in seconds (set via Context parameter if provided)
    try:
        _ttl_param = Context.get_parameter('KUBE_CACHE_TTL')
        CACHE_TTL = float(_ttl_param) if _ttl_param is not None else 5.0
    except Exception:
        CACHE_TTL = 5.0

    @classmethod
    def _get_api(cls):
        if not cls._api:
            config.load_incluster_config()
            cls._api = client.CoreV1Api()
        return cls._api

    @classmethod
    def _cache_expired(cls) -> bool:
        return (time.monotonic() - cls._last_refresh_monotonic) > cls.CACHE_TTL \
            or cls._service_nodes_cache is None or cls._node_services_cache is None

    @classmethod
    def invalidate_cache(cls):
        """Manually invalidate the internal cache so that the next call refreshes from the API."""
        with cls._cache_lock:
            cls._service_nodes_cache = None
            cls._node_services_cache = None
            cls._last_refresh_monotonic = 0.0

    @classmethod
    def _refresh_cache_if_needed(cls, force: bool = False):
        """Refresh caches from Kubernetes when expired or forced.
        Builds both service->nodes and node->services maps in a single API call.
        Any API error will keep existing caches intact to be resilient.
        """
        if not force and not cls._cache_expired():
            return

        with cls._cache_lock:
            if not force and not cls._cache_expired():
                return  # double-checked locking

            api = cls._get_api()
            try:
                pods = api.list_namespaced_pod(cls.NAMESPACE).items
            except Exception:
                # If API call fails, keep old cache (if any)
                return

            service_nodes = defaultdict(set)
            node_services = defaultdict(set)

            for pod in pods:
                pod_name = getattr(pod.metadata, 'name', None)
                node_name = getattr(pod.spec, 'node_name', None)
                if not pod_name or not node_name:
                    continue

                match = cls.SERVICE_PATTERN.match(pod_name)
                if not match:
                    continue

                service_name = match.group(1)
                service_nodes[service_name].add(node_name)
                node_services[node_name].add(service_name)

            # Store as plain dicts of sets; convert to lists on read to preserve public API
            cls._service_nodes_cache = {svc: set(nodes) for svc, nodes in service_nodes.items()}
            cls._node_services_cache = {node: set(svcs) for node, svcs in node_services.items()}
            cls._last_refresh_monotonic = time.monotonic()

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
        cls._refresh_cache_if_needed()

        # Return a copy with lists to avoid external mutation and keep original behavior
        cache = cls._service_nodes_cache or {}
        return {svc: list(nodes) for svc, nodes in cache.items()}

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
        cls._refresh_cache_if_needed()

        cache = cls._node_services_cache or {}
        return {node: list(svcs) for node, svcs in cache.items()}

    @classmethod
    def get_services_on_node(cls, node_name):
        """
        Get services on specified node
        Args:
            node_name: target node name
        Returns:
            List of service names
        """
        cls._refresh_cache_if_needed()
        cache = cls._node_services_cache or {}
        return list(cache.get(node_name, []))

    @classmethod
    def get_nodes_for_service(cls, service_name):
        """
        Get nodes running specified service
        Args:
            service_name: target service name
        Returns:
            List of node names
        """
        cls._refresh_cache_if_needed()
        cache = cls._service_nodes_cache or {}
        return list(cache.get(service_name, []))

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
            if not node_name or not cls.SERVICE_PATTERN.match(pod_name):
                continue

            if pod.status.phase != "Running" or not all([c.ready for c in pod.status.container_statuses]):
                return False

        return True
