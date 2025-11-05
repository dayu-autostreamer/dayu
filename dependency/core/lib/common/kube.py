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
    _service_nodes_cache = None  # {service: set(nodes)}
    _node_services_cache = None  # {node: set(services)}

    # Precomputed list views to avoid per-call conversions
    _service_nodes_list_cache = None  # {service: [nodes]}
    _node_services_list_cache = None  # {node: [services]}

    # Non-blocking refresh state
    _refresh_in_progress = False

    # Optional selectors to reduce API payload
    _label_selector = Context.get_parameter('KUBE_POD_LABEL_SELECTOR') or None
    _field_selector = Context.get_parameter('KUBE_POD_FIELD_SELECTOR') or "spec.nodeName!=null"

    # Cache TTL and refresh mode parsing (only 'ttl' and 'never')
    try:
        _ttl_raw = Context.get_parameter('KUBE_CACHE_TTL')
        if _ttl_raw is None:
            CACHE_TTL = float('inf')
            _refresh_mode = 'never'
        else:
            s = str(_ttl_raw).strip().lower()
            if s == 'never':
                _refresh_mode = 'never'
                CACHE_TTL = float('inf')
            else:
                _refresh_mode = 'ttl'
                CACHE_TTL = float(s)

    except Exception:
        CACHE_TTL = float('inf')
        _refresh_mode = 'never'

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
            cls._service_nodes_list_cache = None
            cls._node_services_list_cache = None
            cls._last_refresh_monotonic = 0.0

    @classmethod
    def _build_maps_from_pods(cls, pods):
        """Build internal maps from a list of Pod objects."""
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

        # Store as plain dicts of sets; create list views for fast reads
        service_nodes_cache = {svc: set(nodes) for svc, nodes in service_nodes.items()}
        node_services_cache = {node: set(svcs) for node, svcs in node_services.items()}

        # Precompute list views (sorted for determinism)
        service_nodes_list_cache = {svc: sorted(list(nodes)) for svc, nodes in service_nodes_cache.items()}
        node_services_list_cache = {node: sorted(list(svcs)) for node, svcs in node_services_cache.items()}

        return service_nodes_cache, node_services_cache, service_nodes_list_cache, node_services_list_cache

    @classmethod
    def _refresh_now(cls):
        """Perform a synchronous refresh from the Kubernetes API for pod topology only.
        Any API error will keep existing caches intact to be resilient.
        """
        api = cls._get_api()
        pods = api.list_namespaced_pod(
            cls.NAMESPACE,
            label_selector=cls._label_selector,
            field_selector=cls._field_selector,
        ).items

        service_nodes_cache, node_services_cache, service_nodes_list_cache, node_services_list_cache = cls._build_maps_from_pods(pods)

        with cls._cache_lock:
            cls._service_nodes_cache = service_nodes_cache
            cls._node_services_cache = node_services_cache
            cls._service_nodes_list_cache = service_nodes_list_cache
            cls._node_services_list_cache = node_services_list_cache
            cls._last_refresh_monotonic = time.monotonic()

    @classmethod
    def _is_cache_initialized(cls) -> bool:
        return cls._service_nodes_cache is not None and cls._node_services_cache is not None

    @classmethod
    def _is_cache_empty(cls) -> bool:
        a = cls._service_nodes_list_cache or {}
        b = cls._node_services_list_cache or {}
        return (len(a) == 0) and (len(b) == 0)

    @classmethod
    def _warmup_blocking_if_needed(cls):
        """
            Block on first TTL access until we have attempted a synchronous refresh.
        """
        if getattr(cls, '_refresh_mode', 'ttl') != 'ttl':
            return
        if cls._is_cache_initialized():
            return
        # Synchronous refresh
        cls._refresh_now()

    @classmethod
    def _refresh_cache_if_needed(cls, force: bool = False):
        """Refresh caches according to configured mode.
        - ttl: refresh on TTL expiry (non-blocking background for subsequent calls)
        - never: no auto refresh except first warm-up or explicit force
        """
        mode = getattr(cls, '_refresh_mode', 'ttl')

        if mode == 'never':
            # Only warm-up if empty or force refresh
            if force or cls._service_nodes_cache is None or cls._node_services_cache is None:
                cls._refresh_now()
            return

        # TTL mode: ensure cold-start is warmed up synchronously once
        if not force and not cls._is_cache_initialized():
            cls._warmup_blocking_if_needed()
            return

        # Default TTL mode
        if not force and not cls._cache_expired():
            return
        if force:
            cls._refresh_now()
            return

        with cls._cache_lock:
            if cls._refresh_in_progress or (not cls._cache_expired()):
                return
            cls._refresh_in_progress = True

        def _bg_refresh():
            try:
                cls._refresh_now()
            finally:
                with cls._cache_lock:
                    cls._refresh_in_progress = False

        threading.Thread(target=_bg_refresh, name="KubeConfigCacheRefresh", daemon=True).start()

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
        cache = cls._service_nodes_list_cache or {}
        return {svc: nodes[:] for svc, nodes in cache.items()}

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

        cache = cls._node_services_list_cache or {}
        return {node: svcs[:] for node, svcs in cache.items()}

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
        cache = cls._node_services_list_cache or {}
        result = list(cache.get(node_name, []))
        # On miss, perform one synchronous refresh
        if not result:
            cls._refresh_now()
            cache = cls._node_services_list_cache or {}
            return list(cache.get(node_name, []))
        return result

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
        cache = cls._service_nodes_list_cache or {}
        result = list(cache.get(service_name, []))
        if not result:
            cls._refresh_now()
            cache = cls._service_nodes_list_cache or {}
            return list(cache.get(service_name, []))
        return result

    @classmethod
    def check_services_running(cls):
        """
        Check if all services are running
        Returns:
            state: bool
        """
        api = cls._get_api()

        pods = api.list_namespaced_pod(
            cls.NAMESPACE,
            label_selector=cls._label_selector,
            field_selector=cls._field_selector,
        ).items
        for pod in pods:
            pod_name = pod.metadata.name
            node_name = pod.spec.node_name
            if not node_name or not cls.SERVICE_PATTERN.match(pod_name):
                continue

            # Some clusters may not populate container_statuses immediately
            statuses = getattr(pod.status, 'container_statuses', None) or []
            if pod.status.phase != "Running" or not all([getattr(c, 'ready', False) for c in statuses]):
                return False

        return True
