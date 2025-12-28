from kubernetes import client, config
from collections import defaultdict
import time
import threading

from .context import Context
from .service import ServiceConfig


class KubeConfig:
    _api = None
    NAMESPACE = Context.get_parameter('NAMESPACE')

    # Caching controls
    _cache_lock = threading.Lock()
    _last_refresh_monotonic = 0.0
    _service_nodes_cache = None  # {service: set(nodes)}
    _node_services_cache = None  # {node: set(services)}
    _node_pods_cache = None  # {node: set(pod_names)}

    # Precomputed list views to avoid per-call conversions
    _service_nodes_list_cache = None  # {service: [nodes]}
    _node_services_list_cache = None  # {node: [services]}
    _node_pods_list_cache = None  # {node: [pod_names]}

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
    def force_refresh(cls):
        cls._refresh_now()

    @classmethod
    def _cache_expired(cls) -> bool:
        return (time.monotonic() - cls._last_refresh_monotonic) > cls.CACHE_TTL \
            or cls._service_nodes_cache is None or cls._node_services_cache is None or cls._node_pods_cache is None

    @classmethod
    def invalidate_cache(cls):
        """Manually invalidate the internal cache so that the next call refreshes from the API."""
        with cls._cache_lock:
            cls._service_nodes_cache = None
            cls._node_services_cache = None
            cls._node_pods_cache = None
            cls._service_nodes_list_cache = None
            cls._node_services_list_cache = None
            cls._node_pods_list_cache = None
            cls._last_refresh_monotonic = 0.0

    @classmethod
    def _build_maps_from_pods(cls, pods):
        """Build internal maps from a list of Pod objects."""
        service_nodes = defaultdict(set)
        node_services = defaultdict(set)
        node_pods = defaultdict(set)

        for pod in pods:
            pod_name = getattr(pod.metadata, 'name', None)
            node_name = getattr(pod.spec, 'node_name', None)
            if not pod_name or not node_name:
                continue

            service_name = ServiceConfig.map_pod_name_to_service(pod_name)
            if not service_name:
                continue

            service_nodes[service_name].add(node_name)
            node_services[node_name].add(service_name)
            node_pods[node_name].add(pod_name)

        # Store as plain dicts of sets; create list views for fast reads
        service_nodes_cache = {svc: set(nodes) for svc, nodes in service_nodes.items()}
        node_services_cache = {node: set(svcs) for node, svcs in node_services.items()}
        node_pods_cache = {node: set(pods_) for node, pods_ in node_pods.items()}

        # Precompute list views (sorted for determinism)
        service_nodes_list_cache = {svc: sorted(list(nodes)) for svc, nodes in service_nodes_cache.items()}
        node_services_list_cache = {node: sorted(list(svcs)) for node, svcs in node_services_cache.items()}
        node_pods_list_cache = {node: sorted(list(pods_)) for node, pods_ in node_pods_cache.items()}

        return (
            service_nodes_cache,
            node_services_cache,
            node_pods_cache,
            service_nodes_list_cache,
            node_services_list_cache,
            node_pods_list_cache,
        )

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

        (
            service_nodes_cache,
            node_services_cache,
            node_pods_cache,
            service_nodes_list_cache,
            node_services_list_cache,
            node_pods_list_cache,
        ) = cls._build_maps_from_pods(pods)

        with cls._cache_lock:
            cls._service_nodes_cache = service_nodes_cache
            cls._node_services_cache = node_services_cache
            cls._node_pods_cache = node_pods_cache
            cls._service_nodes_list_cache = service_nodes_list_cache
            cls._node_services_list_cache = node_services_list_cache
            cls._node_pods_list_cache = node_pods_list_cache
            cls._last_refresh_monotonic = time.monotonic()

    @classmethod
    def _is_cache_initialized(cls) -> bool:
        return (
            cls._service_nodes_cache is not None
            and cls._node_services_cache is not None
            and cls._node_pods_cache is not None
        )

    @classmethod
    def _is_cache_empty(cls) -> bool:
        a = cls._service_nodes_list_cache or {}
        b = cls._node_services_list_cache or {}
        c = cls._node_pods_list_cache or {}
        return (len(a) == 0) and (len(b) == 0) and (len(c) == 0)

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
            if force or cls._service_nodes_cache is None or cls._node_services_cache is None or cls._node_pods_cache is None:
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
    def get_node_pods_dict(cls):
        """
        Return the mapping of the entire node to pod name list.

        Returns:
            {
                node1: [pod1, pod2],
                node2: [pod3],
                ...,
            }
        """
        cls._refresh_cache_if_needed()
        cache = cls._node_pods_list_cache or {}
        # Return a shallow copy to avoid external modification of internal cache
        return {node: pods[:] for node, pods in cache.items()}

    @classmethod
    def get_pods_on_node(cls, node_name):
        """
        According to the node name, obtain the list of pod names on the node that match the cache policy.

        Args:
            node_name (str): Node name
        Returns:
            list[str]: List of pod names, sorted in dictionary order
        """
        cls._refresh_cache_if_needed()
        cache = cls._node_pods_list_cache or {}
        result = list(cache.get(node_name, []))
        if not result:
            # Perform a synchronization refresh upon the first miss, to avoid using expired cache.
            cls._refresh_now()
            cache = cls._node_pods_list_cache or {}
            return list(cache.get(node_name, []))
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
            if not node_name or not ServiceConfig.map_pod_name_to_service(pod_name):
                continue

            # Some clusters may not populate container_statuses immediately
            statuses = getattr(pod.status, 'container_statuses', None) or []
            if pod.status.phase != "Running" or not all([getattr(c, 'ready', False) for c in statuses]):
                return False

        return True

    @classmethod
    def get_pod_memory_from_metrics(cls, target_pod_names):
        """
        Get the metrics of all pods in the specified namespace from metrics.k8s.io,
        Then filter out the target pods and calculate the total memory per pod (sum of all containers, in bytes).
        """
        api = cls._get_api()
        resp = api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=cls.NAMESPACE,
            plural="pods",
        )

        target_set = set(target_pod_names)
        mem_by_pod = {}

        for item in resp.get("items", []):
            pod_name = item["metadata"]["name"]
            if pod_name not in target_set:
                continue

            total_bytes = 0
            for c in item.get("containers", []):
                mem_str = c["usage"]["memory"]  # e.g., '123456Ki'
                total_bytes += cls.parse_k8s_mem_to_bytes(mem_str)

            mem_by_pod[pod_name] = total_bytes

        return mem_by_pod

    @staticmethod
    def parse_k8s_mem_to_bytes(s: str) -> int:
        """
        Convert the memory strings of k8s (e.g.'123456Ki',' 200Mi','1Gi') into bytes (int).
        """
        s = s.strip()
        units = {
            "Ki": 1024,
            "Mi": 1024 ** 2,
            "Gi": 1024 ** 3,
            "Ti": 1024 ** 4,
            "Pi": 1024 ** 5,
            "Ei": 1024 ** 6,
            "K": 1000,
            "M": 1000 ** 2,
            "G": 1000 ** 3,
            "T": 1000 ** 4,
            "P": 1000 ** 5,
            "E": 1000 ** 6,
        }
        for u, mul in units.items():
            if s.endswith(u):
                value = float(s[:-len(u)])
                return int(value * mul)

        return int(float(s))
