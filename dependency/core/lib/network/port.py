import kubernetes as k8s
import threading
import time
from core.lib.common import Context, SystemConstant, NameMaintainer


class PortInfo:
    # Shared API client for services
    _api = None

    # Cache: {service_name: node_port}
    _cache_lock = threading.Lock()
    _last_refresh_monotonic = 0.0
    _nodeport_services_cache = None
    _refresh_in_progress = False

    # Config
    _namespace = Context.get_parameter('NAMESPACE')
    _service_label_selector = Context.get_parameter('KUBE_SERVICE_LABEL_SELECTOR') or None
    try:
        _ttl_raw = Context.get_parameter('KUBE_CACHE_TTL')
        if _ttl_raw is None:
            _refresh_mode = 'never'
            CACHE_TTL = float('inf')
        else:
            s = str(_ttl_raw).strip().lower()
            if s == 'never':
                _refresh_mode = 'never'
                CACHE_TTL = float('inf')
            else:
                _refresh_mode = 'ttl'
                CACHE_TTL = float(s)
    except Exception:
        _refresh_mode = 'never'
        CACHE_TTL = float('inf')

    # Warm-up timeout (seconds) for first cold-start in TTL mode
    _warmup_raw = Context.get_parameter('KUBE_CACHE_WARMUP_TIMEOUT')
    WARMUP_TIMEOUT = float(str(_warmup_raw).strip()) if _warmup_raw is not None else 3.0

    @classmethod
    def force_refresh(cls):
        cls._refresh_now()

    @classmethod
    def _get_api(cls):
        if not cls._api:
            # Try in-cluster, then fallback to local kubeconfig for dev
            try:
                k8s.config.load_incluster_config()
            except Exception:
                try:
                    k8s.config.load_kube_config()
                except Exception:
                    pass
            cls._api = k8s.client.CoreV1Api()
        return cls._api

    @classmethod
    def _cache_expired(cls) -> bool:
        return (time.monotonic() - cls._last_refresh_monotonic) > cls.CACHE_TTL or cls._nodeport_services_cache is None

    @classmethod
    def invalidate_cache(cls):
        with cls._cache_lock:
            cls._nodeport_services_cache = None
            cls._last_refresh_monotonic = 0.0

    @classmethod
    def _refresh_now(cls):
        api = cls._get_api()
        if not api:
            return
        try:
            services = api.list_namespaced_service(
                cls._namespace,
                label_selector=cls._service_label_selector,
            ).items
        except Exception:
            return

        result = {}
        for svc in services:
            try:
                if getattr(svc.spec, 'type', None) != 'NodePort':
                    continue
                ports = getattr(svc.spec, 'ports', None) or []
                if not ports:
                    continue
                node_port = getattr(ports[0], 'node_port', None)
                if node_port:
                    result[svc.metadata.name] = int(node_port)
            except Exception:
                continue

        with cls._cache_lock:
            cls._nodeport_services_cache = result
            cls._last_refresh_monotonic = time.monotonic()

    @classmethod
    def _is_cache_initialized(cls) -> bool:
        return cls._nodeport_services_cache is not None

    @classmethod
    def _is_cache_empty(cls) -> bool:
        cache = cls._nodeport_services_cache or {}
        return len(cache) == 0

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
        mode = getattr(cls, '_refresh_mode', 'ttl')
        if mode == 'never':
            if force or cls._nodeport_services_cache is None:
                cls._refresh_now()
            return

        # TTL mode: ensure cold-start is warmed up synchronously once
        if not force and not cls._is_cache_initialized():
            cls._warmup_blocking_if_needed()
            return

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

        threading.Thread(target=_bg_refresh, name="PortInfoCacheRefresh", daemon=True).start()

    @staticmethod
    def get_component_port(component_name: str) -> int:
        ports_dict = PortInfo.get_all_ports(component_name)
        ports_list = list(ports_dict.values())
        if ports_list:
            return ports_list[0]
        raise Exception(f"Component '{component_name}' does not exist.")

    @staticmethod
    def get_all_ports(keyword: str) -> dict:
        """Return mapping of service_name -> node_port filtered by keyword."""
        PortInfo._refresh_cache_if_needed()
        cache = PortInfo._nodeport_services_cache or {}
        result = {name: port for name, port in cache.items() if keyword in name}
        # On miss, perform one synchronous refresh
        if not result:
            PortInfo._refresh_now()
            cache = PortInfo._nodeport_services_cache or {}
            return {name: port for name, port in cache.items() if keyword in name}
        return result

    @staticmethod
    def get_service_ports_dict(device:str) -> dict:
        component_name = SystemConstant.PROCESSOR.value
        ports_dict = PortInfo.get_all_ports(component_name)
        service_ports_dict = {}
        for svc_name in ports_dict:
            # get sub service name
            if NameMaintainer.standardize_device_name(device) == svc_name.split('-')[-2]:
                des_name = '-'.join(svc_name.split('-')[1:-2])
                service_ports_dict[des_name] = ports_dict[svc_name]

        return service_ports_dict

    @staticmethod
    def get_service_port(device:str, service_name: str) -> int:
        return PortInfo.get_service_ports_dict(device).get(service_name)
