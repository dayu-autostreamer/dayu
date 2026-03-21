import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def build_service(name, service_type, node_port=None):
    ports = [] if node_port is None else [SimpleNamespace(node_port=node_port)]
    return SimpleNamespace(metadata=SimpleNamespace(name=name), spec=SimpleNamespace(type=service_type, ports=ports))


PORT_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "dependency"
    / "core"
    / "lib"
    / "network"
    / "port.py"
)


def load_port_module(monkeypatch, package_name, parameters):
    context_module = importlib.import_module("core.lib.common.context")
    monkeypatch.setattr(
        context_module.Context,
        "get_parameter",
        staticmethod(lambda key: parameters.get(key)),
    )

    spec = importlib.util.spec_from_file_location(package_name, PORT_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(package_name, None)
    return module


@pytest.fixture
def port_info_module(monkeypatch):
    port_module = importlib.import_module("core.lib.network.port")
    monkeypatch.setattr(port_module.PortInfo, "_api", None)
    monkeypatch.setattr(port_module.PortInfo, "_nodeport_services_cache", None)
    monkeypatch.setattr(port_module.PortInfo, "_last_refresh_monotonic", 0.0)
    monkeypatch.setattr(port_module.PortInfo, "_refresh_in_progress", False)
    monkeypatch.setattr(port_module.PortInfo, "_namespace", "dayu")
    monkeypatch.setattr(port_module.PortInfo, "_service_label_selector", None)
    monkeypatch.setattr(port_module.PortInfo, "_refresh_mode", "ttl")
    monkeypatch.setattr(port_module.PortInfo, "CACHE_TTL", 30.0)
    return port_module


@pytest.mark.unit
def test_port_info_refreshes_nodeport_cache_from_kubernetes_api(port_info_module, monkeypatch):
    fake_services = SimpleNamespace(
        items=[
            build_service("processor-face-detection-edgex1-0", "NodePort", 31000),
            build_service("frontend", "ClusterIP", 32000),
            build_service("processor-invalid-edgex1-0", "NodePort"),
        ]
    )
    fake_api = SimpleNamespace(list_namespaced_service=lambda namespace, label_selector=None: fake_services)

    monkeypatch.setattr(port_info_module.PortInfo, "_get_api", classmethod(lambda cls: fake_api))
    monkeypatch.setattr(port_info_module.time, "monotonic", lambda: 123.4)

    port_info_module.PortInfo._refresh_now()

    assert port_info_module.PortInfo._nodeport_services_cache == {"processor-face-detection-edgex1-0": 31000}
    assert port_info_module.PortInfo._last_refresh_monotonic == 123.4


@pytest.mark.unit
def test_port_info_refresh_skips_invalid_nodeports_and_service_iteration_errors(port_info_module, monkeypatch):
    broken_service = SimpleNamespace(
        metadata=SimpleNamespace(name="broken"),
        spec=SimpleNamespace(type="NodePort", ports=[object()]),
    )
    zero_port_service = build_service("processor-zero-edgex1-0", "NodePort", 0)
    valid_service = build_service("processor-valid-edgex1-0", "NodePort", 32000)
    fake_services = SimpleNamespace(items=[broken_service, zero_port_service, valid_service])
    fake_api = SimpleNamespace(list_namespaced_service=lambda namespace, label_selector=None: fake_services)

    monkeypatch.setattr(port_info_module.PortInfo, "_get_api", classmethod(lambda cls: fake_api))
    monkeypatch.setattr(port_info_module.time, "monotonic", lambda: 55.0)

    port_info_module.PortInfo._refresh_now()

    assert port_info_module.PortInfo._nodeport_services_cache == {"processor-valid-edgex1-0": 32000}
    assert port_info_module.PortInfo._last_refresh_monotonic == 55.0


@pytest.mark.unit
def test_port_info_retries_refresh_on_cache_miss_and_returns_component_port(port_info_module, monkeypatch):
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_mode", "never")
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", {"controller-main": 30000})

    def fake_refresh(cls):
        cls._nodeport_services_cache = {
            "controller-main": 30000,
            "processor-face-detection-edgex1-0": 31000,
        }

    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_now", classmethod(fake_refresh))

    assert port_info_module.PortInfo.get_all_ports("processor") == {"processor-face-detection-edgex1-0": 31000}
    assert port_info_module.PortInfo.get_component_port("processor") == 31000


@pytest.mark.unit
def test_port_info_extracts_service_ports_by_standardized_device_name(port_info_module, monkeypatch):
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", {
        "processor-face-detection-edgex1-0": 31000,
        "processor-car-tracking-cloudx1-0": 31001,
    })
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_cache_if_needed", classmethod(lambda cls, force=False: None))

    assert port_info_module.PortInfo.get_service_ports_dict("edge-x1") == {"face-detection": 31000}
    assert port_info_module.PortInfo.get_service_port("edge-x1", "face-detection") == 31000


@pytest.mark.unit
def test_port_info_starts_background_refresh_when_ttl_cache_expires(port_info_module, monkeypatch):
    refresh_calls = []
    started_threads = []

    class DummyThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started_threads.append((self.name, self.daemon))
            self.target()

    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_mode", "ttl")
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", {"processor-face-detection-edgex1-0": 31000})
    monkeypatch.setattr(port_info_module.PortInfo, "_cache_expired", classmethod(lambda cls: True))
    monkeypatch.setattr(port_info_module.PortInfo, "_is_cache_initialized", classmethod(lambda cls: True))
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_now", classmethod(lambda cls: refresh_calls.append(True)))
    monkeypatch.setattr(port_info_module.threading, "Thread", DummyThread)

    port_info_module.PortInfo._refresh_cache_if_needed()

    assert refresh_calls == [True]
    assert started_threads == [("PortInfoCacheRefresh", True)]
    assert port_info_module.PortInfo._refresh_in_progress is False


@pytest.mark.unit
def test_port_info_handles_refresh_failures_and_cache_state_helpers(port_info_module, monkeypatch):
    monkeypatch.setattr(port_info_module.PortInfo, "_get_api", classmethod(lambda cls: None))
    port_info_module.PortInfo._refresh_now()
    assert port_info_module.PortInfo._nodeport_services_cache is None

    failing_api = SimpleNamespace(
        list_namespaced_service=lambda namespace, label_selector=None: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(port_info_module.PortInfo, "_get_api", classmethod(lambda cls: failing_api))
    port_info_module.PortInfo._refresh_now()
    assert port_info_module.PortInfo._nodeport_services_cache is None

    monkeypatch.setattr(port_info_module.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", {})
    monkeypatch.setattr(port_info_module.PortInfo, "_last_refresh_monotonic", 5.0)
    assert port_info_module.PortInfo._is_cache_initialized() is True
    assert port_info_module.PortInfo._is_cache_empty() is True
    assert port_info_module.PortInfo._cache_expired() is False

    warmup_calls = []
    monkeypatch.setattr(
        port_info_module.PortInfo,
        "_refresh_now",
        classmethod(lambda cls: warmup_calls.append("refresh")),
    )
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_mode", "never")
    port_info_module.PortInfo._warmup_blocking_if_needed()
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_mode", "ttl")
    port_info_module.PortInfo._warmup_blocking_if_needed()
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", None)
    port_info_module.PortInfo._warmup_blocking_if_needed()

    assert warmup_calls == ["refresh"]


@pytest.mark.unit
def test_port_info_import_time_configuration_uses_repo_relative_loading(monkeypatch):
    ttl_module = load_port_module(
        monkeypatch,
        "dayu_test_port_info_ttl",
        {
            "NAMESPACE": "dayu",
            "KUBE_SERVICE_LABEL_SELECTOR": "app=dayu",
            "KUBE_CACHE_TTL": "7.5",
            "KUBE_CACHE_WARMUP_TIMEOUT": "1.2",
        },
    )
    assert ttl_module.PortInfo._namespace == "dayu"
    assert ttl_module.PortInfo._service_label_selector == "app=dayu"
    assert ttl_module.PortInfo._refresh_mode == "ttl"
    assert ttl_module.PortInfo.CACHE_TTL == 7.5
    assert ttl_module.PortInfo.WARMUP_TIMEOUT == 1.2

    never_module = load_port_module(
        monkeypatch,
        "dayu_test_port_info_never",
        {
            "NAMESPACE": "dayu",
            "KUBE_CACHE_TTL": "never",
            "KUBE_CACHE_WARMUP_TIMEOUT": None,
        },
    )
    assert never_module.PortInfo._refresh_mode == "never"
    assert math.isinf(never_module.PortInfo.CACHE_TTL)
    assert never_module.PortInfo.WARMUP_TIMEOUT == 3.0

    invalid_module = load_port_module(
        monkeypatch,
        "dayu_test_port_info_invalid",
        {
            "NAMESPACE": "dayu",
            "KUBE_CACHE_TTL": "not-a-number",
            "KUBE_CACHE_WARMUP_TIMEOUT": "2.0",
        },
    )
    assert invalid_module.PortInfo._refresh_mode == "never"
    assert math.isinf(invalid_module.PortInfo.CACHE_TTL)


@pytest.mark.unit
def test_port_info_api_loading_and_cache_management_cover_force_and_error_paths(port_info_module, monkeypatch):
    load_calls = []
    fake_api = object()

    monkeypatch.setattr(
        port_info_module.k8s.config,
        "load_incluster_config",
        lambda: (_ for _ in ()).throw(RuntimeError("no in-cluster config")),
    )
    monkeypatch.setattr(port_info_module.k8s.config, "load_kube_config", lambda: load_calls.append("kubeconfig"))
    monkeypatch.setattr(port_info_module.k8s.client, "CoreV1Api", lambda: fake_api)

    assert port_info_module.PortInfo._get_api() is fake_api
    assert port_info_module.PortInfo._get_api() is fake_api
    assert load_calls == ["kubeconfig"]

    second_module = importlib.import_module("core.lib.network.port")
    monkeypatch.setattr(second_module.PortInfo, "_api", None)
    monkeypatch.setattr(
        second_module.k8s.config,
        "load_incluster_config",
        lambda: (_ for _ in ()).throw(RuntimeError("no in-cluster config")),
    )
    monkeypatch.setattr(
        second_module.k8s.config,
        "load_kube_config",
        lambda: (_ for _ in ()).throw(RuntimeError("no kube config")),
    )
    monkeypatch.setattr(second_module.k8s.client, "CoreV1Api", lambda: fake_api)
    assert second_module.PortInfo._get_api() is fake_api

    refresh_calls = []
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_now", classmethod(lambda cls: refresh_calls.append("refresh")))
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_mode", "never")
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", {"processor-face-edgex1-0": 31000})

    port_info_module.PortInfo._refresh_cache_if_needed()
    port_info_module.PortInfo._refresh_cache_if_needed(force=True)
    assert refresh_calls == ["refresh"]

    port_info_module.PortInfo.invalidate_cache()
    assert port_info_module.PortInfo._nodeport_services_cache is None
    assert port_info_module.PortInfo._last_refresh_monotonic == 0.0


@pytest.mark.unit
def test_port_info_ttl_refresh_skips_when_cache_is_fresh_or_refresh_already_running(port_info_module, monkeypatch):
    refresh_calls = []
    started_threads = []

    class DummyThread:
        def __init__(self, target=None, name=None, daemon=None):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started_threads.append((self.name, self.daemon))
            self.target()

    monkeypatch.setattr(port_info_module.threading, "Thread", DummyThread)
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_mode", "ttl")
    monkeypatch.setattr(port_info_module.PortInfo, "_nodeport_services_cache", {"processor-a": 30001})
    monkeypatch.setattr(port_info_module.PortInfo, "_is_cache_initialized", classmethod(lambda cls: True))
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_now", classmethod(lambda cls: refresh_calls.append("refresh")))

    monkeypatch.setattr(port_info_module.PortInfo, "_cache_expired", classmethod(lambda cls: False))
    port_info_module.PortInfo._refresh_cache_if_needed()
    assert refresh_calls == []

    monkeypatch.setattr(port_info_module.PortInfo, "_cache_expired", classmethod(lambda cls: True))
    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_in_progress", True)
    port_info_module.PortInfo._refresh_cache_if_needed()
    assert refresh_calls == []
    assert started_threads == []

    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_in_progress", False)
    port_info_module.PortInfo._refresh_cache_if_needed(force=True)
    assert refresh_calls == ["refresh"]

    monkeypatch.setattr(port_info_module.PortInfo, "_refresh_now", classmethod(lambda cls: refresh_calls.append("force-refresh")))
    port_info_module.PortInfo.force_refresh()
    assert refresh_calls[-1] == "force-refresh"

    monkeypatch.setattr(port_info_module.PortInfo, "get_all_ports", staticmethod(lambda keyword: {}))
    with pytest.raises(Exception, match="does not exist"):
        port_info_module.PortInfo.get_component_port("processor")
