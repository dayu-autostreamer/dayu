import importlib
from types import SimpleNamespace

import pytest


def build_service(name, service_type, node_port=None):
    ports = [] if node_port is None else [SimpleNamespace(node_port=node_port)]
    return SimpleNamespace(metadata=SimpleNamespace(name=name), spec=SimpleNamespace(type=service_type, ports=ports))


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
