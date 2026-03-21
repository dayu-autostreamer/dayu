import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


kube_module = importlib.import_module("core.lib.common.kube")
port_module = importlib.import_module("core.lib.network.port")


KUBE_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "dependency"
    / "core"
    / "lib"
    / "common"
    / "kube.py"
)


def make_pod(name, node_name, phase="Running", ready=True):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(node_name=node_name),
        status=SimpleNamespace(
            phase=phase,
            container_statuses=[SimpleNamespace(ready=ready)],
        ),
    )


def load_kube_module(monkeypatch, package_name, parameters):
    context_module = importlib.import_module("core.lib.common.context")
    monkeypatch.setattr(
        context_module.Context,
        "get_parameter",
        staticmethod(lambda key: parameters.get(key)),
    )

    qualified_name = f"core.lib.common.{package_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, KUBE_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(qualified_name, None)
    return module


@pytest.fixture
def kube_config(monkeypatch):
    cls = kube_module.KubeConfig
    monkeypatch.setattr(cls, "_core_api", None)
    monkeypatch.setattr(cls, "_custom_api", None)
    monkeypatch.setattr(cls, "_service_nodes_cache", None)
    monkeypatch.setattr(cls, "_node_services_cache", None)
    monkeypatch.setattr(cls, "_node_pods_cache", None)
    monkeypatch.setattr(cls, "_service_nodes_list_cache", None)
    monkeypatch.setattr(cls, "_node_services_list_cache", None)
    monkeypatch.setattr(cls, "_node_pods_list_cache", None)
    monkeypatch.setattr(cls, "_last_refresh_monotonic", 0.0)
    monkeypatch.setattr(cls, "_refresh_in_progress", False)
    monkeypatch.setattr(cls, "NAMESPACE", "dayu")
    monkeypatch.setattr(cls, "_label_selector", None)
    monkeypatch.setattr(cls, "_field_selector", "spec.nodeName!=null")
    monkeypatch.setattr(cls, "_refresh_mode", "ttl")
    monkeypatch.setattr(cls, "CACHE_TTL", 30.0)
    return cls


@pytest.fixture
def port_info(monkeypatch):
    cls = port_module.PortInfo
    monkeypatch.setattr(cls, "_api", None)
    monkeypatch.setattr(cls, "_nodeport_services_cache", None)
    monkeypatch.setattr(cls, "_last_refresh_monotonic", 0.0)
    monkeypatch.setattr(cls, "_refresh_in_progress", False)
    monkeypatch.setattr(cls, "_namespace", "dayu")
    monkeypatch.setattr(cls, "_service_label_selector", None)
    monkeypatch.setattr(cls, "_refresh_mode", "ttl")
    monkeypatch.setattr(cls, "CACHE_TTL", 30.0)
    return cls


@pytest.mark.unit
def test_kube_config_refreshes_pod_topology_and_returns_copies(kube_config, monkeypatch):
    fake_api = SimpleNamespace(
        list_namespaced_pod=lambda namespace, label_selector=None, field_selector=None: SimpleNamespace(
            items=[
                make_pod("processor-face-detection-edgex1-0", "edgex1"),
                make_pod("processor-face-detection-edgex2-0", "edgex2"),
                make_pod("processor-car-tracking-cloudx1-0", "cloudx1"),
                make_pod("frontend-0", "cloudx1"),
                make_pod("processor-invalid-0", None),
            ]
        )
    )

    monkeypatch.setattr(kube_config, "_get_core_api", classmethod(lambda cls: fake_api))
    monkeypatch.setattr(kube_module.time, "monotonic", lambda: 321.0)

    kube_config._refresh_now()

    assert kube_config._last_refresh_monotonic == 321.0
    assert kube_config.get_service_nodes_dict() == {
        "face-detection": ["edgex1", "edgex2"],
        "car-tracking": ["cloudx1"],
    }
    assert kube_config.get_node_services_dict() == {
        "cloudx1": ["car-tracking"],
        "edgex1": ["face-detection"],
        "edgex2": ["face-detection"],
    }
    assert kube_config.get_node_pods_dict() == {
        "cloudx1": ["processor-car-tracking-cloudx1-0"],
        "edgex1": ["processor-face-detection-edgex1-0"],
        "edgex2": ["processor-face-detection-edgex2-0"],
    }
    assert kube_config.get_services_on_node("edgex1") == ["face-detection"]
    assert kube_config.get_nodes_for_service("car-tracking") == ["cloudx1"]
    assert kube_config.get_pods_on_node("edgex2") == ["processor-face-detection-edgex2-0"]

    mutated = kube_config.get_service_nodes_dict()
    mutated["face-detection"].append("mutated")
    assert kube_config.get_service_nodes_dict()["face-detection"] == ["edgex1", "edgex2"]


@pytest.mark.unit
def test_kube_config_handles_refresh_modes_invalidation_and_background_refresh(kube_config, monkeypatch):
    warmups = []
    monkeypatch.setattr(
        kube_config,
        "_warmup_blocking_if_needed",
        classmethod(lambda cls: warmups.append("warmup")),
    )
    monkeypatch.setattr(kube_config, "_is_cache_initialized", classmethod(lambda cls: False))

    kube_config._refresh_cache_if_needed()
    assert warmups == ["warmup"]

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

    monkeypatch.setattr(kube_config, "_is_cache_initialized", classmethod(lambda cls: True))
    monkeypatch.setattr(kube_config, "_cache_expired", classmethod(lambda cls: True))
    monkeypatch.setattr(kube_config, "_refresh_now", classmethod(lambda cls: refresh_calls.append("refresh")))
    monkeypatch.setattr(kube_module.threading, "Thread", DummyThread)

    kube_config._refresh_cache_if_needed()
    kube_config.force_refresh()

    assert refresh_calls == ["refresh", "refresh"]
    assert started_threads == [("KubeConfigCacheRefresh", True)]
    assert kube_config._refresh_in_progress is False

    kube_config._service_nodes_cache = {"svc": {"node"}}
    kube_config._node_services_cache = {"node": {"svc"}}
    kube_config._node_pods_cache = {"node": {"pod"}}
    kube_config._service_nodes_list_cache = {"svc": ["node"]}
    kube_config._node_services_list_cache = {"node": ["svc"]}
    kube_config._node_pods_list_cache = {"node": ["pod"]}
    kube_config.invalidate_cache()

    assert kube_config._service_nodes_cache is None
    assert kube_config._node_services_list_cache is None
    assert kube_config._last_refresh_monotonic == 0.0


@pytest.mark.unit
def test_kube_config_queries_running_state_metrics_and_unit_parsing(kube_config, monkeypatch):
    pods = SimpleNamespace(
        items=[
            make_pod("processor-face-detection-edgex1-0", "edgex1", phase="Running", ready=True),
            make_pod("processor-car-tracking-cloudx1-0", "cloudx1", phase="Pending", ready=False),
        ]
    )
    core_api = SimpleNamespace(
        list_namespaced_pod=lambda namespace, label_selector=None, field_selector=None: pods
    )
    custom_api = SimpleNamespace(
        list_namespaced_custom_object=lambda **kwargs: {
            "items": [
                {
                    "metadata": {"name": "processor-face-detection-edgex1-0"},
                    "containers": [
                        {"usage": {"memory": "1Mi"}},
                        {"usage": {"memory": "0.5Mi"}},
                    ],
                }
            ]
        }
    )

    load_calls = []
    monkeypatch.setattr(kube_module.config, "load_incluster_config", lambda: load_calls.append("incluster"))
    monkeypatch.setattr(kube_module.client, "CoreV1Api", lambda: core_api)
    monkeypatch.setattr(kube_module.client, "CustomObjectsApi", lambda: custom_api)

    assert kube_config._get_core_api() is core_api
    assert kube_config._get_custom_api() is custom_api
    assert load_calls == ["incluster", "incluster"]
    assert kube_config.check_services_running() is False

    pods.items[1] = make_pod("processor-car-tracking-cloudx1-0", "cloudx1", phase="Running", ready=True)
    assert kube_config.check_services_running() is True
    assert kube_config.get_pod_memory_from_metrics(["processor-face-detection-edgex1-0"]) == {
        "processor-face-detection-edgex1-0": 1572864
    }
    assert kube_config.parse_k8s_mem_to_bytes("1.5Gi") == int(1.5 * 1024 ** 3)
    assert kube_config.parse_k8s_mem_to_bytes("200M") == 200000000
    assert kube_config.parse_k8s_mem_to_bytes("512") == 512


@pytest.mark.unit
def test_kube_config_import_time_configuration_uses_repo_relative_loading(monkeypatch):
    ttl_module = load_kube_module(
        monkeypatch,
        "dayu_test_kube_config_ttl",
        {
            "NAMESPACE": "dayu",
            "KUBE_POD_LABEL_SELECTOR": "app=dayu",
            "KUBE_POD_FIELD_SELECTOR": "spec.nodeName=edge-x1",
            "KUBE_CACHE_TTL": "12.5",
        },
    )
    assert ttl_module.KubeConfig.NAMESPACE == "dayu"
    assert ttl_module.KubeConfig._label_selector == "app=dayu"
    assert ttl_module.KubeConfig._field_selector == "spec.nodeName=edge-x1"
    assert ttl_module.KubeConfig._refresh_mode == "ttl"
    assert ttl_module.KubeConfig.CACHE_TTL == 12.5

    never_module = load_kube_module(
        monkeypatch,
        "dayu_test_kube_config_never",
        {
            "NAMESPACE": "dayu",
            "KUBE_CACHE_TTL": "never",
        },
    )
    assert never_module.KubeConfig._refresh_mode == "never"
    assert math.isinf(never_module.KubeConfig.CACHE_TTL)

    invalid_module = load_kube_module(
        monkeypatch,
        "dayu_test_kube_config_invalid",
        {
            "NAMESPACE": "dayu",
            "KUBE_CACHE_TTL": "not-a-number",
        },
    )
    assert invalid_module.KubeConfig._refresh_mode == "never"
    assert math.isinf(invalid_module.KubeConfig.CACHE_TTL)


@pytest.mark.unit
def test_kube_config_skips_non_service_pods_and_non_target_metric_items(kube_config, monkeypatch):
    pods = SimpleNamespace(
        items=[
            make_pod("frontend-0", "cloudx1", phase="Pending", ready=False),
            make_pod("processor-face-detection-edgex1-0", "edgex1", phase="Running", ready=True),
            make_pod("processor-no-node-edgex2-0", None, phase="Pending", ready=False),
        ]
    )
    core_api = SimpleNamespace(
        list_namespaced_pod=lambda namespace, label_selector=None, field_selector=None: pods
    )
    custom_api = SimpleNamespace(
        list_namespaced_custom_object=lambda **kwargs: {
            "items": [
                {
                    "metadata": {"name": "processor-face-detection-edgex1-0"},
                    "containers": [{"usage": {"memory": "2Mi"}}],
                },
                {
                    "metadata": {"name": "processor-other-edgex2-0"},
                    "containers": [{"usage": {"memory": "9Mi"}}],
                },
            ]
        }
    )

    monkeypatch.setattr(kube_config, "_get_core_api", classmethod(lambda cls: core_api))
    monkeypatch.setattr(kube_config, "_get_custom_api", classmethod(lambda cls: custom_api))

    assert kube_config.check_services_running() is True
    assert kube_config.get_pod_memory_from_metrics(["processor-face-detection-edgex1-0"]) == {
        "processor-face-detection-edgex1-0": 2 * 1024 ** 2
    }


@pytest.mark.unit
def test_port_info_supports_api_fallback_warmup_and_missing_component_errors(port_info, monkeypatch):
    load_calls = []
    api_instance = SimpleNamespace()

    def fail_incluster():
        load_calls.append("incluster")
        raise RuntimeError("cluster config unavailable")

    monkeypatch.setattr(port_module.k8s.config, "load_incluster_config", fail_incluster)
    monkeypatch.setattr(port_module.k8s.config, "load_kube_config", lambda: load_calls.append("kubeconfig"))
    monkeypatch.setattr(port_module.k8s.client, "CoreV1Api", lambda: api_instance)

    assert port_info._get_api() is api_instance
    assert load_calls == ["incluster", "kubeconfig"]

    warmups = []
    monkeypatch.setattr(
        port_info,
        "_warmup_blocking_if_needed",
        classmethod(lambda cls: warmups.append("warmup")),
    )
    monkeypatch.setattr(port_info, "_is_cache_initialized", classmethod(lambda cls: False))

    port_info._refresh_cache_if_needed()
    assert warmups == ["warmup"]

    port_info._nodeport_services_cache = {"controller-main": 30000}
    assert port_info.get_component_port("controller") == 30000

    port_info.invalidate_cache()
    assert port_info._nodeport_services_cache is None
    assert port_info._last_refresh_monotonic == 0.0

    monkeypatch.setattr(port_info, "_refresh_cache_if_needed", classmethod(lambda cls, force=False: None))
    monkeypatch.setattr(port_info, "_refresh_now", classmethod(lambda cls: None))
    with pytest.raises(Exception, match="does not exist"):
        port_info.get_component_port("processor")


@pytest.mark.unit
def test_kube_config_handles_cache_miss_refreshes_and_never_mode(kube_config, monkeypatch):
    refresh_calls = []
    original_refresh_cache_if_needed = kube_module.KubeConfig.__dict__["_refresh_cache_if_needed"]

    def fake_refresh(cls):
        refresh_calls.append("refresh")
        cls._service_nodes_cache = {"face-detection": {"edge-a"}}
        cls._node_services_cache = {"edge-a": {"face-detection"}}
        cls._node_pods_cache = {"edge-a": {"processor-face-detection-edge-a-0"}}
        cls._service_nodes_list_cache = {"face-detection": ["edge-a"]}
        cls._node_services_list_cache = {"edge-a": ["face-detection"]}
        cls._node_pods_list_cache = {"edge-a": ["processor-face-detection-edge-a-0"]}

    monkeypatch.setattr(kube_config, "_refresh_cache_if_needed", classmethod(lambda cls, force=False: None))
    monkeypatch.setattr(kube_config, "_refresh_now", classmethod(fake_refresh))
    monkeypatch.setattr(kube_config, "_service_nodes_cache", {})
    monkeypatch.setattr(kube_config, "_node_services_cache", {})
    monkeypatch.setattr(kube_config, "_node_pods_cache", {})
    monkeypatch.setattr(kube_config, "_service_nodes_list_cache", {})
    monkeypatch.setattr(kube_config, "_node_services_list_cache", {})
    monkeypatch.setattr(kube_config, "_node_pods_list_cache", {})

    assert kube_config._is_cache_empty() is True
    assert kube_config.get_services_on_node("edge-a") == ["face-detection"]
    assert kube_config.get_nodes_for_service("face-detection") == ["edge-a"]
    assert kube_config.get_pods_on_node("edge-a") == ["processor-face-detection-edge-a-0"]
    assert refresh_calls == ["refresh"]

    monkeypatch.setattr(kube_config, "_refresh_cache_if_needed", original_refresh_cache_if_needed)

    never_mode_calls = []
    monkeypatch.setattr(kube_config, "_refresh_mode", "never")
    monkeypatch.setattr(kube_config, "_service_nodes_cache", None)
    monkeypatch.setattr(kube_config, "_node_services_cache", None)
    monkeypatch.setattr(kube_config, "_node_pods_cache", None)
    monkeypatch.setattr(
        kube_config,
        "_refresh_now",
        classmethod(lambda cls: never_mode_calls.append("refresh")),
    )
    kube_config._refresh_cache_if_needed()
    kube_config._refresh_cache_if_needed(force=True)
    kube_config._warmup_blocking_if_needed()

    assert never_mode_calls == ["refresh", "refresh"]


@pytest.mark.unit
def test_kube_config_force_refresh_and_miss_requeries_cover_remaining_cache_paths(kube_config, monkeypatch):
    refresh_calls = []

    monkeypatch.setattr(kube_config, "_refresh_mode", "ttl")
    monkeypatch.setattr(kube_config, "_is_cache_initialized", classmethod(lambda cls: True))
    monkeypatch.setattr(kube_config, "_refresh_now", classmethod(lambda cls: refresh_calls.append("refresh")))

    kube_config._warmup_blocking_if_needed()
    assert refresh_calls == []

    monkeypatch.setattr(kube_config, "_is_cache_initialized", classmethod(lambda cls: False))
    kube_config._warmup_blocking_if_needed()
    assert refresh_calls == ["refresh"]

    monkeypatch.setattr(kube_config, "_is_cache_initialized", classmethod(lambda cls: True))
    monkeypatch.setattr(kube_config, "_cache_expired", classmethod(lambda cls: True))
    monkeypatch.setattr(kube_config, "_refresh_in_progress", True)
    kube_config._refresh_cache_if_needed()
    assert refresh_calls == ["refresh"]

    kube_config._refresh_in_progress = False
    kube_config._refresh_cache_if_needed(force=True)
    assert refresh_calls == ["refresh", "refresh"]

    def fake_refresh(cls):
        cls._service_nodes_list_cache = {"face-detection": ["edge-a"]}
        cls._node_pods_list_cache = {"edge-a": ["processor-face-detection-edge-a-0"]}

    monkeypatch.setattr(kube_config, "_refresh_cache_if_needed", classmethod(lambda cls, force=False: None))
    monkeypatch.setattr(kube_config, "_refresh_now", classmethod(fake_refresh))
    monkeypatch.setattr(kube_config, "_service_nodes_list_cache", {})
    monkeypatch.setattr(kube_config, "_node_pods_list_cache", {})

    assert kube_config.get_nodes_for_service("face-detection") == ["edge-a"]
    assert kube_config.get_pods_on_node("edge-a") == ["processor-face-detection-edge-a-0"]
