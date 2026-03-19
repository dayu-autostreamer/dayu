import copy
import importlib
from types import SimpleNamespace

import pytest


@pytest.fixture
def backend_core_instance(mounted_runtime, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pod_name",
        staticmethod(lambda *args, **kwargs: False),
    )
    return backend_core_module.BackendCore()


@pytest.mark.unit
def test_fill_datasource_config_assigns_source_ids_and_internal_urls(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    backend_core_instance.inner_datasource = True

    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "get_pod_node",
        staticmethod(lambda component, namespace: "datasource-node"),
    )
    monkeypatch.setattr(backend_core_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.10.0.8"))
    monkeypatch.setattr(backend_core_module.PortInfo, "get_component_port", staticmethod(lambda component: 31000))

    config = {
        "source_name": "demo",
        "source_type": "video",
        "source_mode": "http_video",
        "source_list": [
            {"name": "camera-a", "url": "http://placeholder", "metadata": {"fps": 25}},
            {"name": "camera-b", "url": "http://placeholder", "metadata": {"fps": 30}},
        ],
    }

    filled = backend_core_instance.fill_datasource_config(copy.deepcopy(config))

    assert filled["source_label"] == "source_config_0"
    assert [source["id"] for source in filled["source_list"]] == [0, 1]
    assert [source["url"] for source in filled["source_list"]] == [
        "http://10.10.0.8:31000/video0",
        "http://10.10.0.8:31000/video1",
    ]


@pytest.mark.unit
def test_fill_datasource_url_preserves_external_urls_and_sorts_edge_nodes(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")

    assert (
        backend_core_instance.fill_datasource_url(
            "rtsp://camera-a/live",
            source_type="video",
            source_mode="rtsp_video",
            source_id=0,
        )
        == "rtsp://camera-a/live"
    )

    monkeypatch.setattr(
        backend_core_module.NodeInfo,
        "get_node_info_role",
        staticmethod(
            lambda: {
                "cloudx1": "cloud",
                "edgex2": "edge",
                "edge3": "edge",
                "edgen4": "edge",
                "edgexn1": "edge",
                "misc-node": "edge",
            }
        ),
    )

    assert backend_core_instance.get_edge_nodes() == [
        {"name": "edge3"},
        {"name": "edgexn1"},
        {"name": "edgex2"},
        {"name": "edgen4"},
        {"name": "misc-node"},
    ]


@pytest.mark.unit
def test_backend_core_validates_datasource_and_visualization_configs(backend_core_instance, tmp_path):
    datasource_path = tmp_path / "datasource.yaml"
    datasource_path.write_text(
        "\n".join(
            [
                "source_name: demo",
                "source_type: video",
                "source_mode: http_video",
                "source_list:",
                "  - name: camera-a",
                "    url: http://camera-a/live",
                "    metadata: {fps: 25}",
            ]
        ),
        encoding="utf-8",
    )

    invalid_datasource_path = tmp_path / "invalid_datasource.yaml"
    invalid_datasource_path.write_text(
        "\n".join(
            [
                "source_name: demo",
                "source_type: video",
                "source_mode: http_video",
                "source_list:",
                "  - name: camera-a",
                "    metadata: {fps: 25}",
            ]
        ),
        encoding="utf-8",
    )

    visualization_path = tmp_path / "visualization.yaml"
    visualization_path.write_text(
        "\n".join(
            [
                "- name: CPU Usage",
                "  type: curve",
                "  variables: [cpu_usage]",
                "  size: 1",
                "  hook_name: cpu_usage",
                "  hook_params: \"{}\"",
            ]
        ),
        encoding="utf-8",
    )

    invalid_visualization_path = tmp_path / "invalid_visualization.yaml"
    invalid_visualization_path.write_text(
        "\n".join(
            [
                "- name: Broken View",
                "  type: curve",
                "  variables: [cpu_usage]",
                "  size: one",
                "  hook_name: cpu_usage",
                "  hook_params: \"[]\"",
            ]
        ),
        encoding="utf-8",
    )

    assert backend_core_instance.check_datasource_config(str(datasource_path))["source_name"] == "demo"
    assert backend_core_instance.check_datasource_config(str(invalid_datasource_path)) is None
    assert backend_core_instance.check_visualization_config(str(visualization_path))[0]["name"] == "CPU Usage"
    assert backend_core_instance.check_visualization_config(str(invalid_visualization_path)) is None


@pytest.mark.unit
def test_prepare_visualization_data_uses_custom_configs_and_handles_failures(backend_core_instance, monkeypatch):
    class FakeTask:
        def __init__(self, source_id=7):
            self.source_id = source_id

        def get_source_id(self):
            return self.source_id

    task = FakeTask()

    backend_core_instance.customized_source_result_visualization_configs = {
        7: [
            {"variables": ["frame"], "hook_name": "frame", "save_expense": True},
            {"variables": ["delay"], "hook_name": "delay"},
            {"variables": ["error"], "hook_name": "error"},
        ]
    }
    backend_core_instance.result_visualization_cache = SimpleNamespace(
        sync_and_get=lambda configs, namespace: [
            lambda task: {"frame": "kept"},
            lambda task: {"delay": 1.2},
            lambda task: (_ for _ in ()).throw(RuntimeError("broken visualizer")),
        ]
    )

    result = backend_core_instance.prepare_result_visualization_data(task, is_last=False)

    assert result == [
        {"id": 0, "data": {"frame": None}},
        {"id": 1, "data": {"delay": 1.2}},
    ]

    backend_core_instance.system_visualization_configs = [
        {"hook_name": "cpu_usage", "variables": []},
        {"hook_name": "memory_usage", "variables": []},
        {"hook_name": "broken", "variables": []},
    ]
    backend_core_instance.system_visualization_cache = SimpleNamespace(
        sync_and_get=lambda configs, namespace: [
            lambda resource=None: {"resource": resource},
            lambda: {"fallback": True},
            lambda resource=None: (_ for _ in ()).throw(RuntimeError("broken system visualizer")),
        ]
    )
    backend_core_instance.get_resource_url = lambda: setattr(backend_core_instance, "resource_url", "http://scheduler")
    monkeypatch.setattr(
        importlib.import_module("backend_core"),
        "http_request",
        lambda address, method=None, **kwargs: {"edgex1": {"available_bandwidth": 12.5}},
    )

    system_result = backend_core_instance.prepare_system_visualizations_data()

    assert system_result == [
        {"id": 0, "data": {"resource": {"edgex1": {"available_bandwidth": 12.5}}}},
        {"id": 1, "data": {"fallback": True}},
    ]


@pytest.mark.unit
def test_backend_core_lookup_helpers_return_expected_items(backend_core_instance):
    backend_core_instance.parse_base_info = lambda: None
    backend_core_instance.schedulers = [{"id": "fixed", "name": "Fixed"}]
    backend_core_instance.dags = [{"dag_id": 1, "dag": {"_start": []}}]
    backend_core_instance.source_configs = [
        {"source_label": "source-config-0", "source_list": [{"id": 0}, {"id": 1}]}
    ]
    backend_core_instance.result_visualization_configs = [{"name": "Result A"}]
    backend_core_instance.system_visualization_configs = [{"name": "CPU Usage"}]
    backend_core_instance.customized_source_result_visualization_configs = {1: [{"name": "Custom Result"}]}
    backend_core_instance.source_label = "source-config-0"

    assert backend_core_instance.find_scheduler_policy_by_id("fixed") == {"id": "fixed", "name": "Fixed"}
    assert backend_core_instance.find_dag_by_id(1) == {"_start": []}
    assert backend_core_instance.find_datasource_configuration_by_label("source-config-0")["source_list"][0]["id"] == 0
    assert backend_core_instance.get_source_ids() == [0, 1]
    assert backend_core_instance.get_result_visualization_config(1) == [{"id": 0, "name": "Custom Result"}]
    assert backend_core_instance.get_system_visualization_config() == [{"id": 0, "name": "CPU Usage"}]
