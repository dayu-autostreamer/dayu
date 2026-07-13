import copy
from types import SimpleNamespace

import pytest


@pytest.fixture
def backend_core_instance(mounted_runtime):
    from backend_core import BackendCore

    return BackendCore()


def test_fill_datasource_config_uses_stable_in_cluster_service_dns(backend_core_instance):
    backend_core_instance.inner_datasource = True
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
        "http://datasource-edge.dayu.svc.cluster.local:8000/video0",
        "http://datasource-edge.dayu.svc.cluster.local:8000/video1",
    ]


def test_external_datasource_and_node_inventory_need_no_kubernetes_discovery(
        backend_core_instance,
):
    backend_core_instance.inner_datasource = False
    assert backend_core_instance.fill_datasource_url(
        "rtsp://camera-a/live", "video", "rtsp_video", 0
    ) == "rtsp://camera-a/live"

    inventory = {
        "cloudx1": {"role": "cloud", "ready": True},
        "edgex2": {"role": "edge", "ready": True},
        "edge3": {"role": "edge", "ready": True},
        "edgen4": {"role": "edge", "ready": True},
        "edgexn1": {"role": "edge", "ready": True},
        "misc-node": {"role": "edge", "ready": True},
        "not-ready": {"role": "edge", "ready": False},
    }
    backend_core_instance.runtime_orchestrator = SimpleNamespace(
        node_inventory=lambda: inventory
    )

    assert backend_core_instance.get_edge_nodes() == [
        {"name": "edge3"},
        {"name": "edgexn1"},
        {"name": "edgex2"},
        {"name": "edgen4"},
        {"name": "misc-node"},
    ]
    assert backend_core_instance.check_node_exist("edgex2") is True
    assert backend_core_instance.check_node_exist("not-ready") is False


def test_backend_core_validates_datasource_and_visualization_configs(
        backend_core_instance, tmp_path,
):
    backend_core_instance.inner_datasource = False
    datasource_path = tmp_path / "datasource.yaml"
    datasource_path.write_text(
        "source_name: demo\nsource_type: video\nsource_mode: http_video\n"
        "source_list:\n  - name: camera-a\n    url: http://camera/live\n"
        "    metadata: {fps: 25}\n",
        encoding="utf-8",
    )
    invalid_datasource = tmp_path / "invalid.yaml"
    invalid_datasource.write_text(
        "source_name: demo\nsource_type: video\nsource_mode: http_video\n"
        "source_list:\n  - name: camera-a\n    metadata: {fps: 25}\n",
        encoding="utf-8",
    )
    visualization = tmp_path / "visualization.yaml"
    visualization.write_text(
        "- name: CPU Usage\n  type: curve\n  variables: [cpu_usage]\n"
        "  size: 1\n  hook_name: cpu_usage\n  hook_params: \"{}\"\n",
        encoding="utf-8",
    )
    invalid_visualization = tmp_path / "invalid_visualization.yaml"
    invalid_visualization.write_text(
        "- name: Broken\n  type: curve\n  variables: [cpu_usage]\n"
        "  size: one\n  hook_params: \"[]\"\n",
        encoding="utf-8",
    )

    assert backend_core_instance.check_datasource_config(str(datasource_path))["source_name"] == "demo"
    assert backend_core_instance.check_datasource_config(str(invalid_datasource)) is None
    assert backend_core_instance.check_visualization_config(str(visualization))[0]["name"] == "CPU Usage"
    assert backend_core_instance.check_visualization_config(str(invalid_visualization)) is None


def test_visualizers_share_one_scheduler_snapshot_per_request(
        backend_core_instance, monkeypatch,
):
    import backend_core as backend_core_module

    calls = []
    backend_core_instance.system_visualization_configs = [
        {"hook_name": "cpu_usage", "variables": []},
        {"hook_name": "memory_usage", "variables": []},
        {"hook_name": "schedule_overhead", "variables": []},
    ]
    backend_core_instance.system_visualization_cache = SimpleNamespace(
        sync_and_get=lambda configs, namespace: [
            lambda resource=None: {"cpu": resource},
            lambda resource=None: {"memory": resource},
            lambda scheduling_overhead=None: {"overhead": scheduling_overhead},
        ]
    )
    backend_core_instance.resource_url = "http://scheduler.dayu.svc:9001/resource"

    def fake_request(url, method=None, **kwargs):
        calls.append(url)
        return 0.25 if url.endswith("/overhead") else {"edge-a": {"cpu_usage": 0.5}}

    monkeypatch.setattr(backend_core_module, "http_request", fake_request)

    result = backend_core_instance.prepare_system_visualizations_data()

    assert calls == [
        "http://scheduler.dayu.svc:9001/resource",
        "http://scheduler.dayu.svc:9001/overhead",
    ]
    assert result[0]["data"]["cpu"] == {"edge-a": {"cpu_usage": 0.5}}
    assert result[1]["data"]["memory"] == {"edge-a": {"cpu_usage": 0.5}}
    assert result[2]["data"]["overhead"] == 0.25


def test_backend_core_lookup_helpers_return_expected_items(backend_core_instance):
    backend_core_instance.parse_base_info = lambda: None
    backend_core_instance.schedulers = [{"id": "fixed", "name": "Fixed"}]
    backend_core_instance.dags = [{"dag_id": 1, "dag": {"_start": []}}]
    backend_core_instance.source_configs = [{
        "source_label": "source-config-0", "source_list": [{"id": 0}, {"id": 1}],
    }]
    backend_core_instance.result_visualization_configs = [{"name": "Result A"}]
    backend_core_instance.system_visualization_configs = [{"name": "CPU Usage"}]
    backend_core_instance.customized_source_result_visualization_configs = {
        1: [{"name": "Custom Result"}],
    }
    backend_core_instance.source_label = "source-config-0"

    assert backend_core_instance.find_scheduler_policy_by_id("fixed")["name"] == "Fixed"
    assert backend_core_instance.find_dag_by_id(1) == {"_start": []}
    assert backend_core_instance.get_source_ids() == [0, 1]
    assert backend_core_instance.get_result_visualization_config(1) == [
        {"id": 0, "name": "Custom Result"}
    ]
    assert backend_core_instance.get_system_visualization_config() == [
        {"id": 0, "name": "CPU Usage"}
    ]
