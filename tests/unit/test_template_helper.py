import importlib
import json

import pytest

from core.lib.common import YamlOps
from core.lib.common import TaskConstant


@pytest.mark.unit
def test_process_image_fills_missing_registry_repository_and_tag(mounted_runtime):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    assert helper.process_image("generator") == "repo:5000/dayuhub/generator:v1.3"
    assert helper.process_image("custom/generator") == "repo:5000/custom/generator:v1.3"
    assert helper.process_image("ghcr.io/dayu/generator:latest") == "ghcr.io/dayu/generator:latest"


@pytest.mark.unit
def test_fill_template_builds_both_side_controller_manifest(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    yaml_doc = YamlOps.read_yaml(mounted_runtime / "controller" / "controller-base.yaml")

    manifest = helper.fill_template(yaml_doc, "controller")

    assert manifest["metadata"]["name"] == "controller"
    assert manifest["spec"]["serviceConfig"] == {"pos": "both", "port": 9000, "targetPort": 9000}
    assert "cloudWorker" in manifest["spec"]
    assert "edgeWorker" in manifest["spec"]

    cloud_container = manifest["spec"]["cloudWorker"]["template"]["spec"]["containers"][0]
    cloud_env = {item["name"]: item["value"] for item in cloud_container["env"]}
    assert cloud_container["image"] == "repo:5000/dayuhub/controller:v1.3"
    assert cloud_container["ports"] == [{"containerPort": 9000}]
    assert cloud_env["NAMESPACE"] == "dayu"
    assert cloud_env["KUBERNETES_SERVICE_HOST"] == "10.0.0.1"
    assert cloud_env["KUBERNETES_SERVICE_PORT"] == "6443"
    assert cloud_env["GUNICORN_PORT"] == "9000"
    cloud_mounts = manifest["spec"]["cloudWorker"]["mounts"]
    assert cloud_mounts[-1]["envName"] == "TEMP_PATH"
    assert cloud_mounts[-1]["target"]["path"] == "/temp"


@pytest.mark.unit
def test_temporary_mount_and_jetpack_suffix_are_stable(mounted_runtime):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    temp_mount = helper.resolve_temporary_file_mount("/data/dayu-files")
    assert temp_mount["source"]["hostPath"]["path"] == "temp/"
    assert temp_mount["source"]["hostPath"]["prefix"] == "/data/dayu-files"
    assert temp_mount["target"]["path"] == "/temp"
    assert temp_mount["envName"] == "TEMP_PATH"
    assert helper.specify_jetpack_image("repo/dayu/processor:v1", 5) == "repo/dayu/processor:v1-jp5"
    assert helper.specify_jetpack_image("repo/dayu/processor:v1", -1) == "repo/dayu/processor:v1"


@pytest.mark.unit
def test_real_camera_generator_mounts_device_to_explicit_container_path(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    yaml_doc = YamlOps.read_yaml(mounted_runtime / "generator" / "generator-for-real-camera.yaml")

    manifest = helper.fill_template(yaml_doc, "generator")

    mounts = manifest["spec"]["edgeWorker"][0]["mounts"]
    camera_mount = mounts[0]
    container = manifest["spec"]["edgeWorker"][0]["template"]["spec"]["containers"][0]
    assert camera_mount["source"]["hostPath"]["path"] == "/dev/video0"
    assert camera_mount["source"]["hostPath"]["pathType"] == "CharDevice"
    assert camera_mount["target"]["path"] == "/dev/video0"
    assert "envName" not in camera_mount
    assert container["securityContext"] == {
        "privileged": True,
        "allowPrivilegeEscalation": True,
        "runAsUser": 0,
    }


@pytest.mark.unit
def test_finetune_generator_yaml_groups_sources_by_selected_node(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    monkeypatch.setattr(helper, "request_source_selection_decision", lambda source_deploy: {0: "edgex1", 1: "edgex1"})
    yaml_doc = YamlOps.read_yaml(mounted_runtime / "generator" / "generator-base.yaml")

    dag = {
        TaskConstant.START.value: {"succ": ["face-detection"], "prev": []},
        "face-detection": {"succ": [], "prev": [TaskConstant.START.value]},
    }
    source_deploy = [
        {
            "source": {"id": 0, "source_mode": "http_video", "url": "http://a", "source_type": "video", "metadata": {"fps": 25}},
            "node_set": ["edgex1", "edgex2"],
            "dag": dag,
        },
        {
            "source": {"id": 1, "source_mode": "rtsp_video", "url": "rtsp://b", "source_type": "stream", "metadata": {"fps": 10}},
            "node_set": ["edgex1"],
            "dag": dag,
        },
    ]

    manifest = helper.finetune_generator_yaml(yaml_doc, source_deploy)

    assert len(manifest["spec"]["edgeWorker"]) == 1
    worker = manifest["spec"]["edgeWorker"][0]
    assert worker["template"]["spec"]["nodeName"] == "edgex1"
    containers = worker["template"]["spec"]["containers"]
    assert len(containers) == 2

    first_env = {item["name"]: item["value"] for item in containers[0]["env"]}
    second_env = {item["name"]: item["value"] for item in containers[1]["env"]}
    assert first_env["SOURCE_ID"] == "0"
    assert second_env["SOURCE_ID"] == "1"
    assert first_env["ALL_EDGE_DEVICES"] == "['edgex1', 'edgex2']"
    assert "face-detection" in first_env["DAG"]
    assert source_deploy[0]["source"]["source_device"] == "edgex1"
    assert source_deploy[1]["source"]["source_device"] == "edgex1"


@pytest.mark.unit
def test_finetune_distributor_and_scheduler_request_helpers_use_scheduler_contracts(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(template_helper_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_all_edge_nodes", staticmethod(lambda: ["edgex1", "edgex2"]))
    monkeypatch.setattr(template_helper_module.PortInfo, "get_component_port", staticmethod(lambda component: 9001))
    monkeypatch.setattr(template_helper_module.PortInfo, "force_refresh", staticmethod(lambda: None))

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    yaml_doc = YamlOps.read_yaml(mounted_runtime / "distributor" / "distributor-base.yaml")
    manifest = helper.finetune_distributor_yaml(yaml_doc, "cloudx1")

    cloud_worker = manifest["spec"]["cloudWorker"]
    env = {item["name"]: item["value"] for item in cloud_worker["template"]["spec"]["containers"][0]["env"]}
    assert cloud_worker["template"]["spec"]["nodeName"] == "cloudx1"
    assert env["RESULT_LOG_RETENTION_RECORDS"] == "100000"
    assert env["RESULT_LOG_RETENTION_PRUNE_INTERVAL"] == "10000"
    assert env["RESULT_LOG_EXPORT_BATCH_SIZE"] == "500"

    dag = {
        TaskConstant.START.value: {"succ": ["face-detection"], "prev": []},
        "face-detection": {"succ": [], "prev": [TaskConstant.START.value]},
    }
    source_deploy = [{"source": {"id": 9}, "node_set": ["edgex1"], "dag": dag}]

    requests = []

    def fake_http_request(url, method=None, **kwargs):
        requests.append((url, method, kwargs))
        if url.endswith("/source_nodes_selection"):
            return {"plan": {"9": "edgex1"}}
        if url.endswith("/initial_deployment"):
            return {"plan": {"face-detection": ["edgex1"]}}
        if url.endswith("/redeployment"):
            return None
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(template_helper_module, "http_request", fake_http_request)
    monkeypatch.setattr(helper, "check_is_redeployment", lambda: False)

    selection_plan = helper.request_source_selection_decision(source_deploy)
    deployment_plan = helper.request_deployment_decision(source_deploy)

    monkeypatch.setattr(helper, "check_is_redeployment", lambda: True)
    redeployment_plan = helper.request_deployment_decision(source_deploy)

    assert selection_plan == {9: "edgex1"}
    assert deployment_plan == {"face-detection": ["edgex1"]}
    assert redeployment_plan is None

    selection_payload = json.loads(requests[0][2]["data"]["data"])
    deployment_payload = json.loads(requests[1][2]["data"]["data"])
    assert selection_payload[0]["all_edge_nodes"] == ["edgex1", "edgex2"]
    assert deployment_payload[0]["dag"]["face-detection"]["service"]["service_name"] == "face-detection"


@pytest.mark.unit
def test_request_deployment_decision_returns_none_when_scheduler_errors(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(template_helper_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(template_helper_module.PortInfo, "get_component_port", staticmethod(lambda component: 9001))

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    monkeypatch.setattr(helper, "check_is_redeployment", lambda: False)

    dag = {
        TaskConstant.START.value: {"succ": ["face-detection"], "prev": []},
        "face-detection": {"succ": [], "prev": [TaskConstant.START.value]},
    }
    source_deploy = [{"source": {"id": 9}, "node_set": ["edgex1"], "dag": dag}]

    def raising_http_request(url, method=None, **kwargs):
        raise RuntimeError("scheduler failed")

    monkeypatch.setattr(template_helper_module, "http_request", raising_http_request)
    assert helper.request_deployment_decision(source_deploy) is None

    monkeypatch.setattr(template_helper_module, "http_request", lambda url, method=None, **kwargs: {"plan": ["bad"]})
    assert helper.request_deployment_decision(source_deploy) is None


@pytest.mark.unit
def test_request_deployment_decision_normalizes_node_to_services_plan(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(template_helper_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(template_helper_module.PortInfo, "get_component_port", staticmethod(lambda component: 9001))

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    monkeypatch.setattr(helper, "check_is_redeployment", lambda: False)
    monkeypatch.setattr(
        template_helper_module,
        "http_request",
        lambda url, method=None, **kwargs: {"plan": {"edgex1": ["face-detection", "unknown"]}},
    )

    dag = {
        TaskConstant.START.value: {"succ": ["face-detection"], "prev": []},
        "face-detection": {"succ": [], "prev": [TaskConstant.START.value]},
    }
    source_deploy = [{"source": {"id": 9}, "node_set": ["edgex1"], "dag": dag}]

    assert helper.request_deployment_decision(source_deploy) == {"face-detection": ["edgex1"]}
