import copy
import importlib

import pytest

from core.lib.common import TaskConstant, YamlOps


def build_source_deploy():
    dag = {
        TaskConstant.START.value: {"succ": ["face-detection"], "prev": []},
        "face-detection": {"succ": [], "prev": [TaskConstant.START.value]},
    }
    return [
        {
            "source": {
                "id": 7,
                "url": "http://camera/live",
                "source_mode": "http_video",
                "source_type": "video",
                "metadata": {"fps": 25},
            },
            "node_set": ["edgex1", "edgex2"],
            "dag": dag,
        }
    ]


@pytest.mark.unit
def test_template_helper_loads_policy_and_application_yaml_from_template_catalog(mounted_runtime):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    policy = YamlOps.read_yaml(mounted_runtime / "scheduler_policies.yaml")[0]
    policy_docs = helper.load_policy_apply_yaml(policy)

    assert set(policy_docs) == {"scheduler", "generator", "controller", "distributor", "monitor"}
    assert policy_docs["scheduler"]["position"] == "cloud"
    assert policy_docs["generator"]["pod-template"]["image"] == "generator"

    service_dict = {
        "face-detection": {
            "yaml": "face-detection.yaml",
            "service_name": "face-detection",
            "node": ["edgex1"],
        }
    }
    loaded_services = helper.load_application_apply_yaml(copy.deepcopy(service_dict))

    assert loaded_services["face-detection"]["service"]["pod-template"]["image"] == "face-detection"


@pytest.mark.unit
def test_template_helper_dispatches_only_requested_scopes(monkeypatch, mounted_runtime):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    called = []
    monkeypatch.setattr(helper, "get_all_selected_edge_nodes", lambda yaml_dict: ["edgex1"])
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(helper, "finetune_generator_yaml", lambda yaml_doc, source_deploy: called.append("generator") or "GEN")
    monkeypatch.setattr(helper, "finetune_controller_yaml", lambda yaml_doc, edge_nodes, cloud_node: called.append("controller") or "CTRL")
    monkeypatch.setattr(helper, "finetune_distributor_yaml", lambda yaml_doc, cloud_node: called.append("distributor") or "DIST")
    monkeypatch.setattr(helper, "finetune_scheduler_yaml", lambda yaml_doc, cloud_node: called.append("scheduler") or "SCH")
    monkeypatch.setattr(helper, "finetune_monitor_yaml", lambda yaml_doc, edge_nodes, cloud_node: called.append("monitor") or "MON")
    monkeypatch.setattr(helper, "finetune_processor_yaml", lambda yaml_doc, cloud_node, source_deploy: called.append("processor") or ["PROC"])

    yaml_dict = {
        "generator": {},
        "controller": {},
        "distributor": {},
        "scheduler": {},
        "monitor": {},
        "processor": {},
    }

    docs = helper.finetune_yaml_parameters(yaml_dict, build_source_deploy(), scopes=["generator", "processor"])

    assert docs == ["GEN", "PROC"]
    assert called == ["generator", "processor"]


@pytest.mark.unit
def test_template_helper_finetunes_controller_and_monitor_nodes_and_jetpack_env(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    controller_doc = YamlOps.read_yaml(mounted_runtime / "controller" / "controller-base.yaml")
    controller_manifest = helper.finetune_controller_yaml(controller_doc, ["edgex1", "edgex2"], "cloudx1")

    assert [worker["template"]["spec"]["nodeName"] for worker in controller_manifest["spec"]["edgeWorker"]] == [
        "edgex1",
        "edgex2",
    ]
    assert controller_manifest["spec"]["cloudWorker"]["template"]["spec"]["nodeName"] == "cloudx1"

    monitor_doc = YamlOps.read_yaml(mounted_runtime / "monitor" / "monitor-base.yaml")
    monkeypatch.setattr(
        helper,
        "get_device_jetpack_major_version",
        lambda node_name: {"edgex1": 5, "edgex2": -1}[node_name],
    )
    monitor_manifest = helper.finetune_monitor_yaml(monitor_doc, ["edgex1", "edgex2"], "cloudx1")

    edge_workers = monitor_manifest["spec"]["edgeWorker"]
    first_env = {item["name"]: item["value"] for item in edge_workers[0]["template"]["spec"]["containers"][0]["env"]}
    second_env = {item["name"]: item["value"] for item in edge_workers[1]["template"]["spec"]["containers"][0]["env"]}

    assert edge_workers[0]["template"]["spec"]["containers"][0]["image"].endswith("-jp5")
    assert edge_workers[1]["template"]["spec"]["containers"][0]["image"].endswith(":v1.3")
    assert first_env["JETPACK"] == "5"
    assert second_env["JETPACK"] == "-1"
    assert monitor_manifest["spec"]["cloudWorker"]["template"]["spec"]["nodeName"] == "cloudx1"


@pytest.mark.unit
def test_template_helper_finetunes_processor_manifests_per_cloud_and_selected_edge_node(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )

    helper = template_helper_module.TemplateHelper(str(mounted_runtime))
    monkeypatch.setattr(helper, "request_deployment_decision", lambda source_deploy: {"face-detection": ["edgex2", "ghost"]})
    monkeypatch.setattr(helper, "get_device_jetpack_major_version", lambda node_name: 5 if node_name == "edgex2" else -1)

    service_dict = {
        "face-detection": {
            "service_name": "face-detection",
            "node": ["edgex1", "edgex2"],
            "service": YamlOps.read_yaml(mounted_runtime / "processor" / "face-detection.yaml"),
        }
    }

    manifests = helper.finetune_processor_yaml(service_dict, "cloudx1", build_source_deploy())

    assert len(manifests) == 2

    cloud_manifest = next(doc for doc in manifests if doc["metadata"]["name"] == "processor-face-detection-cloudx1")
    edge_manifest = next(doc for doc in manifests if doc["metadata"]["name"] == "processor-face-detection-edgex2")

    cloud_env = {
        item["name"]: item["value"]
        for item in cloud_manifest["spec"]["cloudWorker"]["template"]["spec"]["containers"][0]["env"]
    }
    edge_env = {
        item["name"]: item["value"]
        for item in edge_manifest["spec"]["edgeWorker"][0]["template"]["spec"]["containers"][0]["env"]
    }

    assert "edgeWorker" not in cloud_manifest["spec"]
    assert "cloudWorker" not in edge_manifest["spec"]
    assert cloud_manifest["spec"]["cloudWorker"]["template"]["spec"]["nodeName"] == "cloudx1"
    assert edge_manifest["spec"]["edgeWorker"][0]["template"]["spec"]["nodeName"] == "edgex2"
    assert cloud_env["PROCESSOR_SERVICE_NAME"] == "processor-face-detection"
    assert edge_env["PROCESSOR_SERVICE_NAME"] == "processor-face-detection"
    assert edge_env["JETPACK"] == "5"


@pytest.mark.unit
def test_template_helper_handles_invalid_image_selection_redeployment_and_jetpack_fallbacks(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    with pytest.raises(ValueError, match="illegal"):
        helper.process_image("registry/repository/image:tag:extra")

    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "check_pods_with_string_exists",
        staticmethod(lambda namespace, include_str_list: namespace == "dayu" and include_str_list == ["processor"]),
    )
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_node_jetpack_labels",
        staticmethod(lambda node_name: {"jetpack_major": "broken"}),
    )

    assert helper.check_is_redeployment() is True
    assert helper.get_device_jetpack_major_version("edgex1") == -1
    assert set(
        helper.get_all_selected_edge_nodes(
            {"processor": {"svc-a": {"node": ["edgex1", "edgex2"]}, "svc-b": {"node": ["edgex2"]}}}
        )
    ) == {"edgex1", "edgex2"}


@pytest.mark.unit
def test_template_helper_source_selection_returns_none_when_scheduler_is_unavailable(mounted_runtime, monkeypatch):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    requests = []
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(template_helper_module.NodeInfo, "get_all_edge_nodes", staticmethod(lambda: ["edgex1", "edgex2"]))
    monkeypatch.setattr(template_helper_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(template_helper_module.PortInfo, "force_refresh", staticmethod(lambda: requests.append("refresh")))
    monkeypatch.setattr(template_helper_module.PortInfo, "get_component_port", staticmethod(lambda component: 9001))
    monkeypatch.setattr(template_helper_module, "http_request", lambda **kwargs: requests.append(kwargs) or None)

    selection_plan = helper.request_source_selection_decision(build_source_deploy())

    assert selection_plan is None
    request = next(item for item in requests if isinstance(item, dict))
    assert request["method"] == template_helper_module.NetworkAPIMethod.SCHEDULER_SELECT_SOURCE_NODES
    assert "all_edge_nodes" in request["data"]["data"]
