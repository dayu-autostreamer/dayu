import copy
import importlib
from pathlib import Path

import pytest

from core.lib.common import YamlOps

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_source_deploy():
    return [
        {
            "source": {
                "id": 0,
                "name": "camera-0",
                "url": "http://127.0.0.1/video0",
                "source_type": "video",
                "source_mode": "http_video",
                "metadata": {"location": "intersection-a"},
            },
            "node_set": ["edgex1", "edgex2"],
            "dag": {
                "_start": ["face-detection"],
                "face-detection": {
                    "id": "face-detection",
                    "prev": [],
                    "succ": ["gender-classification"],
                },
                "gender-classification": {
                    "id": "gender-classification",
                    "prev": ["face-detection"],
                    "succ": [],
                },
            },
        }
    ]


@pytest.fixture
def real_backend_core(mounted_runtime, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    template_helper_module = importlib.import_module("template_helper")

    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pod_name",
        staticmethod(lambda *args, **kwargs: False),
    )
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_kubernetes_endpoint",
        staticmethod(lambda: {"address": "10.0.0.1", "port": 6443}),
    )
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "check_pods_with_string_exists",
        staticmethod(lambda *args, **kwargs: False),
    )
    monkeypatch.setattr(
        template_helper_module.KubeHelper,
        "get_node_jetpack_labels",
        staticmethod(
            lambda node_name: {
                "node": node_name,
                "jetpack_major": "5" if node_name == "edgex2" else None,
                "l4t_major": None,
            }
        ),
    )
    monkeypatch.setattr(
        template_helper_module.NodeInfo,
        "get_cloud_node",
        staticmethod(lambda: "cloud-master"),
    )

    core = backend_core_module.BackendCore()
    monkeypatch.setattr(
        core.template_helper,
        "request_source_selection_decision",
        lambda source_deploy: {0: "edgex1"},
    )
    monkeypatch.setattr(
        core.template_helper,
        "request_deployment_decision",
        lambda source_deploy: {
            "face-detection": ["edgex1", "edgex2"],
            "gender-classification": ["edgex2"],
        },
    )
    return core


@pytest.mark.e2e
def test_declared_scheduler_and_service_templates_are_loadable():
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(REPO_ROOT / "template"))
    base_info = helper.load_base_info()

    for policy in base_info["scheduler-policies"]:
        policy_docs = helper.load_policy_apply_yaml(policy)
        assert "scheduler" in policy_docs
        assert set(policy["dependency"].keys()).issubset(policy_docs.keys())

    for service in base_info["services"]:
        processor_doc = YamlOps.read_yaml(REPO_ROOT / "template" / "processor" / service["yaml"])
        assert processor_doc["position"] in {"cloud", "edge", "both"}


@pytest.mark.e2e
def test_end_to_end_template_rendering_covers_core_components_and_processors(real_backend_core):
    source_deploy = make_source_deploy()
    policy = real_backend_core.find_scheduler_policy_by_id("fixed")

    service_dict = real_backend_core.extract_service_from_source_deployment(copy.deepcopy(source_deploy))
    yaml_dict = real_backend_core.template_helper.load_policy_apply_yaml(policy)
    yaml_dict["processor"] = real_backend_core.template_helper.load_application_apply_yaml(service_dict)

    docs = real_backend_core.template_helper.finetune_yaml_parameters(
        copy.deepcopy(yaml_dict),
        copy.deepcopy(source_deploy),
    )

    names = {doc["metadata"]["name"] for doc in docs}
    assert {
        "generator",
        "controller",
        "distributor",
        "scheduler",
        "monitor",
        "processor-face-detection-cloudmaster",
        "processor-face-detection-edgex1",
        "processor-face-detection-edgex2",
        "processor-gender-classification-cloudmaster",
        "processor-gender-classification-edgex2",
    }.issubset(names)

    generator_doc = next(doc for doc in docs if doc["metadata"]["name"] == "generator")
    assert generator_doc["spec"]["edgeWorker"][0]["template"]["spec"]["nodeName"] == "edgex1"

    edge_processor_doc = next(doc for doc in docs if doc["metadata"]["name"] == "processor-face-detection-edgex2")
    edge_container = edge_processor_doc["spec"]["edgeWorker"][0]["template"]["spec"]["containers"][0]
    edge_env = {item["name"]: item["value"] for item in edge_container["env"]}
    assert edge_container["image"].endswith("-jp5")
    assert edge_env["JETPACK"] == "5"
    assert edge_env["PROCESSOR_SERVICE_NAME"] == "processor-face-detection"


@pytest.mark.e2e
def test_generator_can_run_on_external_edge_node_without_widening_processing_pool(real_backend_core, monkeypatch):
    source_deploy = make_source_deploy()
    policy = real_backend_core.find_scheduler_policy_by_id("fixed")

    monkeypatch.setattr(
        real_backend_core.template_helper,
        "request_source_selection_decision",
        lambda source_deploy: {0: "edge-free"},
    )

    service_dict = real_backend_core.extract_service_from_source_deployment(copy.deepcopy(source_deploy))
    yaml_dict = real_backend_core.template_helper.load_policy_apply_yaml(policy)
    yaml_dict["processor"] = real_backend_core.template_helper.load_application_apply_yaml(service_dict)

    docs = real_backend_core.template_helper.finetune_yaml_parameters(
        copy.deepcopy(yaml_dict),
        copy.deepcopy(source_deploy),
    )

    generator_doc = next(doc for doc in docs if doc["metadata"]["name"] == "generator")
    generator_worker = generator_doc["spec"]["edgeWorker"][0]
    generator_container = generator_worker["template"]["spec"]["containers"][0]
    generator_env = {item["name"]: item["value"] for item in generator_container["env"]}

    processor_names = {doc["metadata"]["name"] for doc in docs if doc["metadata"]["name"].startswith("processor-")}

    assert generator_worker["template"]["spec"]["nodeName"] == "edge-free"
    assert generator_env["ALL_EDGE_DEVICES"] == "['edgex1', 'edgex2']"
    assert "processor-face-detection-edge-free" not in processor_names
