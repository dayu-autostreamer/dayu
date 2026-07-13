import copy
import importlib

import pytest


def source_deploy():
    return [
        {
            "source": {"id": 7, "url": "http://camera/live"},
            "node_set": ["edge-a", "edge-a", "edge-b"],
            "dag": {
                "_start": ["detect-stage"],
                "detect-stage": {
                    "id": "face-detection-pure",
                    "prev": ["_start"],
                    "succ": ["classify-stage"],
                    "custom": "kept",
                },
                "classify-stage": {
                    "id": "gender-classification-roi",
                    "prev": ["detect-stage"],
                    "succ": [],
                },
            },
        },
        {
            "source": {"id": 8, "url": "http://camera/secondary"},
            "node_set": ["edge-c"],
            "dag": {
                "_start": ["face-detection-pure"],
                "face-detection-pure": {
                    "id": "face-detection-pure",
                    "prev": [],
                    "succ": [],
                },
            },
        },
    ]


@pytest.mark.unit
def test_base_catalog_is_cached_but_callers_receive_isolated_copies(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))

    first = helper.load_base_info()
    first["namespace"] = "mutated"
    first["services"].clear()

    second = helper.load_base_info()
    assert second["namespace"] == "dayu"
    assert second["services"]
    assert not hasattr(module, "KubeHelper")
    assert not hasattr(module, "NodeInfo")
    assert not hasattr(module, "PortInfo")


@pytest.mark.unit
def test_policy_and_application_templates_are_loaded_without_mutating_catalog_input(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))
    policy = next(item for item in helper.load_base_info()["scheduler-policies"] if item["id"] == "fixed")

    policy_docs = helper.load_policy_apply_yaml(policy)
    assert set(policy_docs) == {"scheduler", "generator", "controller", "distributor", "monitor"}
    assert policy_docs["scheduler"]["position"] == "cloud"
    assert policy_docs["generator"]["pod-template"]["image"] == "generator"

    compiled = {
        "face-detection": {
            "catalog_id": "face-detection-pure",
            "service_name": "face-detection",
            "yaml": "face-detection-pure.yaml",
            "node": ["edge-a"],
        }
    }
    original = copy.deepcopy(compiled)
    loaded = helper.load_application_apply_yaml(compiled)

    assert compiled == original
    assert loaded["face-detection"]["service"]["pod-template"]["image"] == "face-detection"


@pytest.mark.unit
def test_normalize_source_deploy_maps_catalog_ids_and_preserves_graph_semantics(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))
    raw = source_deploy()
    original = copy.deepcopy(raw)

    normalized, services = helper.normalize_source_deploy(raw)

    assert raw == original
    assert normalized[0]["node_set"] == ["edge-a", "edge-b"]
    assert set(normalized[0]["dag"]) == {"face-detection", "gender-classification"}
    detector = normalized[0]["dag"]["face-detection"]
    classifier = normalized[0]["dag"]["gender-classification"]
    assert detector["id"] == "face-detection"
    assert detector["prev"] == ["_start"]
    assert detector["succ"] == ["gender-classification"]
    assert detector["custom"] == "kept"
    assert detector["service"]["id"] == "face-detection-pure"
    assert classifier["prev"] == ["face-detection"]
    assert classifier["succ"] == []

    assert set(services) == {"face-detection", "gender-classification"}
    assert services["face-detection"] == {
        "catalog_id": "face-detection-pure",
        "service_name": "face-detection",
        "yaml": "face-detection-pure.yaml",
        "node": ["edge-a", "edge-b", "edge-c"],
        "catalog": next(
            item for item in helper.load_base_info()["services"] if item["id"] == "face-detection-pure"
        ),
    }

    normalized_again, services_again = helper.normalize_source_deploy(normalized)
    assert normalized_again == normalized
    assert services_again == services


@pytest.mark.unit
def test_normalize_source_deploy_rejects_unknown_services_and_graph_references(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))

    unknown_service = source_deploy()[:1]
    unknown_service[0]["dag"]["detect-stage"]["id"] = "not-in-catalog"
    with pytest.raises(ValueError, match="unknown service id"):
        helper.normalize_source_deploy(unknown_service)

    unknown_reference = source_deploy()[:1]
    unknown_reference[0]["dag"]["detect-stage"]["succ"] = ["missing-stage"]
    with pytest.raises(ValueError, match="references unknown nodes"):
        helper.normalize_source_deploy(unknown_reference)


@pytest.mark.unit
def test_normalize_source_deploy_rejects_ambiguous_logical_service_implementations(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))
    raw = source_deploy()[:1]
    raw[0]["dag"]["detect-full"] = {
        "id": "face-detection",
        "prev": [],
        "succ": [],
    }

    with pytest.raises(ValueError, match="logical service name 'face-detection'"):
        helper.normalize_source_deploy(raw)


@pytest.mark.unit
def test_catalog_loaders_reject_missing_and_escaping_template_paths(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))

    with pytest.raises(ValueError, match="missing component templates"):
        helper.load_policy_apply_yaml({"yaml": "fixed-policy.yaml", "dependency": {}})
    with pytest.raises(ValueError, match="invalid template path segment"):
        helper.load_application_apply_yaml({"svc": {"yaml": "../base.yaml"}})


@pytest.mark.unit
def test_process_image_and_jetpack_suffix_follow_catalog_defaults(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))

    assert helper.process_image("generator") == "repo:5000/dayuhub/generator:v1.4"
    assert helper.process_image("custom/generator") == "repo:5000/custom/generator:v1.4"
    assert helper.process_image("ghcr.io/dayu/generator:latest") == "ghcr.io/dayu/generator:latest"
    with pytest.raises(ValueError, match="illegal"):
        helper.process_image("registry/repository/image:tag:extra")

    assert helper.specify_jetpack_image("repo/dayu/processor:v1", 5) == "repo/dayu/processor:v1-jp5"
    assert helper.specify_jetpack_image("repo/dayu/processor:v1", -1) == "repo/dayu/processor:v1"
    assert helper.specify_jetpack_image("repo/dayu/processor:v1", True) == "repo/dayu/processor:v1"
