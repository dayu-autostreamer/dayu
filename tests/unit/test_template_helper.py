import importlib

import pytest

from core.lib.common import YamlOps


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
    assert cloud_env["FILE_PREFIX"] == "/data/dayu-files"
    assert manifest["spec"]["cloudWorker"]["file"]["paths"] == ["/data/dayu-files/temp/"]


@pytest.mark.unit
def test_prepare_file_path_and_jetpack_suffix_are_stable(mounted_runtime):
    template_helper_module = importlib.import_module("template_helper")
    helper = template_helper_module.TemplateHelper(str(mounted_runtime))

    assert helper.prepare_file_path("processor/face-detection") == "/data/dayu-files/processor/face-detection/"
    assert helper.specify_jetpack_image("repo/dayu/processor:v1", 5) == "repo/dayu/processor:v1-jp5"
    assert helper.specify_jetpack_image("repo/dayu/processor:v1", -1) == "repo/dayu/processor:v1"
