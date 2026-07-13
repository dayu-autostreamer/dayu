import importlib

import pytest


def _env(container):
    return {item["name"]: item.get("value") for item in container.get("env", [])}


def _helper_and_policy(mounted_runtime):
    module = importlib.import_module("template_helper")
    helper = module.TemplateHelper(str(mounted_runtime))
    policy = next(item for item in helper.load_base_info()["scheduler-policies"] if item["id"] == "fixed")
    return helper, helper.load_policy_apply_yaml(policy)


@pytest.mark.unit
def test_runtime_renderer_wrapper_emits_tokenless_runtime_service(mounted_runtime):
    helper, templates = _helper_and_policy(mounted_runtime)

    rendered = helper.render_runtime_service(
        templates["controller"],
        {"component": "controller", "target_node": "edge-a", "position": "edge"},
        revision=3,
        install_id="install-a",
        extra_env={"DAYU_RUNTIME_BOOTSTRAP": "{}"},
    )

    manifest = rendered.manifest
    assert manifest["apiVersion"] == "sedna.io/v1alpha1"
    assert manifest["kind"] == "RuntimeService"
    assert manifest["metadata"]["namespace"] == "dayu"
    assert manifest["spec"]["targetNode"] == "edge-a"
    assert manifest["spec"]["deploymentRevision"] == 3
    assert manifest["spec"]["endpoint"] == {"port": 9000}
    pod_spec = manifest["spec"]["podTemplate"]["spec"]
    assert pod_spec["automountServiceAccountToken"] is False
    assert "serviceAccountName" not in pod_spec
    assert "nodeName" not in pod_spec

    container = pod_spec["containers"][0]
    env = _env(container)
    assert container["image"] == "repo:5000/dayuhub/controller:v1.4"
    assert env["DAYU_RUNTIME_BOOTSTRAP"] == "{}"
    assert env["NODE_NAME"] == "edge-a"
    assert env["GUNICORN_PORT"] == "9000"
    assert not ({"KUBERNETES_SERVICE_HOST", "KUBERNETES_SERVICE_PORT", "KUBE_CACHE_TTL"} & set(env))
    assert rendered.unit.endpoint.dns_name.endswith(".dayu.svc.cluster.local")


@pytest.mark.unit
def test_create_runtime_renderer_uses_base_namespace_log_level_and_mount_prefix(mounted_runtime):
    helper, templates = _helper_and_policy(mounted_runtime)
    renderer = helper.create_runtime_renderer("install-b")

    rendered = renderer.render(
        templates["monitor"],
        importlib.import_module("runtime_model").RuntimeSlot("monitor", "cloud-a", "cloud"),
        revision=1,
    )

    manifest = rendered.manifest
    container = manifest["spec"]["podTemplate"]["spec"]["containers"][0]
    env = _env(container)
    assert manifest["metadata"]["labels"]["dayu.io/install-id"] == "install-b"
    assert env["LOG_LEVEL"] == "DEBUG"
    temp_volume = next(
        volume for volume in manifest["spec"]["podTemplate"]["spec"]["volumes"]
        if volume["name"] == "temporary-directory"
    )
    assert temp_volume["hostPath"]["path"] == "/data/dayu-files/temp"


@pytest.mark.unit
def test_generator_renderer_wrapper_creates_one_exactly_placed_runtime_per_source(mounted_runtime):
    helper, templates = _helper_and_policy(mounted_runtime)
    raw = [
        {
            "source": {
                "id": 7,
                "url": "http://camera/live",
                "source_mode": "http_video",
                "source_type": "video",
                "metadata": {"fps": 25},
            },
            "node_set": ["edge-a", "edge-b"],
            "dag": {
                "_start": ["detect"],
                "detect": {
                    "id": "face-detection-pure",
                    "prev": [],
                    "succ": ["classify"],
                },
                "classify": {
                    "id": "gender-classification-roi",
                    "prev": ["detect"],
                    "succ": [],
                },
            },
        }
    ]
    normalized, _ = helper.normalize_source_deploy(raw)

    rendered = helper.render_generator_runtime_services(
        templates["generator"],
        normalized,
        revision=4,
        install_id="install-c",
        selected_nodes={7: "edge-b"},
        common_env={"DAYU_RUNTIME_BOOTSTRAP": "bootstrap"},
    )

    assert len(rendered) == 1
    manifest = rendered[0].manifest
    assert manifest["spec"]["component"] == "generator"
    assert manifest["spec"]["targetNode"] == "edge-b"
    assert "endpoint" not in manifest["spec"]
    container = manifest["spec"]["podTemplate"]["spec"]["containers"][0]
    env = _env(container)
    assert env["SOURCE_ID"] == "7"
    assert env["ALL_EDGE_DEVICES"] == "['edge-a', 'edge-b']"
    assert "face-detection" in env["DAG"]
    assert "gender-classification" in env["DAG"]
    assert env["DAYU_RUNTIME_BOOTSTRAP"] == "bootstrap"


@pytest.mark.unit
def test_runtime_renderer_wrapper_rejects_position_mismatch(mounted_runtime):
    helper, templates = _helper_and_policy(mounted_runtime)

    with pytest.raises(ValueError, match="cannot render"):
        helper.render_runtime_service(
            templates["generator"],
            {"component": "generator", "target_node": "cloud-a", "position": "cloud", "source_id": "7"},
            revision=1,
            install_id="install-d",
        )
