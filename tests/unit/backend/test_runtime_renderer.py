import copy

import pytest

from runtime_model import RuntimeSlot, canonical_hash
from runtime_renderer import RuntimeServiceRenderer
from template_helper import TemplateHelper


def env_map(container):
    return {item["name"]: item.get("value") for item in container.get("env", [])}


def processor_template():
    return {
        "position": "both",
        "pod-template": {
            "image": "face:raw",
            "imagePullPolicy": "Always",
            "env": [
                {"name": "PROCESSOR_NAME", "value": "face"},
            ],
        },
        "edge-pod-template": {"resources": {"limits": {"nvidia.com/gpu": 1}}},
        "port-open": {"pos": "both", "port": 9000},
        "file-mount": [{"pos": "both", "path": "processor/face/"}],
    }


def test_renderer_builds_complete_tokenless_runtime_service_and_native_mounts():
    renderer = RuntimeServiceRenderer(
        namespace="dayu",
        install_id="install-1",
        log_level="DEBUG",
        file_mount_prefix="/data/dayu-files",
        image_resolver=lambda image: f"registry.local/{image}",
    )
    slot = RuntimeSlot("processor", "edge-x1", "edge", logical_service="face-detection")
    rendered = renderer.render(
        processor_template(), slot, revision=7,
        extra_env={"CLOUD_NODE": "cloud-1"},
    )

    manifest = rendered.manifest
    assert manifest["apiVersion"] == "sedna.io/v1alpha1"
    assert manifest["kind"] == "RuntimeService"
    assert manifest["metadata"]["name"].endswith("-r7")
    assert manifest["spec"] == {
        **manifest["spec"],
        "installID": "install-1",
        "deploymentRevision": 7,
        "component": "processor",
        "logicalService": "face-detection",
        "targetNode": "edge-x1",
        "endpoint": {"port": 9000},
    }

    pod_spec = manifest["spec"]["podTemplate"]["spec"]
    assert pod_spec["automountServiceAccountToken"] is False
    assert pod_spec["enableServiceLinks"] is False
    assert "serviceAccountName" not in pod_spec
    assert "nodeName" not in pod_spec
    assert pod_spec["dnsPolicy"] == "ClusterFirst"
    assert pod_spec["dnsConfig"] == {
        "options": [{"name": "ndots", "value": "1"}],
    }
    assert pod_spec["restartPolicy"] == "Always"

    container = pod_spec["containers"][0]
    assert container["image"] == "registry.local/face:raw"
    assert container["resources"]["limits"] == {"nvidia.com/gpu": 1}
    env = env_map(container)
    assert env["NODE_NAME"] == "edge-x1"
    assert env["NODE_ROLE"] == "edge"
    assert env["NAMESPACE"] == "dayu"
    assert env["SERVICE_NAME"] == manifest["metadata"]["name"]
    assert env["LOG_LEVEL"] == "DEBUG"
    assert env["CLOUD_NODE"] == "cloud-1"
    assert env["GUNICORN_PORT"] == "9000"
    assert container["readinessProbe"] == {
        "tcpSocket": {"port": 9000},
        "periodSeconds": 2,
        "timeoutSeconds": 1,
        "failureThreshold": 3,
        "successThreshold": 1,
    }
    assert env["DEFAULT_MOUNT_PATH"] == "/home/data/processor/face"
    assert env["TEMP_PATH"] == "/temp"

    assert pod_spec["volumes"] == [
        {
            "name": "mount-0",
            "hostPath": {"path": "/data/dayu-files/processor/face", "type": "Directory"},
        },
        {
            "name": "temporary-directory",
            "hostPath": {"path": "/data/dayu-files/temp", "type": "DirectoryOrCreate"},
        },
    ]
    assert [item["mountPath"] for item in container["volumeMounts"]] == [
        "/home/data/processor/face", "/temp",
    ]
    assert rendered.unit.spec_hash == canonical_hash(manifest["spec"])
    assert rendered.unit.endpoint.dns_name == (
        f"{manifest['metadata']['name']}.dayu.svc.cluster.local"
    )


@pytest.mark.parametrize("forbidden_name", [
    "KUBE_CACHE_TTL",
    "KUBE_API_ENDPOINT",
    "KUBECONFIG",
    "KUBERNETES_SERVICE_PORT_HTTPS",
])
def test_renderer_rejects_kubernetes_discovery_or_cache_environment(forbidden_name):
    logical = processor_template()
    logical["pod-template"]["env"].append({
        "name": forbidden_name,
        "value": "5",
    })

    with pytest.raises(ValueError, match="must not define Kubernetes discovery/cache env"):
        RuntimeServiceRenderer("dayu", "install-1").render(
            logical,
            RuntimeSlot("processor", "edge-1", "edge", logical_service="detector"),
            1,
        )


def test_renderer_rejects_forbidden_discovery_environment_from_backend_overlay():
    with pytest.raises(ValueError, match="must not define Kubernetes discovery/cache env"):
        RuntimeServiceRenderer("dayu", "install-1").render(
            processor_template(),
            RuntimeSlot("processor", "edge-1", "edge", logical_service="detector"),
            1,
            extra_env={"KUBERNETES_SERVICE_HOST": "10.96.0.1"},
        )


def test_renderer_does_not_expose_position_mismatched_port():
    logical = {
        "position": "both",
        "pod-template": {"image": "monitor"},
        "port-open": {"pos": "cloud", "port": 9000},
    }
    renderer = RuntimeServiceRenderer("dayu", "install-1")
    edge = renderer.render(logical, RuntimeSlot("monitor", "edge-1", "edge"), 1)
    cloud = renderer.render(logical, RuntimeSlot("monitor", "cloud-1", "cloud"), 1)

    assert "endpoint" not in edge.manifest["spec"]
    assert edge.unit.endpoint is None
    assert "GUNICORN_PORT" not in env_map(
        edge.manifest["spec"]["podTemplate"]["spec"]["containers"][0]
    )
    assert cloud.manifest["spec"]["endpoint"] == {"port": 9000}
    cloud_container = cloud.manifest["spec"]["podTemplate"]["spec"]["containers"][0]
    assert "readinessProbe" not in cloud_container


def test_rollout_hash_ignores_revision_bootstrap_and_runtime_name_but_tracks_workload_changes():
    renderer = RuntimeServiceRenderer("dayu", "install-1")
    slot = RuntimeSlot("processor", "edge-1", "edge", logical_service="face")
    first = renderer.render(
        processor_template(), slot, 1,
        extra_env={"DAYU_RUNTIME_BOOTSTRAP": "revision-1"},
    )
    second = renderer.render(
        copy.deepcopy(processor_template()), slot, 2,
        extra_env={"DAYU_RUNTIME_BOOTSTRAP": "revision-2"},
    )
    changed_template = processor_template()
    changed_template["pod-template"]["image"] = "face:new"
    changed = renderer.render(
        changed_template, slot, 2,
        extra_env={"DAYU_RUNTIME_BOOTSTRAP": "revision-2"},
    )

    assert first.unit.runtime_id != second.unit.runtime_id
    assert first.unit.spec_hash != second.unit.spec_hash
    assert first.unit.rollout_hash == second.unit.rollout_hash
    assert changed.unit.rollout_hash != second.unit.rollout_hash


def test_runtime_names_isolate_immediate_reinstall_from_background_garbage_collection():
    slot = RuntimeSlot("processor", "edge-1", "edge", logical_service="face")
    first_install = RuntimeServiceRenderer("dayu", "install-1").render(
        processor_template(), slot, 1,
    )
    next_install = RuntimeServiceRenderer("dayu", "install-2").render(
        processor_template(), slot, 1,
    )

    assert first_install.unit.runtime_id != next_install.unit.runtime_id
    assert first_install.manifest["metadata"]["name"] == first_install.unit.runtime_id
    assert next_install.manifest["metadata"]["name"] == next_install.unit.runtime_id


def test_generator_renderer_creates_one_deterministic_runtime_per_source_even_on_same_node():
    logical = {
        "position": "edge",
        "pod-template": {"image": "generator", "env": [{"name": "BASE", "value": "1"}]},
    }
    source_deploy = [
        {
            "source": {
                "id": 0, "source_mode": "rtsp_video", "source_type": "camera",
                "url": "rtsp://camera/0", "metadata": {"fps": 30},
            },
            "node_set": ["edge-1", "edge-2"],
            "dag": {"start": ["face"], "face": {"prev": [], "succ": []}},
        },
        {
            "source": {
                "id": 1, "source_mode": "rtsp_video", "source_type": "camera",
                "url": "rtsp://camera/1", "metadata": {"fps": 25},
            },
            "node_set": ["edge-1"],
            "dag": {"face": {"prev": [], "succ": []}},
        },
    ]
    renderer = RuntimeServiceRenderer("dayu", "install-1")
    rendered = renderer.render_generator_sources(
        logical, source_deploy, revision=2, selected_nodes={0: "edge-1", 1: "edge-1"},
    )

    assert len(rendered) == 2
    assert len({item.manifest["metadata"]["name"] for item in rendered}) == 2
    assert all(item.manifest["spec"]["targetNode"] == "edge-1" for item in rendered)
    assert all("endpoint" not in item.manifest["spec"] for item in rendered)
    assert [item.unit.slot.source_id for item in rendered] == ["0", "1"]
    first_env = env_map(rendered[0].manifest["spec"]["podTemplate"]["spec"]["containers"][0])
    assert first_env["SOURCE_ID"] == "0"
    assert first_env["SOURCE_URL"] == "rtsp://camera/0"
    assert first_env["TASK_OFFER_LIMIT"] == "0"
    assert "face" in first_env["DAG"]

    limited_sources = copy.deepcopy(source_deploy)
    limited_sources[0]["source"]["task_offer_limit"] = 17
    limited = renderer.render_generator_sources(
        logical, limited_sources[:1], revision=2, selected_nodes={0: "edge-1"},
    )
    limited_env = env_map(
        limited[0].manifest["spec"]["podTemplate"]["spec"]["containers"][0]
    )
    assert limited_env["TASK_OFFER_LIMIT"] == "17"

    rerendered = renderer.render_generator_sources(
        copy.deepcopy(logical), copy.deepcopy(source_deploy), 2,
        selected_nodes={0: "edge-1", 1: "edge-1"},
    )
    assert [item.manifest for item in rerendered] == [item.manifest for item in rendered]


def test_renderer_rejects_unknown_mount_container_and_invalid_position():
    logical = processor_template()
    logical["file-mount"][0]["containers"] = ["missing"]
    renderer = RuntimeServiceRenderer("dayu", "install-1")
    slot = RuntimeSlot("processor", "edge-1", "edge", logical_service="face")
    with pytest.raises(ValueError, match="unknown containers"):
        renderer.render(logical, slot, 1)

    with pytest.raises(ValueError, match="cannot render"):
        renderer.render(
            {"position": "cloud", "pod-template": {"image": "scheduler"}},
            RuntimeSlot("scheduler", "edge-1", "edge"),
            1,
        )


def test_template_helper_managed_wrapper_has_no_kubernetes_dependency(monkeypatch):
    template_helper_module = __import__("template_helper")
    assert not hasattr(template_helper_module, "KubeHelper")
    assert not hasattr(template_helper_module, "NodeInfo")
    assert not hasattr(template_helper_module, "PortInfo")

    helper = TemplateHelper("/unused")
    monkeypatch.setattr(helper, "load_base_info", lambda: {
        "namespace": "dayu",
        "log-level": "INFO",
        "default-file-mount-prefix": "/data/dayu-files",
        "default-image-meta": {"registry": "repo:5000", "repository": "dayuhub", "tag": "v2"},
    })
    rendered = helper.render_runtime_service(
        {"position": "cloud", "pod-template": {"image": "scheduler"},
         "port-open": {"pos": "cloud", "port": 9000}},
        RuntimeSlot("scheduler", "cloud-1", "cloud"),
        revision=1,
        install_id="install-1",
    )
    assert rendered.manifest["spec"]["podTemplate"]["spec"]["containers"][0]["image"] == (
        "repo:5000/dayuhub/scheduler:v2"
    )
