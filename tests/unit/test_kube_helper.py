from datetime import datetime, timezone
import importlib
from types import SimpleNamespace

import pytest


def make_pod(name, phase="Running", ready=True, node_name="edgex1"):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, creation_timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc)),
        status=SimpleNamespace(phase=phase, container_statuses=[SimpleNamespace(ready=ready)]),
        spec=SimpleNamespace(node_name=node_name),
    )


def make_node(name, ip="10.0.0.1", cpu="8", labels=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels=labels or {}),
        status=SimpleNamespace(
            addresses=[SimpleNamespace(type="InternalIP", address=ip)],
            capacity={"cpu": cpu},
        ),
    )


def make_namespace(name):
    return SimpleNamespace(metadata=SimpleNamespace(name=name))


def make_crd(kind, plural):
    return SimpleNamespace(spec=SimpleNamespace(names=SimpleNamespace(kind=kind, plural=plural)))


@pytest.fixture
def kube_helper_module(monkeypatch):
    kube_helper = importlib.import_module("kube_helper")
    monkeypatch.setattr(kube_helper.config, "load_incluster_config", lambda: None)
    return kube_helper


@pytest.mark.unit
def test_kube_helper_applies_custom_resources_and_resolves_crd_plural(kube_helper_module, monkeypatch):
    created_docs = []
    custom_api = SimpleNamespace(
        create_namespaced_custom_object=lambda **kwargs: created_docs.append(kwargs) or {"ok": True}
    )
    monkeypatch.setattr(kube_helper_module.client, "CustomObjectsApi", lambda: custom_api)
    monkeypatch.setattr(
        kube_helper_module.client,
        "ApiextensionsV1Api",
        lambda: SimpleNamespace(
            list_custom_resource_definition=lambda: SimpleNamespace(
                items=[make_crd("JointMultiEdgeService", "jointmultiedgeservices")]
            )
        ),
    )

    doc = {
        "apiVersion": "sedna.io/v1alpha1",
        "kind": "JointMultiEdgeService",
        "metadata": {"namespace": "dayu", "name": "processor-demo"},
    }

    assert kube_helper_module.KubeHelper.get_crd_plural("JointMultiEdgeService") == "jointmultiedgeservices"
    assert kube_helper_module.KubeHelper.apply_custom_resources([None, doc]) is True
    assert created_docs[0]["namespace"] == "dayu"
    assert created_docs[0]["plural"] == "jointmultiedgeservices"


@pytest.mark.unit
def test_kube_helper_pod_state_queries_cover_running_and_matching_filters(kube_helper_module, monkeypatch):
    pods = SimpleNamespace(
        items=[
            make_pod("processor-face-detection-0", phase="Running", ready=True, node_name="edgex1"),
            make_pod("monitor-0", phase="Pending", ready=False, node_name="cloudx1"),
        ]
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(list_namespaced_pod=lambda namespace: pods),
    )

    assert kube_helper_module.KubeHelper.check_pods_running("dayu") is False
    assert kube_helper_module.KubeHelper.check_specific_pods_running("dayu", ["processor"]) is True
    assert kube_helper_module.KubeHelper.check_specific_pods_running("dayu", ["monitor"]) is False
    assert kube_helper_module.KubeHelper.check_pods_without_string_exists("dayu", ["frontend"]) is True
    assert kube_helper_module.KubeHelper.check_pods_with_string_exists("dayu", ["monitor"]) is True
    assert kube_helper_module.KubeHelper.check_pods_exist("dayu") is True
    assert kube_helper_module.KubeHelper.check_pod_name("processor-face-detection", "dayu") is True
    assert kube_helper_module.KubeHelper.get_pod_node("monitor", "dayu") == "cloudx1"


@pytest.mark.unit
def test_kube_helper_collects_service_info_and_node_metadata(kube_helper_module, monkeypatch):
    metrics_items = [
        {
            "metadata": {"name": "processor-face-detection-0"},
            "containers": [{"usage": {"cpu": "2000000000n", "memory": "1024Ki"}}],
        }
    ]
    pods = SimpleNamespace(items=[make_pod("processor-face-detection-0", node_name="edgex1")])
    nodes = SimpleNamespace(items=[make_node("edgex1", ip="10.0.0.8", cpu="8")])

    monkeypatch.setattr(
        kube_helper_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(
            list_namespaced_pod=lambda namespace: pods,
            list_node=lambda: nodes,
            read_node=lambda name: make_node(
                name,
                ip="10.0.0.8",
                cpu="8",
                labels={
                    "jetson.nvidia.com/jetpack.major": "6",
                    "jetson.nvidia.com/l4t.major": "36",
                },
            ),
        ),
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "CustomObjectsApi",
        lambda: SimpleNamespace(
            list_namespaced_custom_object=lambda **kwargs: {"items": metrics_items}
        ),
    )
    monkeypatch.setattr(
        kube_helper_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=1024 * 1024),
    )

    info = kube_helper_module.KubeHelper.get_service_info("processor-face-detection", "dayu")

    assert info[0]["hostname"] == "edgex1"
    assert info[0]["ip"] == "10.0.0.8"
    assert info[0]["cpu"] == "25.00%"
    assert info[0]["memory"] == "100.00%"
    assert kube_helper_module.KubeHelper.get_node_ip("edgex1") == "10.0.0.8"
    assert kube_helper_module.KubeHelper.get_node_cpu("edgex1") == 8
    assert kube_helper_module.KubeHelper.get_node_jetpack_labels("edgex1") == {
        "node": "edgex1",
        "jetpack_major": "6",
        "l4t_major": "36",
    }


@pytest.mark.unit
def test_kube_helper_lists_namespaces_and_resolves_kubernetes_endpoint(kube_helper_module, monkeypatch):
    created = []
    deleted = []
    core_v1 = SimpleNamespace(
        create_namespace=lambda body=None: created.append(body.metadata.name) or SimpleNamespace(status="created"),
        delete_namespace=lambda name=None: deleted.append(name) or SimpleNamespace(status="deleted"),
        list_namespace=lambda: SimpleNamespace(items=[make_namespace("dayu"), make_namespace("dayu-dev")]),
        read_namespaced_endpoints=lambda name, namespace: SimpleNamespace(
            subsets=[SimpleNamespace(addresses=[SimpleNamespace(ip="10.96.0.1")], ports=[SimpleNamespace(port=443)])]
        ),
    )
    monkeypatch.setattr(kube_helper_module.client, "CoreV1Api", lambda: core_v1)
    monkeypatch.setattr(
        kube_helper_module.client,
        "V1Namespace",
        lambda metadata=None: SimpleNamespace(metadata=metadata),
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "V1ObjectMeta",
        lambda name=None: SimpleNamespace(name=name),
    )

    kube_helper_module.KubeHelper.create_namespace("dayu-test")
    kube_helper_module.KubeHelper.delete_namespace("dayu-test")

    assert created == ["dayu-test"]
    assert deleted == ["dayu-test"]
    assert kube_helper_module.KubeHelper.list_namespaces("dev") == ["dayu-dev"]
    assert kube_helper_module.KubeHelper.get_kubernetes_endpoint() == {"address": "10.96.0.1", "port": 443}
