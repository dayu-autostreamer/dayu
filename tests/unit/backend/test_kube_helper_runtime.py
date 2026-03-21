from datetime import datetime, timezone
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def make_doc(name="processor-demo", *, namespace="dayu", with_service_config=False):
    spec = {"foo": "bar"}
    if with_service_config:
        spec["serviceConfig"] = {"pos": "edge-a"}
    return {
        "apiVersion": "sedna.io/v1alpha1",
        "kind": "JointMultiEdgeService",
        "metadata": {"namespace": namespace, "name": name},
        "spec": spec,
    }


def make_pod(name, *, node_name="edge-a", created_at=None):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            creation_timestamp=created_at or datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
        spec=SimpleNamespace(node_name=node_name),
    )


@pytest.fixture
def kube_helper_module(monkeypatch):
    kube_helper = importlib.import_module("kube_helper")
    monkeypatch.setattr(kube_helper.config, "load_incluster_config", lambda: None)
    return kube_helper


@pytest.mark.unit
def test_kube_helper_apply_and_update_wrappers_cover_file_loading_and_failures(kube_helper_module, monkeypatch, tmp_path):
    yaml_path = tmp_path / "resource.yaml"
    yaml_path.write_text("kind: demo\n", encoding="utf-8")
    original_apply = kube_helper_module.KubeHelper.apply_custom_resources
    original_update = kube_helper_module.KubeHelper.update_custom_resources

    monkeypatch.setattr(kube_helper_module.KubeHelper, "apply_custom_resources", staticmethod(lambda docs: docs == ["doc"]))
    monkeypatch.setattr(kube_helper_module.YamlOps, "read_all_yaml", staticmethod(lambda source: ["doc"]))
    assert kube_helper_module.KubeHelper.apply_custom_resources_by_file(str(yaml_path)) is True

    captured_docs = []
    monkeypatch.setattr(kube_helper_module.KubeHelper, "update_custom_resources", staticmethod(lambda docs: captured_docs.append(docs) or True))
    assert kube_helper_module.KubeHelper.update_custom_resources_by_file(str(yaml_path)) is True
    assert captured_docs == [["doc"]]
    monkeypatch.setattr(kube_helper_module.KubeHelper, "apply_custom_resources", original_apply)
    monkeypatch.setattr(kube_helper_module.KubeHelper, "update_custom_resources", original_update)

    exceptions = []
    monkeypatch.setattr(
        kube_helper_module.client,
        "CustomObjectsApi",
        lambda: SimpleNamespace(
            create_namespaced_custom_object=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )
    monkeypatch.setattr(kube_helper_module.KubeHelper, "get_crd_plural", staticmethod(lambda kind: "jointmultiedgeservices"))
    monkeypatch.setattr(kube_helper_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))

    assert kube_helper_module.KubeHelper.apply_custom_resources([make_doc()]) is False
    assert exceptions == ["boom"]

    with pytest.raises(NotImplementedError, match="Update resource is not implemented"):
        kube_helper_module.KubeHelper.update_custom_resources([make_doc()])


@pytest.mark.unit
def test_kube_helper_delete_custom_resources_covers_success_failure_and_file_dispatch(kube_helper_module, monkeypatch, tmp_path):
    deleted_services = []
    deleted_deployments = []
    deleted_resources = []
    yaml_path = tmp_path / "resource.yaml"
    yaml_path.write_text("kind: demo\n", encoding="utf-8")

    monkeypatch.setattr(
        kube_helper_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(delete_namespaced_service=lambda name=None, namespace=None: deleted_services.append((name, namespace))),
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "AppsV1Api",
        lambda: SimpleNamespace(
            list_namespaced_deployment=lambda namespace: SimpleNamespace(
                items=[
                    SimpleNamespace(metadata=SimpleNamespace(name="processor-demo-edge-a")),
                    SimpleNamespace(metadata=SimpleNamespace(name="other-service")),
                ]
            ),
            delete_namespaced_deployment=lambda name=None, namespace=None: deleted_deployments.append((name, namespace)),
        ),
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "CustomObjectsApi",
        lambda: SimpleNamespace(
            delete_namespaced_custom_object=lambda **kwargs: deleted_resources.append(kwargs)
        ),
    )
    monkeypatch.setattr(kube_helper_module.KubeHelper, "get_crd_plural", staticmethod(lambda kind: "jointmultiedgeservices"))

    doc = make_doc(with_service_config=True)
    assert kube_helper_module.KubeHelper.delete_custom_resources([None, doc]) is True
    assert deleted_services == [("processor-demo-edge-a", "dayu")]
    assert deleted_deployments == [("processor-demo-edge-a", "dayu")]
    assert deleted_resources[0]["name"] == "processor-demo"

    monkeypatch.setattr(kube_helper_module.YamlOps, "read_all_yaml", staticmethod(lambda source: [doc]))
    assert kube_helper_module.KubeHelper.delete_custom_resources_by_file(str(yaml_path)) is True

    exceptions = []
    monkeypatch.setattr(
        kube_helper_module.client,
        "AppsV1Api",
        lambda: SimpleNamespace(
            list_namespaced_deployment=lambda namespace: (_ for _ in ()).throw(RuntimeError("delete failed"))
        ),
    )
    monkeypatch.setattr(kube_helper_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))
    assert kube_helper_module.KubeHelper.delete_custom_resources([doc]) is False
    assert exceptions == ["delete failed"]


@pytest.mark.unit
def test_kube_helper_runtime_helpers_cover_missing_resources_and_api_exceptions(kube_helper_module, monkeypatch):
    class DummyApiException(Exception):
        pass

    monkeypatch.setattr(
        kube_helper_module.client,
        "CoreV1Api",
        lambda: SimpleNamespace(
            list_namespace=lambda: (_ for _ in ()).throw(DummyApiException("unavailable")),
            list_node=lambda: SimpleNamespace(items=[]),
        ),
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "ApiextensionsV1Api",
        lambda: SimpleNamespace(list_custom_resource_definition=lambda: SimpleNamespace(items=[])),
    )
    monkeypatch.setattr(
        kube_helper_module.client,
        "exceptions",
        SimpleNamespace(ApiException=DummyApiException),
        raising=False,
    )

    assert kube_helper_module.KubeHelper.list_namespaces() == []

    with pytest.raises(Exception, match="hostname of missing-node not exists"):
        kube_helper_module.KubeHelper.get_node_cpu("missing-node")
    with pytest.raises(AssertionError, match="Crd kind MissingKind not exists"):
        kube_helper_module.KubeHelper.get_crd_plural("MissingKind")
    with pytest.raises(ValueError, match="node_name must be non-empty"):
        kube_helper_module.KubeHelper.get_node_jetpack_labels(" ")
