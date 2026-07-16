from types import SimpleNamespace

import pytest
from kubernetes.client.rest import ApiException

from cluster_client import ClusterClient


def node(name, role, ready=True):
    role_label = {
        "edge": "node-role.kubernetes.io/edge",
        "cloud": "node-role.kubernetes.io/control-plane",
    }.get(role)
    labels = {role_label: ""} if role_label else {}
    return {
        "metadata": {"name": name, "labels": labels},
        "status": {
            "addresses": [{"type": "InternalIP", "address": f"10.0.0.{1 if role == 'cloud' else 2}"}],
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "capacity": {"cpu": "8", "memory": "8Gi"},
            "allocatable": {"cpu": "7500m", "memory": "7Gi"},
        },
    }


def pod(name, uid, node_name, ready=True, labels=None):
    return {
        "metadata": {"name": name, "uid": uid, "labels": labels or {}},
        "spec": {
            "nodeName": node_name,
            "containers": [{
                "name": "runtime",
                "resources": {"requests": {"memory": "100Mi"}, "limits": {"memory": "200Mi"}},
            }],
        },
        "status": {
            "phase": "Running",
            "podIP": "192.168.1.2",
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": [{"name": "runtime", "ready": ready}],
        },
    }


class FakeCore:
    def __init__(self):
        self.nodes = [node("cloud-1", "cloud"), node("edge-1", "edge")]
        self.runtime_pods = [pod("runtime-a-abc", "pod-uid", "edge-1")]
        self.runtime_services = []
        self.runtime_endpoints = []
        self.agent_pods = {}
        self.calls = []
        self.request_timeouts = []

    def list_node(self, **kwargs):
        self.calls.append(("nodes", None))
        self.request_timeouts.append(kwargs.get("_request_timeout"))
        return {"items": self.nodes}

    def list_pod_for_all_namespaces(self, label_selector, **kwargs):
        self.calls.append(("agents", label_selector))
        self.request_timeouts.append(kwargs.get("_request_timeout"))
        return {"items": self.agent_pods.get(label_selector, [])}

    def list_namespaced_pod(self, namespace, **kwargs):
        self.calls.append(("pods", namespace, kwargs.get("label_selector")))
        self.request_timeouts.append(kwargs.get("_request_timeout"))
        return {"items": self.runtime_pods}

    def list_namespaced_service(self, namespace, **kwargs):
        self.calls.append(("services", namespace, kwargs.get("label_selector")))
        self.request_timeouts.append(kwargs.get("_request_timeout"))
        return {"items": self.runtime_services}

    def list_namespaced_endpoints(self, namespace, **kwargs):
        self.calls.append(("endpoints", namespace, kwargs.get("label_selector")))
        self.request_timeouts.append(kwargs.get("_request_timeout"))
        return {"items": self.runtime_endpoints}


class FakeCustom:
    def __init__(self, items=None, error=None, collections=None):
        self.calls = []
        self.error = error
        self.items = items if items is not None else [{
            "metadata": {"name": "runtime-a-abc"},
            "containers": [{"name": "runtime", "usage": {"cpu": "12m", "memory": "150Mi"}}],
        }]
        self.collections = collections or {}

    def list_namespaced_custom_object(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        key = (kwargs.get("group"), kwargs.get("version"), kwargs.get("plural"))
        if key in self.collections:
            return {"items": self.collections[key]}
        return {"items": self.items}


def make_client(core=None, custom=None):
    return ClusterClient(
        namespace="dayu",
        core_api=core or FakeCore(),
        custom_api=custom or FakeCustom(),
        load_config=False,
    )


def test_owned_api_client_disables_urllib3_retries(monkeypatch):
    configuration = SimpleNamespace(retries=None)
    created = []
    loaded = []

    monkeypatch.setattr(
        "cluster_client.client.Configuration.get_default_copy",
        lambda: configuration,
    )
    monkeypatch.setattr(
        "cluster_client.config.load_incluster_config",
        lambda client_configuration: loaded.append(client_configuration),
    )
    monkeypatch.setattr(
        "cluster_client.client.ApiClient",
        lambda configuration: created.append(configuration) or object(),
    )
    monkeypatch.setattr(
        "cluster_client.client.CoreV1Api",
        lambda api_client: FakeCore(),
    )
    monkeypatch.setattr(
        "cluster_client.client.CustomObjectsApi",
        lambda api_client: FakeCustom(),
    )

    client = ClusterClient(namespace="dayu")

    assert loaded == [configuration]
    assert created == [configuration]
    assert configuration.retries == 0
    assert client.api_client is not None


def test_node_inventory_uses_one_list_and_preserves_roles_readiness_and_capacity():
    core = FakeCore()
    core.nodes[0]["metadata"]["labels"]["node-role.kubernetes.io/edge"] = ""
    inventory = make_client(core=core).node_inventory()
    assert core.calls == [("nodes", None)]
    assert inventory["cloud-1"]["role"] == "cloud"
    assert inventory["edge-1"]["role"] == "edge"
    assert inventory["edge-1"]["ready"] is True
    assert inventory["edge-1"]["capacity"] == {"cpu": "8", "memory": "8Gi"}


def test_managed_agent_validation_reports_missing_and_not_ready_by_agent():
    core = FakeCore()
    client = make_client(core=core)
    core.agent_pods[client.sedna_lc_selector] = [pod("lc-edge", "lc-uid", "edge-1")]
    core.agent_pods[client.edgemesh_selector] = [
        pod("mesh-edge", "mesh-uid", "edge-1"),
        pod("mesh-cloud", "mesh-cloud-uid", "cloud-1", ready=False),
    ]

    report = client.validate_managed_agents(["cloud-1", "edge-1"])
    assert report["ok"] is False
    assert report["agents"]["sedna_lc"]["missing_nodes"] == ["cloud-1"]
    assert report["agents"]["edgemesh_agent"]["not_ready_nodes"] == ["cloud-1"]
    assert report["agents"]["edgemesh_agent"]["ready_nodes"] == ["edge-1"]


def test_runtime_cleanup_resources_lists_every_barrier_kind_once_with_one_selector():
    core = FakeCore()
    core.runtime_pods = [pod("runtime-a-pod", "pod-uid", "edge-1")]
    core.runtime_pods[0]["metadata"].update({
        "deletionTimestamp": "2026-07-16T00:00:00Z",
        "finalizers": ["example.io/cleanup"],
    })
    core.runtime_services = [{
        "metadata": {"name": "runtime-a", "uid": "service-uid"},
    }]
    core.runtime_endpoints = [{
        "metadata": {"name": "runtime-a", "uid": "endpoints-uid"},
    }]
    custom = FakeCustom(collections={
        ("apps", "v1", "deployments"): [{
            "metadata": {"name": "runtime-a", "uid": "deployment-uid"},
        }],
        ("apps", "v1", "replicasets"): [{
            "metadata": {"name": "runtime-a-rs", "uid": "replicaset-uid"},
        }],
        ("discovery.k8s.io", "v1", "endpointslices"): [{
            "metadata": {
                "name": "runtime-a-slice",
                "uid": "slice-uid",
                "labels": {"kubernetes.io/service-name": "runtime-a"},
            },
        }],
    })
    client = make_client(core=core, custom=custom)

    resources = client.runtime_cleanup_resources(
        "install-a",
        ownership={
            "runtime_names": ["runtime-a"],
            "pod_uids": ["pod-uid"],
        },
    )

    assert {resource["kind"] for resource in resources} == {
        "Deployment", "ReplicaSet", "Pod", "Service", "Endpoints", "EndpointSlice",
    }
    pod_resource = next(resource for resource in resources if resource["kind"] == "Pod")
    assert pod_resource["node"] == "edge-1"
    assert pod_resource["deletion_timestamp"] == "2026-07-16T00:00:00Z"
    assert pod_resource["finalizers"] == ["example.io/cleanup"]
    assert [call[0] for call in core.calls] == ["pods", "services", "endpoints"]
    assert [call["plural"] for call in custom.calls] == [
        "deployments", "replicasets", "endpointslices",
    ]
    assert all("label_selector" not in call for call in custom.calls)


def test_runtime_cleanup_resources_uses_name_uid_and_owner_closure_when_labels_are_missing():
    core = FakeCore()
    core.runtime_pods = [{
        "metadata": {
            "name": "runtime-a-pod",
            "uid": "pod-uid",
            "ownerReferences": [{"uid": "replicaset-uid"}],
        },
        "spec": {"nodeName": "edge-1"},
    }]
    core.runtime_services = [{
        "metadata": {
            "name": "renamed-service",
            "uid": "service-uid",
            "ownerReferences": [{"uid": "runtime-uid"}],
        },
    }]
    core.runtime_endpoints = [{
        "metadata": {
            "name": "renamed-service",
            "uid": "endpoints-uid",
            "ownerReferences": [{"uid": "service-uid"}],
        },
    }]
    custom = FakeCustom(collections={
        ("apps", "v1", "deployments"): [{
            "metadata": {
                "name": "renamed-deployment",
                "uid": "deployment-uid",
                "ownerReferences": [{"uid": "runtime-uid"}],
            },
        }],
        ("apps", "v1", "replicasets"): [{
            "metadata": {
                "name": "renamed-replicaset",
                "uid": "replicaset-uid",
                "ownerReferences": [{"uid": "deployment-uid"}],
            },
        }],
        ("discovery.k8s.io", "v1", "endpointslices"): [{
            "metadata": {
                "name": "renamed-slice",
                "uid": "slice-uid",
                "ownerReferences": [{"uid": "service-uid"}],
            },
        }],
    })

    resources = make_client(core=core, custom=custom).runtime_cleanup_resources(
        "install-a",
        ownership={
            "runtime_service_uids": ["runtime-uid"],
            "service_uids": ["service-uid"],
        },
    )

    assert {resource["kind"] for resource in resources} == {
        "Deployment", "ReplicaSet", "Pod", "Service", "Endpoints", "EndpointSlice",
    }


def test_runtime_cleanup_resources_caches_endpoint_slice_v1beta1_fallback():
    class LegacyEndpointSliceCustom(FakeCustom):
        def list_namespaced_custom_object(self, **kwargs):
            if (
                kwargs.get("group") == "discovery.k8s.io"
                and kwargs.get("version") == "v1"
            ):
                self.calls.append(kwargs)
                raise ApiException(status=404)
            return super().list_namespaced_custom_object(**kwargs)

    custom = LegacyEndpointSliceCustom(collections={
        ("apps", "v1", "deployments"): [],
        ("apps", "v1", "replicasets"): [],
        ("discovery.k8s.io", "v1beta1", "endpointslices"): [],
    })
    client = make_client(custom=custom)

    client.runtime_cleanup_resources("install-a")
    client.runtime_cleanup_resources("install-a")

    versions = [
        call["version"] for call in custom.calls
        if call.get("group") == "discovery.k8s.io"
    ]
    assert versions == ["v1", "v1beta1", "v1beta1"]


def test_runtime_cleanup_resources_reprobes_v1_after_cached_beta_disappears():
    class UpgradingEndpointSliceCustom(FakeCustom):
        modern = False

        def list_namespaced_custom_object(self, **kwargs):
            if kwargs.get("group") == "discovery.k8s.io":
                self.calls.append(kwargs)
                version = kwargs.get("version")
                if (not self.modern and version == "v1") or (
                    self.modern and version == "v1beta1"
                ):
                    raise ApiException(status=404)
                return {"items": []}
            return super().list_namespaced_custom_object(**kwargs)

    custom = UpgradingEndpointSliceCustom(collections={
        ("apps", "v1", "deployments"): [],
        ("apps", "v1", "replicasets"): [],
    })
    client = make_client(custom=custom)
    client.runtime_cleanup_resources("install-a")
    custom.modern = True

    client.runtime_cleanup_resources("install-a")

    versions = [
        call["version"] for call in custom.calls
        if call.get("group") == "discovery.k8s.io"
    ]
    assert versions == ["v1", "v1beta1", "v1beta1", "v1"]
    assert client._endpoint_slice_version == "v1"


def test_runtime_metrics_batches_pods_metrics_and_nodes_and_requires_exact_uid():
    core = FakeCore()
    custom = FakeCustom()
    client = make_client(core=core, custom=custom)
    assert client.runtime_selector == "dayu.io/mesh-managed=true"
    result = client.runtime_metrics([{"name": "runtime-a-abc", "uid": "pod-uid"}])

    assert [call[0] for call in core.calls] == ["pods", "nodes"]
    assert core.calls[0][2] == client.runtime_selector
    assert len(custom.calls) == 1
    assert custom.calls[0]["label_selector"] == client.runtime_selector
    assert result["runtime-a-abc"]["ready"] is True
    assert result["runtime-a-abc"]["resources"]["runtime"]["requests"]["memory"] == "100Mi"
    assert result["runtime-a-abc"]["usage"]["runtime"]["memory"] == "150Mi"
    assert result["runtime-a-abc"]["node_info"]["role"] == "edge"
    assert result["runtime-a-abc"]["resource_usage"] == {
        "cpu": {
            "status": "available",
            "usage_millicores": 12.0,
            "reference_millicores": 7500.0,
            "utilization_percent": pytest.approx(0.16),
            "basis": "node_allocatable",
        },
        "memory": {
            "status": "available",
            "usage_bytes": 150 * 1024 ** 2,
            "reference_bytes": 7 * 1024 ** 3,
            "utilization_percent": pytest.approx(150 / (7 * 1024) * 100),
            "basis": "node_allocatable",
        },
    }

    assert client.runtime_metrics([{"name": "runtime-a-abc", "uid": "stale-uid"}]) == {}


def test_runtime_metrics_reuses_supplied_inventory_and_makes_only_two_cluster_calls():
    core = FakeCore()
    custom = FakeCustom()
    client = make_client(core=core, custom=custom)
    snapshot = {
        "edge-1": {
            "name": "edge-1", "role": "edge", "ready": True,
            "address": "10.0.0.2", "labels": {},
        },
    }

    result = client.runtime_metrics(
        [{"name": "runtime-a-abc", "uid": "pod-uid"}],
        node_inventory=snapshot,
    )

    assert core.calls == [("pods", "dayu", client.runtime_selector)]
    assert len(custom.calls) == 1
    assert result["runtime-a-abc"]["node_info"] == snapshot["edge-1"]


def test_runtime_metrics_request_timeout_is_explicit_and_capped_by_client_budget():
    core = FakeCore()
    custom = FakeCustom()
    client = ClusterClient(
        namespace="dayu",
        core_api=core,
        custom_api=custom,
        load_config=False,
        request_timeout_seconds=4,
    )

    client.node_inventory(request_timeout_seconds=2.5)
    client.runtime_metrics(
        [{"name": "runtime-a-abc", "uid": "pod-uid"}],
        node_inventory={"edge-1": {}},
        request_timeout_seconds=30,
    )

    assert core.request_timeouts == [2, 4]
    assert custom.calls[-1]["_request_timeout"] == 4
    assert all(type(timeout) is int for timeout in core.request_timeouts)
    assert type(custom.calls[-1]["_request_timeout"]) is int


def test_explicit_empty_inventory_does_not_trigger_a_second_node_list():
    core = FakeCore()
    client = make_client(core=core)

    result = client.runtime_metrics(
        [{"name": "runtime-a-abc", "uid": "pod-uid"}],
        node_inventory={},
    )

    assert [call[0] for call in core.calls] == ["pods"]
    assert result["runtime-a-abc"]["node_info"] is None
    assert result["runtime-a-abc"]["resource_usage"]["cpu"] == {
        "status": "available",
        "usage_millicores": 12.0,
        "reference_millicores": None,
        "utilization_percent": None,
        "basis": "",
    }


def test_runtime_metrics_sums_all_containers_and_parses_kubernetes_quantities():
    core = FakeCore()
    core.runtime_pods[0]["spec"]["containers"].append({
        "name": "sidecar",
        "resources": {},
    })
    custom = FakeCustom(items=[{
        "metadata": {"name": "runtime-a-abc"},
        "containers": [
            {"name": "runtime", "usage": {"cpu": "500000n", "memory": "1Mi"}},
            {"name": "sidecar", "usage": {"cpu": "250u", "memory": "500Ki"}},
        ],
    }])

    summary = make_client(core=core, custom=custom).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]

    assert summary["cpu"]["usage_millicores"] == pytest.approx(0.75)
    assert summary["cpu"]["utilization_percent"] == pytest.approx(0.01)
    assert summary["memory"]["usage_bytes"] == 1024 ** 2 + 500 * 1024
    assert summary["memory"]["utilization_percent"] == pytest.approx(
        (1024 ** 2 + 500 * 1024) / (7 * 1024 ** 3) * 100
    )


def test_runtime_metrics_uses_labeled_node_capacity_fallback():
    core = FakeCore()
    core.nodes[1]["status"]["allocatable"] = {"cpu": "0", "memory": "invalid"}

    summary = make_client(core=core).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]

    assert summary["cpu"]["reference_millicores"] == 8000.0
    assert summary["cpu"]["basis"] == "node_capacity"
    assert summary["memory"]["reference_bytes"] == 8 * 1024 ** 3
    assert summary["memory"]["basis"] == "node_capacity"


@pytest.mark.parametrize(
    ("custom", "expected_status"),
    [
        (FakeCustom(items=[]), "unavailable"),
        (FakeCustom(error=RuntimeError("metrics API unavailable")), "error"),
    ],
)
def test_runtime_metrics_distinguishes_missing_samples_from_metrics_api_errors(
        custom, expected_status,
):
    summary = make_client(custom=custom).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]

    assert summary["cpu"]["status"] == expected_status
    assert summary["memory"]["status"] == expected_status
    assert summary["cpu"]["utilization_percent"] is None


def test_runtime_metrics_does_not_report_partial_or_malformed_pod_usage_as_available():
    custom = FakeCustom(items=[{
        "metadata": {"name": "runtime-a-abc"},
        "containers": [{"name": "runtime", "usage": {"cpu": "0"}}],
    }])

    summary = make_client(custom=custom).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]

    assert summary["cpu"]["status"] == "available"
    assert summary["cpu"]["usage_millicores"] == 0.0
    assert summary["cpu"]["utilization_percent"] == 0.0
    assert summary["memory"]["status"] == "unavailable"
    assert summary["memory"]["usage_bytes"] is None


@pytest.mark.parametrize("quantity", ["NaN", "Infinity", "-1m", True])
def test_runtime_metrics_rejects_non_finite_negative_and_boolean_quantities(quantity):
    custom = FakeCustom(items=[{
        "metadata": {"name": "runtime-a-abc"},
        "containers": [{
            "name": "runtime",
            "usage": {"cpu": quantity, "memory": "1Mi"},
        }],
    }])

    cpu = make_client(custom=custom).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]["cpu"]

    assert cpu["status"] == "unavailable"
    assert cpu["usage_millicores"] is None


def test_runtime_metrics_requires_every_expected_pod_container():
    core = FakeCore()
    core.runtime_pods[0]["spec"]["containers"].append({
        "name": "sidecar",
        "resources": {},
    })

    summary = make_client(core=core).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]

    assert summary["cpu"]["status"] == "unavailable"
    assert summary["memory"]["status"] == "unavailable"


def test_runtime_metrics_rejects_a_metrics_object_with_a_different_uid():
    custom = FakeCustom(items=[{
        "metadata": {"name": "runtime-a-abc", "uid": "replacement-uid"},
        "containers": [{"name": "runtime", "usage": {"cpu": "12m", "memory": "1Mi"}}],
    }])

    summary = make_client(custom=custom).runtime_metrics([
        {"name": "runtime-a-abc", "uid": "pod-uid"},
    ])["runtime-a-abc"]["resource_usage"]

    assert summary["cpu"]["status"] == "unavailable"
    assert summary["memory"]["status"] == "unavailable"
