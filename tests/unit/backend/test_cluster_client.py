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
            "capacity": {"cpu": "8"},
            "allocatable": {"cpu": "7500m"},
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
        self.agent_pods = {}
        self.calls = []

    def list_node(self, **kwargs):
        self.calls.append(("nodes", None))
        return {"items": self.nodes}

    def list_pod_for_all_namespaces(self, label_selector, **kwargs):
        self.calls.append(("agents", label_selector))
        return {"items": self.agent_pods.get(label_selector, [])}

    def list_namespaced_pod(self, namespace, **kwargs):
        self.calls.append(("pods", namespace, kwargs.get("label_selector")))
        return {"items": self.runtime_pods}


class FakeCustom:
    def __init__(self):
        self.calls = []

    def list_namespaced_custom_object(self, **kwargs):
        self.calls.append(kwargs)
        return {"items": [{
            "metadata": {"name": "runtime-a-abc"},
            "containers": [{"name": "runtime", "usage": {"cpu": "12m", "memory": "150Mi"}}],
        }]}


def make_client(core=None, custom=None):
    return ClusterClient(
        namespace="dayu",
        core_api=core or FakeCore(),
        custom_api=custom or FakeCustom(),
        load_config=False,
    )


def test_node_inventory_uses_one_list_and_preserves_roles_readiness_and_capacity():
    core = FakeCore()
    core.nodes[0]["metadata"]["labels"]["node-role.kubernetes.io/edge"] = ""
    inventory = make_client(core=core).node_inventory()
    assert core.calls == [("nodes", None)]
    assert inventory["cloud-1"]["role"] == "cloud"
    assert inventory["edge-1"]["role"] == "edge"
    assert inventory["edge-1"]["ready"] is True
    assert inventory["edge-1"]["capacity"] == {"cpu": "8"}


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


def test_runtime_metrics_batches_pods_metrics_and_nodes_and_requires_exact_uid():
    core = FakeCore()
    custom = FakeCustom()
    client = make_client(core=core, custom=custom)
    result = client.runtime_metrics([{"name": "runtime-a-abc", "uid": "pod-uid"}])

    assert [call[0] for call in core.calls] == ["pods", "nodes"]
    assert core.calls[0][2] == client.runtime_selector
    assert len(custom.calls) == 1
    assert custom.calls[0]["label_selector"] == client.runtime_selector
    assert result["runtime-a-abc"]["ready"] is True
    assert result["runtime-a-abc"]["resources"]["runtime"]["requests"]["memory"] == "100Mi"
    assert result["runtime-a-abc"]["usage"]["runtime"]["memory"] == "150Mi"
    assert result["runtime-a-abc"]["node_info"]["role"] == "edge"

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
