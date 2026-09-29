import json

import pytest

from core.lib.content import Task
from core.lib.runtime import RuntimeContext, RuntimeEndpoint, RuntimeResolver


def canonical_route(component="processor", node="edge-a", service="detector"):
    slot = {"component": component, "target_node": node}
    if service:
        slot["logical_service"] = service
    return {
        "slot": slot,
        "runtime_id": f"{component}-{node}-r7",
        "runtime_revision": 7,
        "spec_hash": "sha256:test",
        "endpoint": {
            "dns_name": f"{component}-{node}.dayu.svc.cluster.local",
            "port": 9000,
            "runtime_service_uid": "runtime-service-uid",
            "service_uid": "service-uid",
            "pod_uid": "pod-uid",
        },
    }


@pytest.mark.unit
def test_runtime_context_reads_nodes_and_only_static_infrastructure(monkeypatch):
    monkeypatch.setenv("DAYU_RUNTIME_BOOTSTRAP", json.dumps({
        "install_id": "install-a",
        "runtime_directory_revision": 9,
        "local_node": "edge-a",
        "cloud_node": "cloud-a",
        "nodes": {"edge-a": {"role": "edge", "address": "10.0.0.2"}},
        "endpoints": {"scheduler": {"fqdn": "scheduler.dayu.svc", "port": 9001}},
    }))
    context = RuntimeContext.from_env()
    assert context.local_node == "edge-a"
    assert context.node_address("edge-a") == "10.0.0.2"
    assert context.directory_revision == 9
    assert context.resolve_static_endpoint("scheduler").base_url == "http://scheduler.dayu.svc:9001"
    with pytest.raises(ValueError, match="task-routed"):
        context.resolve_static_endpoint("controller")


@pytest.mark.unit
def test_runtime_resolver_understands_canonical_nested_identity_and_fails_closed():
    resolver = RuntimeResolver(RuntimeContext({"local_node": "edge-a"}))
    task_routes = [canonical_route()]
    endpoint = resolver.resolve(
        "processor", task=task_routes, target_node="edge-a", logical_service="detector", exact=True
    )
    assert endpoint.runtime_id == "processor-edge-a-r7"
    assert endpoint.deployment_revision == 7
    assert endpoint.endpoint_pod_uid == "pod-uid"
    assert endpoint.url("/predict_local") == "http://processor-edge-a.dayu.svc.cluster.local.:9000/predict_local"
    assert endpoint.fqdn == "processor-edge-a.dayu.svc.cluster.local"
    assert endpoint.to_dict()["fqdn"] == "processor-edge-a.dayu.svc.cluster.local"

    incomplete = canonical_route()
    incomplete["endpoint"].pop("pod_uid")
    with pytest.raises(ValueError, match="endpoint_pod_uid"):
        resolver.resolve(
            "processor", task=[incomplete], target_node="edge-a", logical_service="detector", exact=True
        )
    with pytest.raises(LookupError, match="runtime route missing"):
        resolver.resolve("controller", task=task_routes, target_node="edge-a", exact=True)


@pytest.mark.unit
def test_task_runtime_snapshot_round_trip_and_fork_are_immutable():
    dag = Task.extract_dag_from_dag_deployment({
        "detector": {
            "service": {"service_name": "detector", "execute_device": "edge-a"},
            "next_nodes": [],
        }
    })
    task = Task(
        source_id=1, task_id=2, source_device="edge-a", all_edge_devices=["edge-a"], dag=dag,
        runtime_directory_revision=11, runtime_routes=[canonical_route()],
    )
    restored = Task.deserialize(task.serialize())
    assert restored.get_runtime_directory_revision() == 11
    assert restored.get_runtime_routes() == [canonical_route()]
    assert restored.fork_task("detector").get_runtime_routes() == [canonical_route()]
