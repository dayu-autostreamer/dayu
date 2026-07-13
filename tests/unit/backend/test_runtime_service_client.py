import copy

import pytest
from kubernetes.client.rest import ApiException

from runtime_model import RuntimeEndpoint, RuntimeSlot, RuntimeUnit

from runtime_service_client import (
    RUNTIME_GROUP,
    RUNTIME_PLURAL,
    RUNTIME_VERSION,
    RuntimeServiceClient,
    RuntimeServiceInvalidStatus,
    RuntimeServiceRejected,
    RuntimeServiceTimeout,
)


def runtime_obj(name="runtime-a", revision=3, ready=True, accepted=True, resource_version="4"):
    status = "True" if ready else "False"
    return {
        "apiVersion": "sedna.io/v1alpha1",
        "kind": "RuntimeService",
        "metadata": {"name": name, "namespace": "dayu", "generation": 1, "uid": "runtime-uid",
                     "resourceVersion": resource_version},
        "spec": {"deploymentRevision": revision},
        "status": {
            "observedGeneration": 1,
            "observedRevision": revision,
            "observedSpecHash": "b" * 64,
            "podRef": {"name": f"{name}-pod", "uid": "pod-uid"},
            "endpoint": {
                "serviceRef": {"name": name, "uid": "service-uid"},
                "dnsName": f"{name}.dayu.svc.cluster.local",
                "port": 9000,
            },
            "conditions": [
                {"type": "SpecAccepted", "status": "True" if accepted else "False",
                 "reason": "Valid" if accepted else "Invalid", "message": "bad spec"},
                {"type": "Ready", "status": status},
                {"type": "Activated", "status": status},
            ],
        },
    }


class FakeAPI:
    def __init__(self, items=()):
        self.items = list(items)
        self.calls = []

    def create_namespaced_custom_object(self, **kwargs):
        self.calls.append(("create", kwargs))
        return copy.deepcopy(kwargs["body"])

    def get_namespaced_custom_object(self, **kwargs):
        self.calls.append(("get", kwargs))
        for item in self.items:
            if item["metadata"]["name"] == kwargs["name"]:
                return item
        raise ApiException(status=404)

    def list_namespaced_custom_object(self, **kwargs):
        self.calls.append(("list", kwargs))
        return {"metadata": {"resourceVersion": "3"}, "items": copy.deepcopy(self.items)}

    def delete_namespaced_custom_object(self, **kwargs):
        self.calls.append(("delete", kwargs))
        return {}


class FakeWatch:
    def __init__(self, events):
        self.events = events
        self.stopped = False
        self.stream_calls = []

    def stream(self, func, **kwargs):
        self.stream_calls.append((func, kwargs))
        yield from copy.deepcopy(self.events)

    def stop(self):
        self.stopped = True


class ErrorWatch(FakeWatch):
    def __init__(self, status):
        super().__init__(())
        self.status = status

    def stream(self, func, **kwargs):
        self.stream_calls.append((func, kwargs))
        raise ApiException(status=self.status)


def test_client_requires_backend_shared_custom_api():
    with pytest.raises(ValueError, match="shared ClusterClient"):
        RuntimeServiceClient("dayu", api=None)


def test_client_uses_fixed_gvr_and_uid_guarded_delete():
    api = FakeAPI()
    client = RuntimeServiceClient("dayu", api=api)
    manifest = runtime_obj()
    manifest.pop("status")
    manifest["metadata"].pop("generation")
    manifest["metadata"].pop("resourceVersion")

    client.create(manifest)
    client.delete("runtime-a", uid="runtime-uid")

    create_kwargs = api.calls[0][1]
    assert (create_kwargs["group"], create_kwargs["version"], create_kwargs["plural"]) == (
        RUNTIME_GROUP, RUNTIME_VERSION, RUNTIME_PLURAL,
    )
    delete_kwargs = api.calls[1][1]
    assert delete_kwargs["body"]["preconditions"] == {"uid": "runtime-uid"}
    assert delete_kwargs["body"]["propagationPolicy"] == "Foreground"


def test_wait_returns_immediately_for_exact_observed_ready_revision():
    api = FakeAPI([runtime_obj()])
    client = RuntimeServiceClient(
        "dayu", api=api, watch_factory=lambda: pytest.fail("watch should not start"),
    )
    result = client.wait_for_conditions({"runtime-a": 3})
    assert result["runtime-a"]["status"]["observedRevision"] == 3


def test_wait_requires_authoritative_observed_spec_hash():
    obj = runtime_obj()
    obj["status"]["observedSpecHash"] = ""
    api = FakeAPI([obj])
    client = RuntimeServiceClient("dayu", api=api)

    with pytest.raises(RuntimeServiceInvalidStatus, match="observedSpecHash"):
        client.wait_for_conditions({"runtime-a": 3})


def test_bind_observed_unit_commits_status_hash_and_exact_endpoint_uids():
    slot = RuntimeSlot("processor", "edge-1", "edge", logical_service="face")
    planned = RuntimeUnit(
        slot=slot,
        runtime_id="runtime-a",
        runtime_revision=3,
        spec_hash="a" * 64,
        endpoint=RuntimeEndpoint("runtime-a.dayu.svc.cluster.local", 9000),
    )

    committed = RuntimeServiceClient.bind_observed_unit(planned, runtime_obj())

    assert planned.spec_hash == "a" * 64
    assert committed.spec_hash == "b" * 64
    assert committed.endpoint.runtime_service_uid == "runtime-uid"
    assert committed.endpoint.service_uid == "service-uid"
    assert committed.endpoint.pod_uid == "pod-uid"
    assert committed.runtime_service_uid == "runtime-uid"
    assert committed.pod_name == "runtime-a-pod"
    assert committed.pod_uid == "pod-uid"


def test_bind_endpointless_unit_keeps_hidden_workload_identity_out_of_directory_route():
    planned = RuntimeUnit(
        slot=RuntimeSlot("monitor", "edge-1", "edge"),
        runtime_id="runtime-a",
        runtime_revision=3,
        spec_hash="a" * 64,
    )
    obj = runtime_obj()
    obj["status"].pop("endpoint")

    committed = RuntimeServiceClient.bind_observed_unit(planned, obj)

    assert committed.endpoint is None
    assert committed.runtime_service_uid == "runtime-uid"
    assert committed.pod_name == "runtime-a-pod"
    assert committed.pod_uid == "pod-uid"
    assert "resource_identity" not in committed.to_dict()
    assert committed.to_state_dict()["resource_identity"] == {
        "runtime_service_uid": "runtime-uid",
        "pod_name": "runtime-a-pod",
        "pod_uid": "pod-uid",
    }


def test_wait_follows_one_watch_from_list_resource_version():
    api = FakeAPI([runtime_obj(ready=False, resource_version="3")])
    watcher = FakeWatch([{"type": "MODIFIED", "object": runtime_obj(resource_version="4")}])
    client = RuntimeServiceClient("dayu", api=api, watch_factory=lambda: watcher)

    result = client.wait_for_conditions(
        {"runtime-a": 3}, timeout_seconds=2, label_selector="dayu.io/install-id=install-1",
    )
    assert result["runtime-a"]["metadata"]["resourceVersion"] == "4"
    assert watcher.stream_calls[0][1]["resource_version"] == "3"
    assert watcher.stream_calls[0][1]["label_selector"] == "dayu.io/install-id=install-1"
    assert watcher.stopped is True


def test_wait_fails_fast_on_rejected_spec():
    api = FakeAPI([runtime_obj(ready=False, accepted=False)])
    client = RuntimeServiceClient("dayu", api=api)
    with pytest.raises(RuntimeServiceRejected, match="Invalid: bad spec"):
        client.wait_for_conditions({"runtime-a": 3})


def test_wait_rejects_stale_observed_revision(monkeypatch):
    obj = runtime_obj()
    obj["status"]["observedRevision"] = 2
    api = FakeAPI([obj])
    watcher = FakeWatch([])
    client = RuntimeServiceClient("dayu", api=api, watch_factory=lambda: watcher)
    times = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr("runtime_service_client.time.monotonic", lambda: next(times, 2.0))
    with pytest.raises(RuntimeServiceTimeout, match="runtime-a"):
        client.wait_for_conditions({"runtime-a": 3}, timeout_seconds=1)


def test_wait_replaces_snapshot_after_410_instead_of_retaining_deleted_ready_object():
    class SequencedAPI(FakeAPI):
        def __init__(self):
            super().__init__()
            self.responses = [
                {
                    "metadata": {"resourceVersion": "3"},
                    "items": [runtime_obj("runtime-a", ready=True), runtime_obj("runtime-b", ready=False)],
                },
                {
                    "metadata": {"resourceVersion": "5"},
                    "items": [runtime_obj("runtime-b", ready=True, resource_version="5")],
                },
            ]

        def list_namespaced_custom_object(self, **kwargs):
            self.calls.append(("list", kwargs))
            return copy.deepcopy(self.responses.pop(0))

    api = SequencedAPI()
    gone = ErrorWatch(410)
    resumed = FakeWatch([{
        "type": "ADDED",
        "object": runtime_obj("runtime-a", ready=True, resource_version="6"),
    }])
    watches = iter((gone, resumed))
    client = RuntimeServiceClient(
        "dayu", api=api, watch_factory=lambda: next(watches),
    )

    result = client.wait_for_conditions(
        {"runtime-a": 3, "runtime-b": 3}, timeout_seconds=2,
    )

    assert result["runtime-a"]["metadata"]["resourceVersion"] == "6"
    assert result["runtime-b"]["metadata"]["resourceVersion"] == "5"
    assert gone.stopped is True
    assert resumed.stream_calls[0][1]["resource_version"] == "5"


def test_delete_many_uses_uid_guarded_deletes_and_one_shared_watch():
    first = runtime_obj("runtime-a", resource_version="3")
    second = runtime_obj("runtime-b", resource_version="3")
    api = FakeAPI([first, second])
    watcher = FakeWatch([
        {"type": "DELETED", "object": runtime_obj("runtime-a", resource_version="4")},
        {"type": "DELETED", "object": runtime_obj("runtime-b", resource_version="5")},
    ])
    client = RuntimeServiceClient(
        "dayu", api=api, watch_factory=lambda: watcher,
    )

    assert client.delete_many(
        {"runtime-a": "uid-a", "runtime-b": "uid-b"},
        label_selector="app.kubernetes.io/managed-by=dayu-backend",
    ) is True

    deletes = [kwargs for operation, kwargs in api.calls if operation == "delete"]
    lists = [kwargs for operation, kwargs in api.calls if operation == "list"]
    assert [item["name"] for item in deletes] == ["runtime-a", "runtime-b"]
    assert [item["body"]["preconditions"]["uid"] for item in deletes] == ["uid-a", "uid-b"]
    assert len(lists) == 1
    assert len(watcher.stream_calls) == 1
    assert watcher.stream_calls[0][1]["resource_version"] == "3"


def test_delete_many_refuses_to_fall_back_to_name_only_deletion():
    api = FakeAPI()
    client = RuntimeServiceClient("dayu", api=api)

    with pytest.raises(ValueError, match="immutable UIDs.*runtime-a"):
        client.delete_many({"runtime-a": None})

    assert api.calls == []
