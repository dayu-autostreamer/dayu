import copy
import threading

import pytest
from kubernetes.client.rest import ApiException

from runtime_model import RuntimeEndpoint, RuntimeSlot, RuntimeUnit

from runtime_service_client import (
    RUNTIME_GROUP,
    RUNTIME_PLURAL,
    RUNTIME_VERSION,
    RuntimeServiceClient,
    RuntimeServiceCancelled,
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
    assert type(create_kwargs["_request_timeout"]) is int
    assert type(delete_kwargs["_request_timeout"]) is int
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


def test_wait_cancellation_bounds_list_and_watch_request_windows():
    api = FakeAPI([runtime_obj(ready=False, resource_version="3")])
    watcher = FakeWatch([
        {"type": "MODIFIED", "object": runtime_obj(resource_version="4")},
    ])
    client = RuntimeServiceClient(
        "dayu", api=api, watch_factory=lambda: watcher,
        request_timeout_seconds=30,
    )

    result = client.wait_for_conditions(
        {"runtime-a": 3},
        timeout_seconds=20,
        cancel_event=threading.Event(),
    )

    assert result["runtime-a"]["metadata"]["resourceVersion"] == "4"
    list_kwargs = next(kwargs for operation, kwargs in api.calls if operation == "list")
    assert list_kwargs["_request_timeout"] == 2.0
    assert type(list_kwargs["_request_timeout"]) is int
    watch_kwargs = watcher.stream_calls[0][1]
    assert watch_kwargs["timeout_seconds"] == 1
    assert watch_kwargs["_request_timeout"] == 2
    assert type(watch_kwargs["_request_timeout"]) is int


def test_wait_with_pre_cancelled_token_does_not_contact_kubernetes():
    api = FakeAPI([runtime_obj()])
    client = RuntimeServiceClient("dayu", api=api)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(RuntimeServiceCancelled, match="cancelled"):
        client.wait_for_conditions(
            {"runtime-a": 3},
            cancel_event=cancelled,
        )

    assert api.calls == []


def test_wait_cancellation_inside_watch_stops_stream_and_propagates():
    cancel_event = threading.Event()

    class CancellingWatch(FakeWatch):
        def stream(self, func, **kwargs):
            self.stream_calls.append((func, kwargs))
            cancel_event.set()
            yield {"type": "MODIFIED", "object": runtime_obj()}

    api = FakeAPI([runtime_obj(ready=False, resource_version="3")])
    watcher = CancellingWatch(())
    client = RuntimeServiceClient("dayu", api=api, watch_factory=lambda: watcher)

    with pytest.raises(RuntimeServiceCancelled, match="cancelled"):
        client.wait_for_conditions(
            {"runtime-a": 3},
            cancel_event=cancel_event,
        )

    assert watcher.stopped is True


def test_wait_fails_fast_on_rejected_spec():
    api = FakeAPI([runtime_obj(ready=False, accepted=False)])
    client = RuntimeServiceClient("dayu", api=api)
    with pytest.raises(RuntimeServiceRejected, match="Invalid: bad spec"):
        client.wait_for_conditions({"runtime-a": 3})


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_wait_rejects_non_finite_or_non_positive_deadlines(timeout):
    client = RuntimeServiceClient("dayu", api=FakeAPI())

    with pytest.raises(ValueError, match="finite and positive"):
        client.wait_for_conditions({"runtime-a": 3}, timeout_seconds=timeout)


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


def test_delete_many_accepts_uid_guarded_background_deletes_without_waiting_for_gc():
    first = runtime_obj("runtime-a", resource_version="3")
    second = runtime_obj("runtime-b", resource_version="3")
    first["metadata"]["uid"] = "uid-a"
    second["metadata"]["uid"] = "uid-b"
    api = FakeAPI([first, second])
    deleted_first = runtime_obj("runtime-a", resource_version="4")
    deleted_second = runtime_obj("runtime-b", resource_version="5")
    deleted_first["metadata"]["uid"] = "uid-a"
    deleted_second["metadata"]["uid"] = "uid-b"
    client = RuntimeServiceClient(
        "dayu",
        api=api,
        watch_factory=lambda: pytest.fail("deletion acceptance must not start a watch"),
    )

    assert client.delete_many(
        {"runtime-a": "uid-a", "runtime-b": "uid-b"},
    ) is True

    deletes = [kwargs for operation, kwargs in api.calls if operation == "delete"]
    lists = [kwargs for operation, kwargs in api.calls if operation == "list"]
    assert [item["name"] for item in deletes] == ["runtime-a", "runtime-b"]
    assert [item["body"]["preconditions"]["uid"] for item in deletes] == ["uid-a", "uid-b"]
    assert all(item["body"]["propagationPolicy"] == "Background" for item in deletes)
    assert lists == []


def test_delete_many_submits_uid_guarded_foreground_deletes_for_uninstall():
    item = runtime_obj("runtime-a", resource_version="3")
    item["metadata"]["uid"] = "uid-a"
    api = FakeAPI([item])
    client = RuntimeServiceClient("dayu", api=api)

    assert client.delete_many(
        {"runtime-a": "uid-a"},
        propagation_policy="Foreground",
    ) is True

    delete = next(kwargs for operation, kwargs in api.calls if operation == "delete")
    assert delete["body"]["preconditions"] == {"uid": "uid-a"}
    assert delete["body"]["propagationPolicy"] == "Foreground"


def test_delete_many_rejects_unknown_propagation_policy():
    client = RuntimeServiceClient("dayu", api=FakeAPI([]))

    with pytest.raises(ValueError, match="propagation_policy"):
        client.delete_many(
            {"runtime-a": "uid-a"},
            propagation_policy="Orphan",
        )


def test_delete_many_shares_one_deadline_across_deletes_list_and_watch(monkeypatch):
    class Clock:
        now = 0.0

    clock = Clock()
    monkeypatch.setattr(
        "runtime_service_client.time.monotonic",
        lambda: clock.now,
    )

    first = runtime_obj("runtime-a", resource_version="3")
    second = runtime_obj("runtime-b", resource_version="3")
    first["metadata"]["uid"] = "uid-a"
    second["metadata"]["uid"] = "uid-b"

    class AdvancingAPI(FakeAPI):
        def delete_namespaced_custom_object(self, **kwargs):
            result = super().delete_namespaced_custom_object(**kwargs)
            clock.now += 2.0
            return result

        def list_namespaced_custom_object(self, **kwargs):
            result = super().list_namespaced_custom_object(**kwargs)
            clock.now += 1.0
            return result

    api = AdvancingAPI([first, second])
    client = RuntimeServiceClient(
        "dayu",
        api=api,
        watch_factory=lambda: pytest.fail("deletion acceptance must not start a watch"),
        request_timeout_seconds=30,
    )

    assert client.delete_many(
        {"runtime-a": "uid-a", "runtime-b": "uid-b"},
        timeout_seconds=10,
    ) is True

    deletes = [kwargs for operation, kwargs in api.calls if operation == "delete"]
    assert [kwargs["_request_timeout"] for kwargs in deletes] == [10, 8]
    assert all(operation != "list" for operation, _ in api.calls)


def test_delete_many_stops_batch_when_shared_deadline_expires(monkeypatch):
    class Clock:
        now = 0.0

    clock = Clock()
    monkeypatch.setattr(
        "runtime_service_client.time.monotonic",
        lambda: clock.now,
    )

    class SlowDeleteAPI(FakeAPI):
        def delete_namespaced_custom_object(self, **kwargs):
            result = super().delete_namespaced_custom_object(**kwargs)
            clock.now += 3.0
            return result

    api = SlowDeleteAPI()
    client = RuntimeServiceClient(
        "dayu",
        api=api,
        request_timeout_seconds=30,
    )

    with pytest.raises(RuntimeServiceTimeout, match="timed out deleting"):
        client.delete_many(
            {"runtime-a": "uid-a", "runtime-b": "uid-b"},
            timeout_seconds=5,
        )

    deletes = [kwargs for operation, kwargs in api.calls if operation == "delete"]
    assert [kwargs["_request_timeout"] for kwargs in deletes] == [5, 2]
    assert all(operation != "list" for operation, _ in api.calls)


def test_delete_many_treats_same_name_different_uid_as_replacement_not_target():
    class ConflictAPI(FakeAPI):
        def delete_namespaced_custom_object(self, **kwargs):
            self.calls.append(("delete", kwargs))
            raise ApiException(status=409)

    replacement = runtime_obj("runtime-a", resource_version="7")
    replacement["metadata"]["uid"] = "new-uid"
    api = ConflictAPI([replacement])
    client = RuntimeServiceClient(
        "dayu", api=api, watch_factory=lambda: pytest.fail("watch should not start"),
    )

    assert client.delete_many({"runtime-a": "old-uid"}) is True
    delete_kwargs = next(kwargs for operation, kwargs in api.calls if operation == "delete")
    assert delete_kwargs["body"]["preconditions"] == {"uid": "old-uid"}
    assert any(operation == "get" for operation, _ in api.calls)


def test_delete_many_with_pre_cancelled_token_does_not_delete():
    api = FakeAPI([runtime_obj()])
    client = RuntimeServiceClient("dayu", api=api)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(RuntimeServiceCancelled, match="cancelled"):
        client.delete_many(
            {"runtime-a": "runtime-uid"},
            cancel_event=cancelled,
        )

    assert api.calls == []


def test_delete_many_cancellation_between_requests_keeps_exact_uid_boundary():
    cancel_event = threading.Event()

    class CancellingAPI(FakeAPI):
        def delete_namespaced_custom_object(self, **kwargs):
            result = super().delete_namespaced_custom_object(**kwargs)
            cancel_event.set()
            return result

    api = CancellingAPI()
    client = RuntimeServiceClient("dayu", api=api)

    with pytest.raises(RuntimeServiceCancelled, match="cancelled"):
        client.delete_many(
            {"runtime-a": "uid-a", "runtime-b": "uid-b"},
            cancel_event=cancel_event,
        )

    delete_calls = [kwargs for operation, kwargs in api.calls if operation == "delete"]
    assert len(delete_calls) == 1
    delete_kwargs = delete_calls[0]
    assert delete_kwargs["body"]["preconditions"] == {"uid": "uid-a"}


def test_delete_many_refuses_to_fall_back_to_name_only_deletion():
    api = FakeAPI()
    client = RuntimeServiceClient("dayu", api=api)

    with pytest.raises(ValueError, match="immutable UIDs.*runtime-a"):
        client.delete_many({"runtime-a": None})

    assert api.calls == []


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), 0, -1])
def test_delete_many_rejects_non_finite_or_non_positive_deadlines(timeout):
    client = RuntimeServiceClient("dayu", api=FakeAPI())

    with pytest.raises(ValueError, match="finite and positive"):
        client.delete_many({"runtime-a": "uid-a"}, timeout_seconds=timeout)


def test_waiting_single_delete_treats_same_name_new_uid_as_target_absent():
    replacement = runtime_obj("runtime-a", resource_version="8")
    replacement["metadata"]["uid"] = "new-uid"
    api = FakeAPI([replacement])
    client = RuntimeServiceClient("dayu", api=api)

    assert client.delete(
        "runtime-a",
        uid="old-uid",
        wait=True,
        timeout_seconds=2,
    ) is True
