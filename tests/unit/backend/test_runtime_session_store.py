import copy

import pytest
from kubernetes.client.rest import ApiException

from runtime_model import (
    RuntimeCleanupResource,
    RuntimeSession,
    RuntimeUninstallProgress,
)
from runtime_session_store import (
    RuntimeSessionConflict,
    RuntimeSessionCorrupt,
    RuntimeSessionStore,
    SESSION_DATA_KEY,
)


def make_session(phase="new"):
    return RuntimeSession(
        install_id="install-1",
        operation_id="operation-1",
        phase=phase,
        source_deploy=[{"source": {"id": 0}}],
        updated_at="2026-07-12T00:00:00Z",
    )


class FakeCoreAPI:
    def __init__(self):
        self.value = None
        self.next_rv = 1
        self.calls = []

    def _stored(self, body):
        value = copy.deepcopy(body)
        value["metadata"]["resourceVersion"] = str(self.next_rv)
        value["metadata"]["uid"] = "configmap-uid"
        self.next_rv += 1
        self.value = value
        return copy.deepcopy(value)

    def read_namespaced_config_map(self, **kwargs):
        self.calls.append(("read", kwargs))
        if self.value is None:
            raise ApiException(status=404)
        return copy.deepcopy(self.value)

    def create_namespaced_config_map(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.value is not None:
            raise ApiException(status=409)
        return self._stored(kwargs["body"])

    def replace_namespaced_config_map(self, **kwargs):
        self.calls.append(("replace", kwargs))
        if self.value is None:
            raise ApiException(status=404)
        expected = str(kwargs["body"]["metadata"].get("resourceVersion") or "")
        actual = str((self.value or {}).get("metadata", {}).get("resourceVersion") or "")
        if expected != actual:
            raise ApiException(status=409)
        return self._stored(kwargs["body"])

    def delete_namespaced_config_map(self, **kwargs):
        self.calls.append(("delete", kwargs))
        if self.value is None:
            raise ApiException(status=404)
        if kwargs["body"]["preconditions"]["uid"] != self.value["metadata"]["uid"]:
            raise ApiException(status=409)
        if (
            kwargs["body"]["preconditions"]["resourceVersion"]
            != self.value["metadata"]["resourceVersion"]
        ):
            raise ApiException(status=409)
        self.value = None
        return {}


def test_store_requires_backend_shared_core_api():
    with pytest.raises(ValueError, match="shared ClusterClient"):
        RuntimeSessionStore("dayu", api=None)


def test_store_create_load_replace_and_uid_guarded_delete():
    api = FakeCoreAPI()
    store = RuntimeSessionStore("dayu", api=api)
    assert store.load() is None

    created = store.compare_and_swap(make_session(), expected_resource_version=None)
    assert created.resource_version == "1"
    assert created.session == make_session()
    assert "manifest" not in api.value["data"][SESSION_DATA_KEY]

    updated_session = make_session(phase="activating")
    updated = store.compare_and_swap(updated_session, expected_resource_version=created.resource_version)
    assert updated.resource_version == "2"
    assert store.load().session.phase == "activating"

    assert store.delete(expected_resource_version="2") is True
    assert api.value is None
    delete_call = next(kwargs for operation, kwargs in api.calls if operation == "delete")
    assert delete_call["body"]["preconditions"] == {
        "uid": "configmap-uid",
        "resourceVersion": "2",
    }
    assert store.delete() is True
    assert api.calls
    assert all(
        type(kwargs["_request_timeout"]) is int
        for _, kwargs in api.calls
    )


def test_store_reports_create_and_replace_conflicts():
    api = FakeCoreAPI()
    store = RuntimeSessionStore("dayu", api=api)
    store.compare_and_swap(make_session(), None)
    with pytest.raises(RuntimeSessionConflict):
        store.compare_and_swap(make_session(), None)
    with pytest.raises(RuntimeSessionConflict):
        store.compare_and_swap(make_session("ready"), "stale")
    with pytest.raises(RuntimeSessionConflict):
        store.delete(expected_resource_version="stale")


def test_store_round_trips_durable_uninstall_progress():
    api = FakeCoreAPI()
    store = RuntimeSessionStore("dayu", api=api)
    session = RuntimeSession(
        install_id="install-1",
        operation_id="operation-1",
        phase="finalizing-uninstall",
        uninstall=RuntimeUninstallProgress(
            started_at="2026-07-16T00:00:00+00:00",
            last_progress_at="2026-07-16T00:01:00+00:00",
            deletion_submitted=True,
            remaining=(RuntimeCleanupResource(
                kind="Pod",
                name="processor-pod",
                uid="pod-uid",
            ),),
        ),
    )

    store.compare_and_swap(session, None)

    assert store.load().session == session


def test_store_treats_remote_delete_during_replace_as_a_cas_conflict():
    api = FakeCoreAPI()
    store = RuntimeSessionStore("dayu", api=api)
    created = store.compare_and_swap(make_session(), None)
    api.value = None

    with pytest.raises(RuntimeSessionConflict):
        store.compare_and_swap(
            make_session("activating"),
            created.resource_version,
        )


def test_store_detects_corrupted_session_hash():
    api = FakeCoreAPI()
    store = RuntimeSessionStore("dayu", api=api)
    store.compare_and_swap(make_session(), None)
    api.value["data"][SESSION_DATA_KEY] = "{}"
    with pytest.raises(RuntimeSessionCorrupt, match="hash mismatch"):
        store.load()
