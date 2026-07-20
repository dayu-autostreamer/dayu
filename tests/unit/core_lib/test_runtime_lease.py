import json

import pytest

from core.lib.network import NetworkAPIMethod
from core.lib.runtime import (
    RuntimeContext,
    RuntimeLeaseClient,
    RuntimeLeaseIdentityError,
    RuntimeLeaseRetired,
    RuntimeLeaseUnavailable,
)


class LeaseTask:
    def __init__(self, revision=7, root_uuid="root-7"):
        self.revision = revision
        self.root_uuid = root_uuid

    def get_runtime_directory_revision(self):
        return self.revision

    def get_root_uuid(self):
        return self.root_uuid


def runtime_context(ttl=45):
    return RuntimeContext({
        "lease_ttl_seconds": ttl,
        "endpoints": {
            "scheduler": {
                "component": "scheduler",
                "fqdn": "scheduler.dayu.svc.cluster.local",
                "port": 9000,
            },
        },
    })


@pytest.mark.unit
def test_runtime_lease_client_uses_exact_task_key_and_scheduler_endpoint():
    calls = []

    def requester(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["data"]["data"])
        response = {
            "revision": payload["revision"],
            "root_uuid": payload["root_uuid"],
        }
        if kwargs["method"] == NetworkAPIMethod.SCHEDULER_RELEASE_TASK_LEASE:
            response["released"] = True
        else:
            response["expires_at"] = 123.0
            response["valid_for_seconds"] = 45.0
        return response

    client = RuntimeLeaseClient(runtime_context(), requester=requester)
    task = LeaseTask()

    assert client.acquire(task)["expires_at"] == 123.0
    assert client.renew(task)["valid_for_seconds"] == 45.0
    assert client.release(task)["released"] is True

    assert [call["method"] for call in calls] == [
        NetworkAPIMethod.SCHEDULER_ACQUIRE_TASK_LEASE,
        NetworkAPIMethod.SCHEDULER_RENEW_TASK_LEASE,
        NetworkAPIMethod.SCHEDULER_RELEASE_TASK_LEASE,
    ]
    assert all(
        call["url"]
        == "http://scheduler.dayu.svc.cluster.local.:9000/runtime-directory/task-leases"
        for call in calls
    )
    acquire_payload = json.loads(calls[0]["data"]["data"])
    assert acquire_payload == {
        "revision": 7,
        "root_uuid": "root-7",
        "ttl_seconds": 45.0,
    }
    release_payload = json.loads(calls[-1]["data"]["data"])
    assert release_payload == {"revision": 7, "root_uuid": "root-7"}


@pytest.mark.unit
def test_runtime_lease_client_attaches_commitment_only_to_admission():
    calls = []

    def requester(**kwargs):
        calls.append(json.loads(kwargs["data"]["data"]))
        payload = calls[-1]
        response = {
            "revision": payload["revision"],
            "root_uuid": payload["root_uuid"],
        }
        if kwargs["method"] == NetworkAPIMethod.SCHEDULER_RELEASE_TASK_LEASE:
            response["released"] = True
        else:
            response.update({"expires_at": 123.0, "valid_for_seconds": 45.0})
        return response

    class CommittedLeaseTask(LeaseTask):
        @staticmethod
        def get_schedule_commitment():
            return {
                "root_uuid": "root-7",
                "runtime_directory_revision": 7,
                "decision_id": "decision-7",
            }

    client = RuntimeLeaseClient(runtime_context(), requester=requester)
    task = CommittedLeaseTask()
    client.acquire(task)
    client.renew(task)
    client.release(task)

    assert calls[0]["commitment"]["decision_id"] == "decision-7"
    assert "commitment" not in calls[1]
    assert "commitment" not in calls[2]


@pytest.mark.unit
def test_runtime_lease_client_fails_closed_on_invalid_task_or_response():
    client = RuntimeLeaseClient(runtime_context(), requester=lambda **kwargs: None)

    with pytest.raises(RuntimeLeaseIdentityError, match="positive"):
        client.acquire(LeaseTask(revision=0))
    with pytest.raises(RuntimeLeaseIdentityError, match="root_uuid"):
        client.acquire(LeaseTask(root_uuid=""))
    with pytest.raises(RuntimeLeaseUnavailable, match="not confirmed"):
        client.acquire(LeaseTask())

    mismatched = RuntimeLeaseClient(
        runtime_context(),
        requester=lambda **kwargs: {
            "revision": 8,
            "root_uuid": "root-7",
            "expires_at": 123.0,
        },
    )
    with pytest.raises(RuntimeLeaseUnavailable, match="identity mismatch"):
        mismatched.renew(LeaseTask())

    missing_lifetime = RuntimeLeaseClient(
        runtime_context(),
        requester=lambda **kwargs: {
            "revision": 7,
            "root_uuid": "root-7",
            "expires_at": 123.0,
        },
    )
    with pytest.raises(RuntimeLeaseUnavailable, match="relative lifetime"):
        missing_lifetime.renew(LeaseTask())

    excessive_lifetime = RuntimeLeaseClient(
        runtime_context(),
        requester=lambda **kwargs: {
            "revision": 7,
            "root_uuid": "root-7",
            "expires_at": 123.0,
            "valid_for_seconds": 46.0,
        },
    )
    with pytest.raises(RuntimeLeaseUnavailable, match="relative lifetime"):
        excessive_lifetime.renew(LeaseTask())


@pytest.mark.unit
def test_runtime_context_reads_and_validates_lease_ttl(monkeypatch):
    monkeypatch.setenv("DAYU_RUNTIME_LEASE_TTL_SECONDS", "90")
    assert RuntimeContext({}).lease_ttl_seconds == 90.0
    assert RuntimeContext({"leaseTTLSeconds": 15}).lease_ttl_seconds == 15.0
    with pytest.raises(ValueError, match="finite positive"):
        RuntimeContext({"lease_ttl_seconds": 0})


@pytest.mark.unit
def test_runtime_lease_client_distinguishes_retired_revision():
    client = RuntimeLeaseClient(
        runtime_context(),
        requester=lambda **kwargs: {
            "revision": 7,
            "root_uuid": "root-7",
            "retired": True,
            "deadline": 123.0,
        },
    )

    for operation in (client.acquire, client.renew):
        with pytest.raises(RuntimeLeaseRetired, match="revision 7") as raised:
            operation(LeaseTask())
        assert raised.value.deadline == 123.0
