import json

import pytest

from core.scheduler.runtime_directory import (
    RuntimeDirectoryConflict,
    RuntimeDirectoryError,
    RuntimeDirectoryStore,
    RedisRuntimeDirectoryStore,
    create_runtime_directory_store,
)
from core.scheduler.task_lease import InMemoryTaskLeaseStore, create_task_lease_store
from core.scheduler.task_lease import RedisTaskLeaseStore
from core.lib.runtime import RuntimeEndpoint


def route(component, node, runtime_revision, service="", suffix="x"):
    value = {
        "component": component,
        "target_node": node,
        "position": "cloud" if node == "cloud" else "edge",
        "runtime_id": f"runtime-{suffix}-r{runtime_revision}",
        "runtime_revision": runtime_revision,
        "spec_hash": "a" * 64,
    }
    if service:
        value["logical_service"] = service
    if component in ("processor", "controller", "distributor"):
        value.update({
            "dns_name": f"runtime-{suffix}.dayu.svc.cluster.local",
            "port": 9000,
            "runtime_service_uid": f"rs-{suffix}",
            "service_uid": f"svc-{suffix}",
            "pod_uid": f"pod-{suffix}",
        })
    return value


def directory(revision, routes):
    return {"install_id": "test", "directory_revision": revision, "routes": routes}


class FakeDirectoryRedis:
    """Small semantic fake for exercising restart persistence and CAS results."""

    def __init__(self):
        self.values = {}
        self.sets = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, nx=False, **_kwargs):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def delete(self, key):
        existed = key in self.values or key in self.sets
        self.values.pop(key, None)
        self.sets.pop(key, None)
        return int(existed)

    def srem(self, key, value):
        values = self.sets.get(key, set())
        removed = int(value in values)
        values.discard(value)
        if not values:
            self.sets.pop(key, None)
        return removed

    def scard(self, key):
        return len(self.sets.get(key, set()))

    def eval(self, script, key_count, *values):
        keys = values[:key_count]
        args = values[key_count:]
        if script == RedisRuntimeDirectoryStore._REPLACE_SCRIPT:
            assert len(keys) == 1 and len(args) == 4
            active_key = keys[0]
            current = json.loads(self.values[active_key]) if active_key in self.values else {}
            current_revision = int(current.get("revision", 0))
            expected, candidate_revision = int(args[0]), int(args[1])
            if current_revision != expected:
                return [0, current_revision]
            if candidate_revision <= current_revision:
                return [1, current_revision]
            if current.get("install_id") and current["install_id"] != args[2]:
                return [2, current_revision]
            self.values[active_key] = args[3]
            return [3, candidate_revision]
        if script == RedisRuntimeDirectoryStore._PROPOSE_SCRIPT:
            assert len(keys) == 3 and len(args) == 6
            active_key, proposal_key, proposal_index_key = keys
            current = json.loads(self.values[active_key]) if active_key in self.values else {}
            current_revision = int(current.get("revision", 0))
            base_revision, candidate_revision = int(args[0]), int(args[1])
            if current_revision != base_revision:
                return [0, str(current_revision)]
            if candidate_revision <= current_revision:
                return [1, str(current_revision)]
            if current.get("install_id") and current["install_id"] != args[2]:
                return [2, str(current_revision)]
            existing = self.values.get(proposal_key)
            if existing:
                value = json.loads(existing)
                if int(value["base_revision"]) == base_revision and value["directory"]["hash"] == args[3]:
                    self.sets.setdefault(proposal_index_key, set()).add(proposal_key)
                    return [4, existing]
                return [3, existing]
            self.values[proposal_key] = args[4]
            self.sets.setdefault(proposal_index_key, set()).add(proposal_key)
            return [4, args[4]]
        if script == RedisRuntimeDirectoryStore._COMMIT_SCRIPT:
            assert len(keys) == 3 and len(args) == 1
            active_key, proposal_key, proposal_index_key = keys
            proposal_raw = self.values.get(proposal_key)
            if not proposal_raw:
                self.srem(proposal_index_key, proposal_key)
                return [0, ""]
            proposal = json.loads(proposal_raw)
            current = json.loads(self.values[active_key]) if active_key in self.values else {}
            current_revision = int(current.get("revision", 0))
            if current_revision != int(args[0]) or int(proposal["base_revision"]) != current_revision:
                return [1, str(current_revision)]
            directory_raw = json.dumps(proposal["directory"], separators=(",", ":"))
            self.values[active_key] = directory_raw
            del self.values[proposal_key]
            self.srem(proposal_index_key, proposal_key)
            return [2, directory_raw]
        if script == RedisRuntimeDirectoryStore._CLEAR_SCRIPT:
            assert len(keys) == 2 and len(args) == 1
            active_key, proposal_index_key = keys
            current_raw = self.values.get(active_key)
            previous_revision = 0
            if current_raw:
                current = json.loads(current_raw)
                if current.get("install_id") and current["install_id"] != args[0]:
                    return [0, int(current.get("revision", 0))]
                previous_revision = int(current.get("revision", 0))
            for proposal_key in self.sets.pop(proposal_index_key, set()):
                self.values.pop(proposal_key, None)
            self.values.pop(active_key, None)
            return [1, previous_revision]
        if script == RedisRuntimeDirectoryStore._REJECT_SCRIPT:
            assert len(keys) == 2 and len(args) == 0
            proposal_key, proposal_index_key = keys
            proposal_raw = self.values.pop(proposal_key, None)
            self.srem(proposal_index_key, proposal_key)
            return [1, proposal_raw] if proposal_raw else [0, ""]
        raise AssertionError("unexpected Lua script")


@pytest.mark.unit
def test_directory_cas_accepts_independent_runtime_revision_and_emits_canonical_snapshot():
    store = RuntimeDirectoryStore()
    snapshot = store.replace(
        directory(1, [route("processor", "edge-1", 7, service="detector")]),
        expected_revision=0,
    )

    assert snapshot["revision"] == snapshot["directory_revision"] == 1
    assert snapshot["routes"][0]["runtime_revision"] == 7
    assert snapshot["deployment"] == {"detector": ["edge-1"]}
    assert snapshot["nodes"] == ["edge-1"]
    assert len(snapshot["hash"]) == 64
    assert "slot" not in snapshot["routes"][0]
    assert "endpoint" not in snapshot["routes"][0]

    with pytest.raises(RuntimeDirectoryConflict):
        store.replace(directory(2, []), expected_revision=0)


@pytest.mark.unit
def test_directory_proposal_commit_and_strict_compact_routes():
    store = RuntimeDirectoryStore(directory(1, [
        route("processor", "edge-1", 3, service="detector", suffix="processor"),
        route("controller", "edge-1", 4, suffix="controller"),
        route("distributor", "cloud", 4, suffix="distributor"),
    ]))
    candidate = directory(2, [
        route("processor", "edge-1", 5, service="detector", suffix="processor2"),
        route("controller", "edge-1", 5, suffix="controller2"),
        route("distributor", "cloud", 5, suffix="distributor2"),
    ])
    proposal = store.propose(candidate, base_revision=1, proposal_id="p1")
    assert proposal["directory"]["revision"] == 2
    assert store.commit("p1", expected_revision=1)["revision"] == 2

    compact = store.compact_routes_for_plan({
        "dag": {"detector": {"service": {"execute_device": "edge-1"}}}
    }, source_device="edge-1", cloud_node="cloud")
    assert {(item["component"], item["target_node"]) for item in compact} == {
        ("processor", "edge-1"), ("controller", "edge-1"), ("distributor", "cloud")
    }

    with pytest.raises(RuntimeDirectoryError, match="found 0"):
        store.compact_routes_for_plan({
            "dag": {"detector": {"service": {"execute_device": "edge-2"}}}
        })


@pytest.mark.unit
def test_task_leases_are_multi_tenant_and_old_revision_can_drain():
    now = [100.0]
    leases = InMemoryTaskLeaseStore(clock=lambda: now[0])

    leases.acquire(3, "task-a", active_revision=3, ttl_seconds=10)
    leases.acquire(3, "task-b", active_revision=3, ttl_seconds=20)
    assert leases.count(3) == 2
    with pytest.raises(RuntimeDirectoryConflict):
        leases.acquire(2, "late-old-task", active_revision=3)

    # Existing revision-3 work can renew after revision 4 becomes active.
    leases.renew(3, "task-a", ttl_seconds=30)
    leases.release(3, "task-b")
    assert leases.count(3) == 1
    now[0] = 131.0
    assert leases.count(3) == 0


@pytest.mark.unit
def test_memory_directory_clear_is_install_scoped_and_idempotent():
    store = RuntimeDirectoryStore(directory(1, [
        route("controller", "edge-1", 1, suffix="controller"),
    ]))

    with pytest.raises(RuntimeDirectoryConflict, match="install_id"):
        store.clear("another-install")
    assert store.clear("test")["previous_revision"] == 1
    assert store.clear("test")["previous_revision"] == 0


def test_production_directory_factory_requires_durable_redis_endpoint():
    class MissingRedisContext:
        @staticmethod
        def resolve_static_endpoint(component, required=False):
            assert component == "redis"
            assert required is False
            return None

    with pytest.raises(RuntimeDirectoryError, match="requires a Redis endpoint"):
        create_runtime_directory_store(MissingRedisContext())
    with pytest.raises(RuntimeDirectoryError, match="durable task leases"):
        create_task_lease_store(MissingRedisContext())


@pytest.mark.unit
def test_redis_stores_use_absolute_cluster_dns_at_connection_boundary(monkeypatch):
    import redis

    calls = []

    class ConnectionRedis(FakeDirectoryRedis):
        pass

    def connect(**kwargs):
        calls.append(kwargs)
        return ConnectionRedis()

    monkeypatch.setattr(redis, "Redis", connect)
    endpoint = RuntimeEndpoint(
        component="redis",
        fqdn="redis.dayu.svc.cluster.local",
        port=6379,
    )

    RedisRuntimeDirectoryStore.from_endpoint(endpoint, install_id="install-a")
    RedisTaskLeaseStore.from_endpoint(endpoint, install_id="install-a")

    assert calls == [
        {
            "host": "redis.dayu.svc.cluster.local.",
            "port": 6379,
            "decode_responses": True,
        },
        {
            "host": "redis.dayu.svc.cluster.local.",
            "port": 6379,
            "decode_responses": True,
        },
    ]


@pytest.mark.unit
def test_redis_directory_survives_scheduler_restart_and_keeps_proposal_cas():
    redis = FakeDirectoryRedis()
    first = RedisRuntimeDirectoryStore(redis, install_id="test", clock=lambda: 100.0)
    active = directory(1, [
        route("processor", "edge-1", 3, service="detector", suffix="processor"),
        route("controller", "edge-1", 3, suffix="controller"),
    ])
    first.replace(active, expected_revision=0)
    candidate = directory(2, [
        route("processor", "edge-1", 4, service="detector", suffix="processor2"),
        route("controller", "edge-1", 3, suffix="controller"),
    ])
    proposal = first.propose(candidate, base_revision=1, proposal_id="rollout-1")

    # A new Scheduler process reconstructs both the active snapshot and the
    # pending proposal from Redis without asking Backend or Kubernetes.
    restarted = RedisRuntimeDirectoryStore(redis, install_id="test", clock=lambda: 101.0)
    assert restarted.snapshot()["hash"] == first.snapshot()["hash"]
    assert restarted.propose(candidate, base_revision=1, proposal_id="rollout-1") == proposal
    assert restarted.commit("rollout-1", expected_revision=1)["revision"] == 2

    second_restart = RedisRuntimeDirectoryStore(redis, install_id="test")
    assert second_restart.snapshot()["revision"] == 2
    with pytest.raises(RuntimeDirectoryConflict):
        second_restart.replace(directory(3, []), expected_revision=1)
    candidate_three = directory(3, [
        route("processor", "edge-1", 5, service="detector", suffix="processor3"),
        route("controller", "edge-1", 3, suffix="controller"),
    ])
    second_restart.propose(
        candidate_three,
        base_revision=2,
        proposal_id="rejected-before-uninstall",
    )
    assert second_restart.reject("rejected-before-uninstall", "superseded") == {
        "proposal_id": "rejected-before-uninstall",
        "rejected": True,
        "reason": "superseded",
        "revision": 3,
    }
    assert second_restart._proposal_index_key not in redis.sets
    pending = second_restart.propose(
        candidate_three,
        base_revision=2,
        proposal_id="pending-at-uninstall",
    )
    assert pending["proposal_id"] == "pending-at-uninstall"
    assert second_restart.clear("test") == {
        "cleared": True,
        "install_id": "test",
        "previous_revision": 2,
    }
    assert RedisRuntimeDirectoryStore(redis, install_id="test").snapshot()["revision"] == 0
    assert second_restart._proposal_index_key not in redis.sets
    assert second_restart._proposal_key("pending-at-uninstall") not in redis.values
    with pytest.raises(RuntimeDirectoryError, match="does not exist"):
        second_restart.commit("pending-at-uninstall", expected_revision=0)
    assert second_restart.clear("test")["previous_revision"] == 0
    with pytest.raises(RuntimeDirectoryConflict, match="install_id"):
        second_restart.clear("another-install")
