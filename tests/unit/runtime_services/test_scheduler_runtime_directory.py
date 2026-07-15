import inspect
import json
import threading

import pytest

from core.scheduler.runtime_directory import (
    RuntimeDirectoryConflict,
    RuntimeDirectoryError,
    RuntimeDirectoryStore,
    RedisRuntimeDirectoryStore,
    create_runtime_directory_store,
)
from core.scheduler.task_lease import (
    InMemoryTaskLeaseStore,
    TaskLeaseRetired,
    create_task_lease_store,
)
from core.scheduler.task_lease import RedisTaskLeaseStore
from core.scheduler.scheduler import Scheduler
from core.scheduler.scheduler_server import SchedulerServer
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
        self.zsets = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, nx=False, **_kwargs):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def delete(self, key):
        existed = key in self.values or key in self.sets or key in self.zsets
        self.values.pop(key, None)
        self.sets.pop(key, None)
        self.zsets.pop(key, None)
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
        if script == RedisRuntimeDirectoryStore._COMMIT_WITH_RETIREMENT_SCRIPT:
            assert len(keys) == 5 and len(args) == 3
            (
                active_key,
                proposal_key,
                proposal_index_key,
                lease_key,
                retirement_key,
            ) = keys
            proposal_raw = self.values.get(proposal_key)
            if not proposal_raw:
                self.srem(proposal_index_key, proposal_key)
                return [0, "", ""]
            proposal = json.loads(proposal_raw)
            current = json.loads(self.values[active_key]) if active_key in self.values else {}
            current_revision = int(current.get("revision", 0))
            expected_revision = int(args[0])
            if (
                current_revision != expected_revision
                or int(proposal["base_revision"]) != current_revision
            ):
                return [1, str(current_revision), ""]
            now, requested_deadline = float(args[1]), float(args[2])
            retirement = json.loads(self.values[retirement_key]) \
                if retirement_key in self.values else {
                    "deadline": requested_deadline,
                    "retired": False,
                    "revoked_count": 0,
                }
            retirement["deadline"] = min(
                float(retirement["deadline"]), requested_deadline,
            )
            deadline = retirement["deadline"]
            leases = self.zsets.setdefault(lease_key, {})
            if retirement["retired"]:
                leases.clear()
            elif deadline <= now:
                forced = {
                    member: expiry for member, expiry in leases.items()
                    if expiry >= deadline
                }
                retirement["revoked_count"] += len(forced)
                retirement["retired"] = True
                leases.clear()
            else:
                self.zsets[lease_key] = {
                    member: min(expiry, deadline)
                    for member, expiry in leases.items()
                    if expiry > now
                }
                leases = self.zsets[lease_key]
            self.values[retirement_key] = json.dumps(retirement)
            directory_raw = json.dumps(proposal["directory"], separators=(",", ":"))
            self.values[active_key] = directory_raw
            del self.values[proposal_key]
            self.srem(proposal_index_key, proposal_key)
            status = {
                "revision": expected_revision,
                "count": len(leases),
                **retirement,
            }
            return [2, directory_raw, json.dumps(status)]
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


class FakeTaskLeaseRedis:
    """Semantic Redis fake for the lease scripts' durable state transitions."""

    def __init__(self, active_revision):
        self.active_revision = active_revision
        self.leases = {}
        self.retirements = {}

    @staticmethod
    def _revision(key):
        parts = key.split(":")
        return int(parts[-2] if parts[-1] == "retirement" else parts[-1])

    def _prune(self, revision, now):
        self.leases[revision] = {
            member: expiry
            for member, expiry in self.leases.get(revision, {}).items()
            if expiry > now
        }

    def _status(self, revision, now):
        retirement = self.retirements.get(revision)
        if retirement and not retirement["retired"] and retirement["deadline"] <= now:
            self.leases[revision] = {
                member: expiry
                for member, expiry in self.leases.get(revision, {}).items()
                if expiry >= retirement["deadline"]
            }
            retirement["revoked_count"] += len(self.leases.get(revision, {}))
            self.leases[revision] = {}
            retirement["retired"] = True
        elif not (retirement and retirement["retired"]):
            self._prune(revision, now)
        return {
            "count": len(self.leases.get(revision, {})),
            "deadline": retirement["deadline"] if retirement else False,
            "retired": bool(retirement and retirement["retired"]),
            "revoked_count": retirement["revoked_count"] if retirement else 0,
        }

    def eval(self, script, key_count, *values):
        keys = values[:key_count]
        args = values[key_count:]
        revision = self._revision(keys[-1] if script == RedisTaskLeaseStore._ACQUIRE_SCRIPT else keys[0])
        if script == RedisTaskLeaseStore._ACQUIRE_SCRIPT:
            assert key_count == 3
            assert keys[0].endswith(":active")
            requested, now, expiry, member, _ttl = args
            retirement = self.retirements.get(revision)
            if retirement:
                return [1, str(retirement["deadline"])]
            if self.active_revision != int(requested):
                return [0, str(self.active_revision)]
            self._prune(revision, float(now))
            self.leases.setdefault(revision, {})[str(member)] = float(expiry)
            return [2, str(expiry)]
        if script == RedisTaskLeaseStore._RENEW_SCRIPT:
            assert key_count == 3
            assert keys[2].endswith(":active")
            member, now, expiry, _ttl, requested_revision = args
            if (
                int(requested_revision) != self.active_revision
                and revision not in self.retirements
            ):
                return [-2, str(self.active_revision)]
            status = self._status(revision, float(now))
            if status["retired"]:
                return [-1, str(status["deadline"])]
            if str(member) not in self.leases.get(revision, {}):
                return [0, ""]
            if status["deadline"] is not False:
                expiry = min(float(expiry), status["deadline"])
            self.leases[revision][str(member)] = float(expiry)
            return [1, str(expiry)]
        if script == RedisTaskLeaseStore._RELEASE_SCRIPT:
            member, now = args
            self._prune(revision, float(now))
            if self.leases.get(revision, {}).pop(str(member), None) is not None:
                return 1
            return 2 if revision in self.retirements else 0
        if script == RedisTaskLeaseStore._RETIRE_SCRIPT:
            now, deadline = map(float, args)
            retirement = self.retirements.setdefault(revision, {
                "deadline": deadline,
                "retired": False,
                "revoked_count": 0,
            })
            retirement["deadline"] = min(retirement["deadline"], deadline)
            if retirement["deadline"] <= now and not retirement["retired"]:
                retirement["revoked_count"] += len(self.leases.get(revision, {}))
                self.leases[revision] = {}
                retirement["retired"] = True
            elif not retirement["retired"]:
                self._prune(revision, now)
                self.leases[revision] = {
                    member: min(expiry, retirement["deadline"])
                    for member, expiry in self.leases.get(revision, {}).items()
                }
            return json.dumps(self._status(revision, now))
        if script == RedisTaskLeaseStore._STATUS_SCRIPT:
            return json.dumps(self._status(revision, float(args[0])))
        raise AssertionError("unexpected task lease Lua script")


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
def test_task_leases_are_multi_tenant_and_expire_without_retirement():
    now = [100.0]
    leases = InMemoryTaskLeaseStore(clock=lambda: now[0])

    leases.acquire(3, "task-a", active_revision=3, ttl_seconds=10)
    leases.acquire(3, "task-b", active_revision=3, ttl_seconds=20)
    assert leases.count(3) == 2
    with pytest.raises(RuntimeDirectoryConflict):
        leases.acquire(2, "late-old-task", active_revision=3)

    # A revision switch without an armed retirement fails closed. Once the
    # deadline exists, existing work can renew only inside that bound.
    with pytest.raises(RuntimeDirectoryConflict, match="active revision is 4"):
        leases.renew(3, "task-a", ttl_seconds=30, active_revision=4)
    leases.retire(3, deadline=125)
    leases.renew(3, "task-a", ttl_seconds=30)
    leases.release(3, "task-b")
    assert leases.count(3) == 1
    now[0] = 131.0
    assert leases.count(3) == 0


@pytest.mark.unit
def test_task_lease_retirement_is_bounded_and_idempotent():
    now = [100.0]
    leases = InMemoryTaskLeaseStore(clock=lambda: now[0])
    leases.acquire(3, "task-a", active_revision=3, ttl_seconds=100)
    leases.acquire(3, "task-b", active_revision=3, ttl_seconds=100)
    leases.acquire(3, "naturally-expired", active_revision=3, ttl_seconds=10)

    status = leases.retire(3, deadline=120)
    assert status == {
        "revision": 3,
        "count": 3,
        "deadline": 120.0,
        "retired": False,
        "revoked_count": 0,
    }
    assert leases.renew(3, "task-a", ttl_seconds=100)["expires_at"] == 120.0
    with pytest.raises(TaskLeaseRetired, match="revision 3"):
        leases.acquire(3, "late-task", active_revision=3)

    # Recovery may repeat retirement, but it can only tighten the persisted
    # deadline.  A stale caller cannot reopen the grace period.
    assert leases.retire(3, deadline=140)["deadline"] == 120.0
    assert leases.retire(3, deadline=115)["deadline"] == 115.0

    # Reconciliation may run after the deadline. Only leases that were still
    # valid at 115 are forced; the 10-second lease ended naturally beforehand.
    now[0] = 130.0
    assert leases.status(3) == {
        "revision": 3,
        "count": 0,
        "deadline": 115.0,
        "retired": True,
        "revoked_count": 2,
    }
    with pytest.raises(TaskLeaseRetired, match="deadline 115.0"):
        leases.renew(3, "task-a")
    assert leases.release(3, "task-a")["already_released"] is True


@pytest.mark.unit
def test_immediate_retirement_revokes_only_one_revision():
    now = [100.0]
    leases = InMemoryTaskLeaseStore(clock=lambda: now[0])
    leases.acquire(3, "old-task", active_revision=3, ttl_seconds=30)
    leases.acquire(4, "new-task", active_revision=4, ttl_seconds=30)

    assert leases.retire(3, deadline=now[0]) == {
        "revision": 3,
        "count": 0,
        "deadline": 100.0,
        "retired": True,
        "revoked_count": 1,
    }
    assert leases.count(4) == 1


@pytest.mark.unit
def test_retirement_deadline_is_never_rounded_later():
    leases = InMemoryTaskLeaseStore(clock=lambda: 100.0)
    requested = 120.123456

    assert leases.retire(3, deadline=requested)["deadline"] == 120.123
    assert leases.status(3)["deadline"] <= requested


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
            "socket_connect_timeout": 5.0,
            "socket_timeout": 5.0,
        },
        {
            "host": "redis.dayu.svc.cluster.local.",
            "port": 6379,
            "decode_responses": True,
            "socket_connect_timeout": 5.0,
            "socket_timeout": 5.0,
        },
    ]


@pytest.mark.unit
def test_scheduler_runtime_state_handlers_run_outside_the_event_loop():
    for method_name in (
        "get_runtime_directory",
        "put_runtime_directory",
        "clear_runtime_directory",
        "propose_runtime_directory",
        "commit_runtime_directory",
        "reject_runtime_directory",
        "count_task_leases",
        "retire_task_leases",
        "acquire_task_lease",
        "renew_task_lease",
        "release_task_lease",
    ):
        assert not inspect.iscoroutinefunction(getattr(SchedulerServer, method_name))

    assert inspect.iscoroutinefunction(SchedulerServer.get_resource_lock)


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


@pytest.mark.unit
def test_memory_scheduler_commit_switches_directory_and_arms_retirement_together():
    now = [100.0]
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._runtime_state_lock = threading.RLock()
    scheduler._runtime_clock = lambda: now[0]
    scheduler.runtime_directory = RuntimeDirectoryStore(
        directory(1, [
            route(
                "processor", "edge-1", 1,
                service="detector", suffix="processor",
            ),
        ]),
        clock=lambda: now[0],
    )
    scheduler.task_leases = InMemoryTaskLeaseStore(clock=lambda: now[0])
    scheduler.acquire_task_lease(1, "in-flight", ttl_seconds=100)
    scheduler.propose_runtime_directory(
        directory(2, [
            route(
                "processor", "cloud", 2,
                service="detector", suffix="processor2",
            ),
        ]),
        base_revision=1,
        proposal_id="rollout",
    )

    with pytest.raises(
        RuntimeDirectoryError,
        match="retirement_grace_seconds must be finite and positive",
    ):
        scheduler.commit_runtime_directory(
            "rollout",
            expected_revision=1,
            retirement_grace_seconds=float("nan"),
        )
    assert scheduler.runtime_directory_revision() == 1

    committed = scheduler.commit_runtime_directory(
        "rollout",
        expected_revision=1,
        retirement_grace_seconds=20,
    )

    assert committed["revision"] == 2
    assert committed["retirement"] == {
        "revision": 1,
        "count": 1,
        "deadline": 120.0,
        "retired": False,
        "revoked_count": 0,
    }
    assert scheduler.renew_task_lease(
        1, "in-flight", ttl_seconds=100,
    )["expires_at"] == 120.0
    with pytest.raises(TaskLeaseRetired, match="deadline 120.0"):
        scheduler.acquire_task_lease(1, "late", ttl_seconds=10)


@pytest.mark.unit
def test_redis_commit_atomically_clamps_old_revision_lease_scores():
    redis = FakeDirectoryRedis()
    store = RedisRuntimeDirectoryStore(
        redis,
        install_id="test",
        clock=lambda: 100.0,
    )
    store.replace(
        directory(1, [
            route(
                "processor", "edge-1", 1,
                service="detector", suffix="processor",
            ),
        ]),
        expected_revision=0,
    )
    store.propose(
        directory(2, [
            route(
                "processor", "cloud", 2,
                service="detector", suffix="processor2",
            ),
        ]),
        base_revision=1,
        proposal_id="rollout",
    )
    lease_key = "dayu:runtime-directory:task-leases:test:1"
    retirement_key = f"{lease_key}:retirement"
    redis.zsets[lease_key] = {
        "long-running": 200.0,
        "within-grace": 110.0,
        "expired": 99.0,
    }

    committed = store.commit_with_retirement(
        "rollout",
        expected_revision=1,
        retirement_grace_seconds=20,
        lease_key=lease_key,
        retirement_key=retirement_key,
        now=100.0,
    )

    assert committed["revision"] == 2
    assert committed["retirement"] == {
        "revision": 1,
        "count": 2,
        "deadline": 120.0,
        "retired": False,
        "revoked_count": 0,
    }
    assert redis.zsets[lease_key] == {
        "long-running": 120.0,
        "within-grace": 110.0,
    }
    assert json.loads(redis.values[retirement_key]) == {
        "deadline": 120.0,
        "retired": False,
        "revoked_count": 0,
    }


@pytest.mark.unit
def test_redis_task_lease_retirement_survives_restart_and_checks_active_atomically():
    now = [100.0]
    redis = FakeTaskLeaseRedis(active_revision=3)
    first = RedisTaskLeaseStore(redis, install_id="test", clock=lambda: now[0])

    assert first.acquire(3, "task-a", active_revision=3, ttl_seconds=100)[
        "expires_at"
    ] == 200.0
    assert first.acquire(3, "naturally-expired", active_revision=3, ttl_seconds=10)[
        "valid_for_seconds"
    ] == 10.0

    # The Python caller still believes revision 3 is active, but the Redis Lua
    # transaction observes the directory CAS performed by another replica.
    redis.active_revision = 4
    with pytest.raises(RuntimeDirectoryConflict, match="active revision is 4"):
        first.acquire(3, "late-task", active_revision=3)
    with pytest.raises(RuntimeDirectoryConflict, match="active revision is 4"):
        first.renew(3, "task-a", ttl_seconds=100, active_revision=3)

    assert first.retire(3, deadline=120)["deadline"] == 120.0
    with pytest.raises(TaskLeaseRetired, match="deadline 120.0"):
        first.acquire(3, "late-retired-task", active_revision=4)
    restarted = RedisTaskLeaseStore(redis, install_id="test", clock=lambda: now[0])
    assert restarted.retire(3, deadline=140)["deadline"] == 120.0
    assert restarted.renew(3, "task-a", ttl_seconds=100)["expires_at"] == 120.0

    now[0] = 130.0
    assert restarted.status(3) == {
        "revision": 3,
        "count": 0,
        "deadline": 120.0,
        "retired": True,
        "revoked_count": 1,
    }
    with pytest.raises(TaskLeaseRetired, match="revision 3"):
        restarted.renew(3, "task-a")
    assert restarted.release(3, "task-a")["already_released"] is True


@pytest.mark.unit
def test_redis_scheduler_lease_path_does_not_pre_read_active_directory():
    redis = FakeTaskLeaseRedis(active_revision=3)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._runtime_state_lock = threading.RLock()
    scheduler.task_leases = RedisTaskLeaseStore(
        redis,
        install_id="test",
        clock=lambda: 100.0,
    )

    def unexpected_directory_read():
        raise AssertionError("Redis lease admission must be one atomic script")

    scheduler.runtime_directory_revision = unexpected_directory_read

    acquired = scheduler.acquire_task_lease(3, "task-a", ttl_seconds=100)
    renewed = scheduler.renew_task_lease(3, "task-a", ttl_seconds=100)

    assert acquired["expires_at"] == 200.0
    assert renewed["expires_at"] == 200.0
