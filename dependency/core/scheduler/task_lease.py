"""Task drain leases keyed by immutable runtime-directory revision.

A task acquires a lease for the directory revision copied into that task.  A
new directory can become active while old tasks keep renewing their existing
leases; the control plane retires the old RuntimeServices only after the old
revision's lease count reaches zero.  This is intentionally separate from
directory proposal/CAS state -- leases are many-to-one, not a control lock.
"""

import math
import threading
import time
from abc import ABC, abstractmethod

from .runtime_directory import (
    RuntimeDirectoryConflict,
    RuntimeDirectoryError,
    RuntimeDirectoryNotFound,
)


def _revision(value):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise RuntimeDirectoryError("task lease revision must be an integer")
    if value < 1:
        raise RuntimeDirectoryError("task lease revision must be positive")
    return value


def _root_uuid(value):
    value = str(value or "").strip()
    if not value:
        raise RuntimeDirectoryError("task lease root_uuid is required")
    if len(value) > 256:
        raise RuntimeDirectoryError("task lease root_uuid must not exceed 256 characters")
    return value


def _ttl(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise RuntimeDirectoryError("task lease ttl_seconds must be numeric")
    if value <= 0:
        raise RuntimeDirectoryError("task lease ttl_seconds must be positive")
    return value


class TaskLeaseStore(ABC):
    """Persistence interface for multi-tenant task drain leases."""

    @abstractmethod
    def acquire(self, revision, root_uuid, active_revision, ttl_seconds=60.0):
        raise NotImplementedError

    @abstractmethod
    def renew(self, revision, root_uuid, ttl_seconds=60.0):
        raise NotImplementedError

    @abstractmethod
    def release(self, revision, root_uuid):
        raise NotImplementedError

    @abstractmethod
    def count(self, revision):
        raise NotImplementedError


class InMemoryTaskLeaseStore(TaskLeaseStore):
    """Thread-safe implementation for tests and bootstrap-less development."""

    def __init__(self, clock=None):
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._leases = {}

    def _prune_locked(self):
        now = self._clock()
        self._leases = {
            key: expires_at for key, expires_at in self._leases.items()
            if expires_at > now
        }

    @staticmethod
    def _payload(revision, root_uuid, expires_at):
        return {
            "revision": revision,
            "root_uuid": root_uuid,
            "expires_at": expires_at,
        }

    def acquire(self, revision, root_uuid, active_revision, ttl_seconds=60.0):
        revision = _revision(revision)
        active_revision = _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        if revision != active_revision:
            raise RuntimeDirectoryConflict(
                f"task lease revision {revision} is not active (active revision is {active_revision})"
            )
        with self._lock:
            self._prune_locked()
            expires_at = self._clock() + ttl_seconds
            self._leases[(revision, root_uuid)] = expires_at
            return self._payload(revision, root_uuid, expires_at)

    def renew(self, revision, root_uuid, ttl_seconds=60.0):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        with self._lock:
            self._prune_locked()
            key = (revision, root_uuid)
            if key not in self._leases:
                raise RuntimeDirectoryNotFound("task lease does not exist or expired")
            expires_at = self._clock() + ttl_seconds
            self._leases[key] = expires_at
            return self._payload(revision, root_uuid, expires_at)

    def release(self, revision, root_uuid):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        with self._lock:
            self._prune_locked()
            key = (revision, root_uuid)
            if key not in self._leases:
                raise RuntimeDirectoryNotFound("task lease does not exist or expired")
            del self._leases[key]
            result = self._payload(revision, root_uuid, self._clock())
            result["released"] = True
            return result

    def count(self, revision):
        revision = _revision(revision)
        with self._lock:
            self._prune_locked()
            return sum(1 for lease_revision, _ in self._leases if lease_revision == revision)


class RedisTaskLeaseStore(TaskLeaseStore):
    """Redis ZSET implementation used by scheduler replicas in production."""

    _RENEW_SCRIPT = """
local key = KEYS[1]
local member = ARGV[1]
local now = tonumber(ARGV[2])
local expires = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZSCORE', key, member) == false then
  return 0
end
redis.call('ZADD', key, expires, member)
redis.call('EXPIRE', key, tonumber(ARGV[4]))
return 1
"""

    def __init__(self, redis_client, install_id="", clock=None, key_prefix="dayu:runtime-directory:task-leases"):
        self.redis = redis_client
        self.install_id = str(install_id or "default")
        self.clock = clock or time.time
        self.key_prefix = str(key_prefix)

    @classmethod
    def from_endpoint(cls, endpoint, install_id="", clock=None):
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError("redis package is required for production task leases") from exc
        return cls(
            redis.Redis(
                host=endpoint.connection_host,
                port=endpoint.port or 6379,
                decode_responses=True,
            ),
            install_id=install_id,
            clock=clock,
        )

    def _key(self, revision):
        return f"{self.key_prefix}:{self.install_id}:{revision}"

    @staticmethod
    def _payload(revision, root_uuid, expires_at):
        return {
            "revision": revision,
            "root_uuid": root_uuid,
            "expires_at": expires_at,
        }

    def acquire(self, revision, root_uuid, active_revision, ttl_seconds=60.0):
        revision = _revision(revision)
        active_revision = _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        if revision != active_revision:
            raise RuntimeDirectoryConflict(
                f"task lease revision {revision} is not active (active revision is {active_revision})"
            )
        now = self.clock()
        expires_at = now + ttl_seconds
        key = self._key(revision)
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, "-inf", now)
            pipe.zadd(key, {root_uuid: expires_at})
            pipe.expire(key, max(1, int(math.ceil(ttl_seconds)) + 60))
            pipe.execute()
        return self._payload(revision, root_uuid, expires_at)

    def renew(self, revision, root_uuid, ttl_seconds=60.0):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        now = self.clock()
        expires_at = now + ttl_seconds
        renewed = self.redis.eval(
            self._RENEW_SCRIPT,
            1,
            self._key(revision),
            root_uuid,
            now,
            expires_at,
            max(1, int(math.ceil(ttl_seconds)) + 60),
        )
        if int(renewed or 0) != 1:
            raise RuntimeDirectoryNotFound("task lease does not exist or expired")
        return self._payload(revision, root_uuid, expires_at)

    def release(self, revision, root_uuid):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        removed = self.redis.zrem(self._key(revision), root_uuid)
        if int(removed or 0) != 1:
            raise RuntimeDirectoryNotFound("task lease does not exist or expired")
        result = self._payload(revision, root_uuid, self.clock())
        result["released"] = True
        return result

    def count(self, revision):
        revision = _revision(revision)
        key = self._key(revision)
        now = self.clock()
        with self.redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, "-inf", now)
            pipe.zcard(key)
            _, count = pipe.execute()
        return int(count)


def create_task_lease_store(runtime_context):
    """Create the production durable store; tests inject memory explicitly."""

    endpoint = runtime_context.resolve_static_endpoint("redis", required=False)
    if endpoint is None:
        raise RuntimeDirectoryError(
            "managed Scheduler requires a Redis endpoint for durable task leases"
        )
    return RedisTaskLeaseStore.from_endpoint(
        endpoint,
        install_id=runtime_context.install_id,
    )
