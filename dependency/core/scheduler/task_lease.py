"""Task leases and bounded retirement for RuntimeDirectory revisions.

Only the active directory revision can acquire new leases.  Once retirement
starts, existing leases may finish until one persisted deadline, but renewals
can never extend beyond it.  Reaching the deadline atomically fences the
revision and revokes its remaining leases, so a stuck task cannot block a
runtime rollout forever.
"""

import json
import math
import threading
import time
from abc import ABC, abstractmethod

from .runtime_directory import (
    REDIS_SOCKET_TIMEOUT_SECONDS,
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
    if not math.isfinite(value) or value <= 0:
        raise RuntimeDirectoryError("task lease ttl_seconds must be finite and positive")
    return value


def _deadline(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise RuntimeDirectoryError("task lease retirement deadline must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise RuntimeDirectoryError(
            "task lease retirement deadline must be a finite positive timestamp"
        )
    # Redis Lua JSON uses finite decimal precision. Flooring to milliseconds
    # guarantees its persisted value can never round past the caller's safety
    # deadline while retaining more precision than the reconcile loop needs.
    value = math.floor(value * 1000.0) / 1000.0
    if value <= 0:
        raise RuntimeDirectoryError("task lease retirement deadline is too small")
    return value


def _lease_payload(revision, root_uuid, expires_at, valid_for_seconds=None):
    payload = {
        "revision": revision,
        "root_uuid": root_uuid,
        "expires_at": expires_at,
    }
    if valid_for_seconds is not None:
        payload["valid_for_seconds"] = valid_for_seconds
    return payload


def _valid_for_seconds(expires_at, now, ttl_seconds):
    return max(0.0, min(ttl_seconds, expires_at - now))


class TaskLeaseRetired(RuntimeDirectoryConflict):
    """Raised when a task tries to use a fenced directory revision."""

    def __init__(self, revision, deadline=None):
        message = f"runtime directory revision {revision} is retired"
        if deadline is not None:
            message += f" at deadline {deadline}"
        super().__init__(message)
        self.revision = revision
        self.deadline = deadline


class TaskLeaseStore(ABC):
    """Persistence interface for task leases and revision retirement."""

    @abstractmethod
    def acquire(self, revision, root_uuid, active_revision, ttl_seconds=60.0):
        raise NotImplementedError

    @abstractmethod
    def renew(
        self,
        revision,
        root_uuid,
        ttl_seconds=60.0,
        active_revision=None,
    ):
        raise NotImplementedError

    @abstractmethod
    def release(self, revision, root_uuid):
        raise NotImplementedError

    @abstractmethod
    def count(self, revision):
        raise NotImplementedError

    @abstractmethod
    def status(self, revision):
        raise NotImplementedError

    @abstractmethod
    def retire(self, revision, deadline):
        raise NotImplementedError


class InMemoryTaskLeaseStore(TaskLeaseStore):
    """Thread-safe implementation for tests and bootstrap-less development."""

    def __init__(self, clock=None):
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._leases = {}
        self._retirements = {}

    def _prune_locked(self):
        now = self._clock()
        for revision in tuple(self._retirements):
            retirement = self._retirements[revision]
            if not retirement["retired"] and retirement["deadline"] <= now:
                # Only leases that were still live at the immutable deadline
                # count as forced. Reconciliation may run after that instant.
                self._leases = {
                    key: expires_at
                    for key, expires_at in self._leases.items()
                    if key[0] != revision
                    or expires_at >= retirement["deadline"]
                }
            self._retire_if_due_locked(revision, now)
        self._leases = {
            key: expires_at for key, expires_at in self._leases.items()
            if expires_at > now
        }

    def _retire_if_due_locked(self, revision, now=None):
        retirement = self._retirements.get(revision)
        if retirement is None or retirement["retired"]:
            return
        now = self._clock() if now is None else now
        if now < retirement["deadline"]:
            return
        revoked = sum(1 for lease_revision, _ in self._leases if lease_revision == revision)
        self._leases = {
            key: expires_at
            for key, expires_at in self._leases.items()
            if key[0] != revision
        }
        retirement["retired"] = True
        retirement["revoked_count"] += revoked

    def _status_locked(self, revision):
        retirement = self._retirements.get(revision)
        return {
            "revision": revision,
            "count": sum(
                1 for lease_revision, _ in self._leases
                if lease_revision == revision
            ),
            "deadline": retirement["deadline"] if retirement else None,
            "retired": bool(retirement and retirement["retired"]),
            "revoked_count": retirement["revoked_count"] if retirement else 0,
        }

    def acquire(self, revision, root_uuid, active_revision, ttl_seconds=60.0):
        revision = _revision(revision)
        active_revision = _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        with self._lock:
            self._prune_locked()
            retirement = self._retirements.get(revision)
            if retirement is not None:
                raise TaskLeaseRetired(revision, retirement["deadline"])
            if revision != active_revision:
                raise RuntimeDirectoryConflict(
                    f"task lease revision {revision} is not active "
                    f"(active revision is {active_revision})"
                )
            now = self._clock()
            expires_at = now + ttl_seconds
            self._leases[(revision, root_uuid)] = expires_at
            return _lease_payload(
                revision,
                root_uuid,
                expires_at,
                _valid_for_seconds(expires_at, self._clock(), ttl_seconds),
            )

    def renew(
        self,
        revision,
        root_uuid,
        ttl_seconds=60.0,
        active_revision=None,
    ):
        revision = _revision(revision)
        if active_revision is not None:
            active_revision = _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        with self._lock:
            self._prune_locked()
            retirement = self._retirements.get(revision)
            if retirement is None and active_revision is not None and revision != active_revision:
                raise RuntimeDirectoryConflict(
                    f"task lease revision {revision} is not active "
                    f"(active revision is {active_revision})"
                )
            if retirement is not None and retirement["retired"]:
                raise TaskLeaseRetired(revision, retirement["deadline"])
            key = (revision, root_uuid)
            if key not in self._leases:
                raise RuntimeDirectoryNotFound("task lease does not exist or expired")
            now = self._clock()
            expires_at = now + ttl_seconds
            if retirement is not None:
                expires_at = min(expires_at, retirement["deadline"])
            self._leases[key] = expires_at
            return _lease_payload(
                revision,
                root_uuid,
                expires_at,
                _valid_for_seconds(expires_at, self._clock(), ttl_seconds),
            )

    def release(self, revision, root_uuid):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        with self._lock:
            self._prune_locked()
            key = (revision, root_uuid)
            if key not in self._leases:
                retirement = self._retirements.get(revision)
                if retirement is not None:
                    result = _lease_payload(revision, root_uuid, self._clock())
                    result.update({"released": True, "already_released": True})
                    return result
                raise RuntimeDirectoryNotFound("task lease does not exist or expired")
            del self._leases[key]
            result = _lease_payload(revision, root_uuid, self._clock())
            result["released"] = True
            return result

    def count(self, revision):
        return self.status(revision)["count"]

    def status(self, revision):
        revision = _revision(revision)
        with self._lock:
            self._prune_locked()
            return self._status_locked(revision)

    def retire(self, revision, deadline):
        revision = _revision(revision)
        deadline = _deadline(deadline)
        with self._lock:
            self._prune_locked()
            retirement = self._retirements.get(revision)
            if retirement is None:
                retirement = {
                    "deadline": deadline,
                    "retired": False,
                    "revoked_count": 0,
                }
                self._retirements[revision] = retirement
            else:
                retirement["deadline"] = min(retirement["deadline"], deadline)

            # Existing scores must obey the same immutable upper bound as
            # future renewals.  This closes the publication-to-retire window.
            effective_deadline = retirement["deadline"]
            self._leases = {
                key: min(expires_at, effective_deadline)
                if key[0] == revision else expires_at
                for key, expires_at in self._leases.items()
            }
            self._retire_if_due_locked(revision)
            return self._status_locked(revision)


class RedisTaskLeaseStore(TaskLeaseStore):
    """Redis ZSET implementation used by scheduler replicas in production."""

    _ACQUIRE_SCRIPT = """
local revision = tonumber(ARGV[1])
local retirement_raw = redis.call('GET', KEYS[2])
if retirement_raw then
  local retirement = cjson.decode(retirement_raw)
  return {1, tostring(retirement.deadline)}
end
local active_raw = redis.call('GET', KEYS[1])
local active_revision = 0
if active_raw then
  local active = cjson.decode(active_raw)
  active_revision = tonumber(active.revision or active.directory_revision or 0)
end
if active_revision ~= revision then
  return {0, tostring(active_revision)}
end
local now = tonumber(ARGV[2])
local expires = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
redis.call('ZADD', KEYS[3], expires, ARGV[4])
redis.call('EXPIRE', KEYS[3], tonumber(ARGV[5]))
return {2, tostring(expires)}
"""

    _RENEW_SCRIPT = """
local key = KEYS[1]
local retirement_key = KEYS[2]
local active_key = KEYS[3]
local member = ARGV[1]
local now = tonumber(ARGV[2])
local expires = tonumber(ARGV[3])
local revision = tonumber(ARGV[5])
local retirement_raw = redis.call('GET', retirement_key)
if not retirement_raw then
  local active_raw = redis.call('GET', active_key)
  local active_revision = 0
  if active_raw then
    local active = cjson.decode(active_raw)
    active_revision = tonumber(active.revision or active.directory_revision or 0)
  end
  if active_revision ~= revision then
    return {-2, tostring(active_revision)}
  end
end
if retirement_raw then
  local retirement = cjson.decode(retirement_raw)
  local deadline = tonumber(retirement.deadline)
  if retirement.retired == true or deadline <= now then
    if retirement.retired ~= true then
      redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
      local revoked = redis.call('ZCARD', key)
      redis.call('DEL', key)
      retirement.retired = true
      retirement.revoked_count = tonumber(retirement.revoked_count or 0) + revoked
      redis.call('SET', retirement_key, cjson.encode(retirement))
    end
    return {-1, tostring(deadline)}
  end
  if expires > deadline then
    expires = deadline
  end
end
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZSCORE', key, member) == false then
  return {0, ''}
end
redis.call('ZADD', key, expires, member)
redis.call('EXPIRE', key, tonumber(ARGV[4]))
return {1, tostring(expires)}
"""

    _RELEASE_SCRIPT = """
local key = KEYS[1]
local retirement_key = KEYS[2]
local member = ARGV[1]
local now = tonumber(ARGV[2])
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
local removed = redis.call('ZREM', key, member)
if removed == 1 then
  return 1
end
if redis.call('GET', retirement_key) then
  return 2
end
return 0
"""

    _RETIRE_SCRIPT = """
local key = KEYS[1]
local retirement_key = KEYS[2]
local now = tonumber(ARGV[1])
local requested_deadline = tonumber(ARGV[2])
local retirement_raw = redis.call('GET', retirement_key)
local retirement = nil
if retirement_raw then
  retirement = cjson.decode(retirement_raw)
else
  retirement = {deadline=requested_deadline, retired=false, revoked_count=0}
end
if requested_deadline < tonumber(retirement.deadline) then
  retirement.deadline = requested_deadline
end
local deadline = tonumber(retirement.deadline)
if retirement.retired ~= true and deadline <= now then
  redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
  local revoked = redis.call('ZCARD', key)
  redis.call('DEL', key)
  retirement.retired = true
  retirement.revoked_count = tonumber(retirement.revoked_count or 0) + revoked
elseif retirement.retired ~= true then
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
  local leases = redis.call('ZRANGE', key, 0, -1, 'WITHSCORES')
  for index = 1, #leases, 2 do
    if tonumber(leases[index + 1]) > deadline then
      redis.call('ZADD', key, deadline, leases[index])
    end
  end
end
redis.call('SET', retirement_key, cjson.encode(retirement))
local count = redis.call('ZCARD', key)
return cjson.encode({
  count=count,
  deadline=deadline,
  retired=retirement.retired == true,
  revoked_count=tonumber(retirement.revoked_count or 0)
})
"""

    _STATUS_SCRIPT = """
local key = KEYS[1]
local retirement_key = KEYS[2]
local now = tonumber(ARGV[1])
local retirement_raw = redis.call('GET', retirement_key)
local deadline = false
local retired = false
local revoked_count = 0
if retirement_raw then
  local retirement = cjson.decode(retirement_raw)
  deadline = tonumber(retirement.deadline)
  retired = retirement.retired == true
  revoked_count = tonumber(retirement.revoked_count or 0)
  if not retired and deadline <= now then
    redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
    revoked_count = revoked_count + redis.call('ZCARD', key)
    redis.call('DEL', key)
    retired = true
    retirement.retired = true
    retirement.revoked_count = revoked_count
    redis.call('SET', retirement_key, cjson.encode(retirement))
  end
end
if not retired then
  redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
end
return cjson.encode({
  count=redis.call('ZCARD', key),
  deadline=deadline,
  retired=retired,
  revoked_count=revoked_count
})
"""

    def __init__(
            self,
            redis_client,
            install_id="",
            clock=None,
            key_prefix="dayu:runtime-directory:task-leases",
            directory_key_prefix="dayu:runtime-directory",
    ):
        self.redis = redis_client
        self.install_id = str(install_id or "default")
        self.clock = clock or time.time
        self.key_prefix = str(key_prefix)
        self._active_key = f"{str(directory_key_prefix).rstrip(':')}:{self.install_id}:active"

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
                socket_connect_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
                socket_timeout=REDIS_SOCKET_TIMEOUT_SECONDS,
            ),
            install_id=install_id,
            clock=clock,
        )

    def _key(self, revision):
        return f"{self.key_prefix}:{self.install_id}:{revision}"

    def _retirement_key(self, revision):
        return f"{self.key_prefix}:{self.install_id}:{revision}:retirement"

    def acquire(self, revision, root_uuid, active_revision, ttl_seconds=60.0):
        revision = _revision(revision)
        if active_revision is not None:
            _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        now = self.clock()
        expires_at = now + ttl_seconds
        code, value = self.redis.eval(
            self._ACQUIRE_SCRIPT,
            3,
            self._active_key,
            self._retirement_key(revision),
            self._key(revision),
            revision,
            now,
            expires_at,
            root_uuid,
            max(1, int(math.ceil(ttl_seconds)) + 60),
        )
        code = int(code)
        if code == 0:
            raise RuntimeDirectoryConflict(
                f"task lease revision {revision} is not active (active revision is {value})"
            )
        if code == 1:
            raise TaskLeaseRetired(revision, float(value))
        expires_at = float(value)
        valid_for_seconds = _valid_for_seconds(
            expires_at, self.clock(), ttl_seconds
        )
        return _lease_payload(
            revision, root_uuid, expires_at, valid_for_seconds
        )

    def renew(
        self,
        revision,
        root_uuid,
        ttl_seconds=60.0,
        active_revision=None,
    ):
        revision = _revision(revision)
        if active_revision is not None:
            _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        now = self.clock()
        expires_at = now + ttl_seconds
        result = self.redis.eval(
            self._RENEW_SCRIPT,
            3,
            self._key(revision),
            self._retirement_key(revision),
            self._active_key,
            root_uuid,
            now,
            expires_at,
            max(1, int(math.ceil(ttl_seconds)) + 60),
            revision,
        )
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RuntimeDirectoryError("Redis task lease renewal returned an invalid result")
        code, value = int(result[0]), result[1]
        if code == -1:
            raise TaskLeaseRetired(revision, float(value))
        if code == -2:
            raise RuntimeDirectoryConflict(
                f"task lease revision {revision} is not active "
                f"(active revision is {value})"
            )
        if code != 1:
            raise RuntimeDirectoryNotFound("task lease does not exist or expired")
        expires_at = float(value)
        valid_for_seconds = _valid_for_seconds(
            expires_at, self.clock(), ttl_seconds
        )
        return _lease_payload(
            revision, root_uuid, expires_at, valid_for_seconds
        )

    def release(self, revision, root_uuid):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        release_code = int(self.redis.eval(
            self._RELEASE_SCRIPT,
            2,
            self._key(revision),
            self._retirement_key(revision),
            root_uuid,
            self.clock(),
        ) or 0)
        if release_code == 0:
            raise RuntimeDirectoryNotFound("task lease does not exist or expired")
        result = _lease_payload(revision, root_uuid, self.clock())
        result["released"] = True
        if release_code == 2:
            result["already_released"] = True
        return result

    def count(self, revision):
        return self.status(revision)["count"]

    def status(self, revision):
        revision = _revision(revision)
        raw = self.redis.eval(
            self._STATUS_SCRIPT,
            2,
            self._key(revision),
            self._retirement_key(revision),
            self.clock(),
        )
        try:
            result = json.loads(raw)
            return {
                "revision": revision,
                "count": int(result["count"]),
                "deadline": (
                    None if result.get("deadline") in (None, False)
                    else float(result["deadline"])
                ),
                "retired": bool(result.get("retired")),
                "revoked_count": int(result.get("revoked_count") or 0),
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeDirectoryError("persisted task lease retirement is corrupt") from exc

    def retire(self, revision, deadline):
        revision = _revision(revision)
        deadline = _deadline(deadline)
        raw = self.redis.eval(
            self._RETIRE_SCRIPT,
            2,
            self._key(revision),
            self._retirement_key(revision),
            self.clock(),
            deadline,
        )
        try:
            result = json.loads(raw)
            return {
                "revision": revision,
                "count": int(result["count"]),
                "deadline": float(result["deadline"]),
                "retired": bool(result.get("retired")),
                "revoked_count": int(result.get("revoked_count") or 0),
            }
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeDirectoryError("persisted task lease retirement is corrupt") from exc


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
