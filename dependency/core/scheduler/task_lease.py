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


def _task_context(value, revision, root_uuid):
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise RuntimeDirectoryError("task execution context must be an object")
    try:
        normalized = json.loads(json.dumps(value, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise RuntimeDirectoryError("task execution context must be JSON serializable") from exc
    normalized_root = str(normalized.get("root_uuid") or root_uuid)
    if normalized_root != root_uuid:
        raise RuntimeDirectoryError("task execution context root_uuid does not match lease")
    normalized_revision = normalized.get("runtime_directory_revision", revision)
    try:
        normalized_revision = int(normalized_revision)
    except (TypeError, ValueError) as exc:
        raise RuntimeDirectoryError(
            "task execution context runtime_directory_revision must be an integer"
        ) from exc
    if normalized_revision != revision:
        raise RuntimeDirectoryError(
            "task execution context runtime_directory_revision does not match lease"
        )
    normalized["root_uuid"] = root_uuid
    normalized["runtime_directory_revision"] = revision
    return normalized


def _context_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _reservation_matches_context(reservation, context):
    for field in (
        "source_id",
        "task_id",
        "decision_id",
        "plan_digest",
        "deployment_version",
    ):
        expected = reservation.get(field)
        actual = context.get(field)
        if expected not in (None, "") and expected != actual:
            return False
    return True


def _record_payload(context, timestamp, expires_at, status):
    payload = dict(context)
    payload.update({
        "expires_at": float(expires_at),
        "status": status,
    })
    if status == "pending":
        payload["reserved_at"] = float(timestamp)
    else:
        payload["admitted_at"] = float(timestamp)
    return payload


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
    """Persistence interface for task admission, leases, and retirement."""

    @abstractmethod
    def reserve(
        self,
        revision,
        root_uuid,
        context,
        active_revision,
        ttl_seconds=60.0,
    ):
        raise NotImplementedError

    @abstractmethod
    def cancel_reservation(self, revision, root_uuid, decision_id=None):
        """Cancel a pending decision that never became a task lease."""

        raise NotImplementedError

    @abstractmethod
    def acquire(
        self,
        revision,
        root_uuid,
        active_revision,
        ttl_seconds=60.0,
        context=None,
    ):
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

    @abstractmethod
    def list_reservations(self):
        raise NotImplementedError

    @abstractmethod
    def get_reservation(self, revision, root_uuid):
        """Return one live pending reservation without scanning the store."""

        raise NotImplementedError

    @abstractmethod
    def list_active(self):
        raise NotImplementedError


class InMemoryTaskLeaseStore(TaskLeaseStore):
    """Thread-safe implementation for tests and bootstrap-less development."""

    def __init__(self, clock=None):
        self._clock = clock or time.time
        self._lock = threading.RLock()
        self._leases = {}
        self._retirements = {}
        self._reservations = {}
        self._contexts = {}

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
        self._contexts = {
            key: record for key, record in self._contexts.items()
            if key in self._leases
        }
        self._reservations = {
            root_uuid: record
            for root_uuid, record in self._reservations.items()
            if record["expires_at"] > now
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
        self._contexts = {
            key: record
            for key, record in self._contexts.items()
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

    def reserve(
        self,
        revision,
        root_uuid,
        context,
        active_revision,
        ttl_seconds=60.0,
    ):
        revision = _revision(revision)
        active_revision = _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        normalized = _task_context(context, revision, root_uuid)
        with self._lock:
            self._prune_locked()
            retirement = self._retirements.get(revision)
            if retirement is not None:
                raise TaskLeaseRetired(revision, retirement["deadline"])
            if revision != active_revision:
                raise RuntimeDirectoryConflict(
                    f"task reservation revision {revision} is not active "
                    f"(active revision is {active_revision})"
                )
            if any(active_root == root_uuid for _, active_root in self._contexts):
                raise RuntimeDirectoryConflict(
                    "task reservation root_uuid is already admitted"
                )
            previous = self._reservations.get(root_uuid)
            if (
                previous is not None
                and previous["context"].get("runtime_directory_revision") != revision
            ):
                self._reservations.pop(root_uuid, None)
                previous = None
            if previous is not None and previous["context"] != normalized:
                raise RuntimeDirectoryConflict(
                    "task reservation changed for an existing root_uuid"
                )
            now = self._clock()
            expires_at = now + ttl_seconds
            reserved_at = previous["reserved_at"] if previous else now
            self._reservations[root_uuid] = {
                "context": normalized,
                "reserved_at": reserved_at,
                "expires_at": expires_at,
            }
            return _record_payload(
                normalized, reserved_at, expires_at, "pending"
            )

    def acquire(
        self,
        revision,
        root_uuid,
        active_revision,
        ttl_seconds=60.0,
        context=None,
    ):
        revision = _revision(revision)
        active_revision = _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        normalized = _task_context(context, revision, root_uuid)
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
            key = (revision, root_uuid)
            previous = self._contexts.get(key)
            if previous is not None and previous["context"] != normalized:
                raise RuntimeDirectoryConflict(
                    "task execution context changed for an existing root_uuid"
                )
            reservation = self._reservations.get(root_uuid)
            if reservation is not None and not _reservation_matches_context(
                reservation["context"], normalized
            ):
                raise RuntimeDirectoryConflict(
                    "task execution context does not match its reservation"
                )
            now = self._clock()
            expires_at = now + ttl_seconds
            self._leases[key] = expires_at
            admitted_at = previous["admitted_at"] if previous else now
            self._contexts[key] = {
                "context": normalized,
                "admitted_at": admitted_at,
            }
            self._reservations.pop(root_uuid, None)
            return _lease_payload(
                revision,
                root_uuid,
                expires_at,
                _valid_for_seconds(expires_at, self._clock(), ttl_seconds),
            )

    def cancel_reservation(self, revision, root_uuid, decision_id=None):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        decision_id = str(decision_id or "").strip()
        with self._lock:
            self._prune_locked()
            reservation = self._reservations.get(root_uuid)
            if reservation is None:
                return {
                    "revision": revision,
                    "root_uuid": root_uuid,
                    "cancelled": True,
                    "already_cancelled": True,
                    "decision_id": decision_id,
                }
            context = reservation["context"]
            if int(context.get("runtime_directory_revision") or 0) != revision:
                raise RuntimeDirectoryConflict(
                    "task reservation revision does not match cancellation request"
                )
            expected_decision = str(context.get("decision_id") or "")
            if decision_id and expected_decision != decision_id:
                raise RuntimeDirectoryConflict(
                    "task reservation decision_id does not match cancellation request"
                )
            self._reservations.pop(root_uuid, None)
            return {
                "revision": revision,
                "root_uuid": root_uuid,
                "cancelled": True,
                "decision_id": expected_decision,
            }

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
            self._contexts.pop(key, None)
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

    def list_reservations(self):
        with self._lock:
            self._prune_locked()
            return [
                _record_payload(
                    record["context"],
                    record["reserved_at"],
                    record["expires_at"],
                    "pending",
                )
                for record in self._reservations.values()
            ]

    def get_reservation(self, revision, root_uuid):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        with self._lock:
            self._prune_locked()
            record = self._reservations.get(root_uuid)
            if record is None:
                return None
            context = record["context"]
            if int(context.get("runtime_directory_revision") or 0) != revision:
                return None
            return _record_payload(
                context,
                record["reserved_at"],
                record["expires_at"],
                "pending",
            )

    def list_active(self):
        with self._lock:
            self._prune_locked()
            return [
                _record_payload(
                    record["context"],
                    record["admitted_at"],
                    self._leases[key],
                    "active",
                )
                for key, record in self._contexts.items()
            ]


class RedisTaskLeaseStore(TaskLeaseStore):
    """Redis implementation used by the production Scheduler."""

    _RESERVE_SCRIPT = """
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
local member = ARGV[4]
local context = ARGV[5]
local reserved_at = now
if redis.call('HGET', KEYS[5], member) then
  return {4, ''}
end
local expired = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for _, expired_member in ipairs(expired) do
  redis.call('HDEL', KEYS[4], expired_member)
end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
local score = redis.call('ZSCORE', KEYS[3], member)
local existing = redis.call('HGET', KEYS[4], member)
if existing then
  local record = cjson.decode(existing)
  local previous_context = cjson.decode(record.context)
  if tonumber(previous_context.runtime_directory_revision or 0) ~= revision then
    redis.call('ZREM', KEYS[3], member)
    redis.call('HDEL', KEYS[4], member)
    existing = false
  else
    if record.context ~= context then
      return {3, existing}
    end
    reserved_at = tonumber(record.reserved_at)
  end
end
if not existing then
  redis.call('HSET', KEYS[4], member, cjson.encode({context=context, reserved_at=now}))
end
redis.call('ZADD', KEYS[3], expires, member)
return {2, tostring(expires), tostring(reserved_at)}
"""

    _CANCEL_RESERVATION_SCRIPT = """
local revision = tonumber(ARGV[1])
local now = tonumber(ARGV[2])
local member = ARGV[3]
local decision_id = ARGV[4]
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now)
for _, expired_member in ipairs(expired) do
  redis.call('HDEL', KEYS[2], expired_member)
end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local score = redis.call('ZSCORE', KEYS[1], member)
local raw = redis.call('HGET', KEYS[2], member)
if not score or not raw then
  return {0, ''}
end
local record = cjson.decode(raw)
local context = cjson.decode(record.context)
if tonumber(context.runtime_directory_revision or 0) ~= revision then
  return {2, tostring(context.runtime_directory_revision or '')}
end
local expected_decision = tostring(context.decision_id or '')
if decision_id ~= '' and expected_decision ~= decision_id then
  return {3, expected_decision}
end
redis.call('ZREM', KEYS[1], member)
redis.call('HDEL', KEYS[2], member)
return {1, expected_decision}
"""

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
local member = ARGV[4]
local ttl = tonumber(ARGV[5])
local context = ARGV[6]
local expired = redis.call('ZRANGEBYSCORE', KEYS[3], '-inf', now)
for _, expired_member in ipairs(expired) do
  redis.call('HDEL', KEYS[4], expired_member)
end
redis.call('ZREMRANGEBYSCORE', KEYS[3], '-inf', now)
local existing = redis.call('HGET', KEYS[4], member)
if existing then
  local record = cjson.decode(existing)
  if record.context ~= context then
    return {3, existing}
  end
end
local expired_reservations = redis.call('ZRANGEBYSCORE', KEYS[5], '-inf', now)
for _, expired_member in ipairs(expired_reservations) do
  redis.call('HDEL', KEYS[6], expired_member)
end
redis.call('ZREMRANGEBYSCORE', KEYS[5], '-inf', now)
local reservation_score = redis.call('ZSCORE', KEYS[5], member)
local reservation_raw = redis.call('HGET', KEYS[6], member)
if reservation_score and reservation_raw then
  local reservation = cjson.decode(reservation_raw)
  local expected = cjson.decode(reservation.context)
  local actual = cjson.decode(context)
  local fields = {
    'source_id',
    'task_id',
    'decision_id',
    'plan_digest',
    'deployment_version'
  }
  for _, field in ipairs(fields) do
    if expected[field] ~= nil and expected[field] ~= '' then
      if actual[field] == nil or tostring(expected[field]) ~= tostring(actual[field]) then
        return {4, reservation_raw}
      end
    end
  end
end
if not existing then
  redis.call('HSET', KEYS[4], member, cjson.encode({context=context, admitted_at=now}))
end
redis.call('ZADD', KEYS[3], expires, member)
redis.call('ZREM', KEYS[5], member)
redis.call('HDEL', KEYS[6], member)
redis.call('EXPIRE', KEYS[3], ttl)
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
local ttl = tonumber(ARGV[4])
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
      local expired = redis.call('ZRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
      for _, expired_member in ipairs(expired) do
        redis.call('HDEL', KEYS[4], expired_member)
      end
      redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
      local revoked = redis.call('ZCARD', key)
      local revoked_members = redis.call('ZRANGE', key, 0, -1)
      for _, revoked_member in ipairs(revoked_members) do
        redis.call('HDEL', KEYS[4], revoked_member)
      end
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
local expired = redis.call('ZRANGEBYSCORE', key, '-inf', now)
for _, expired_member in ipairs(expired) do
  redis.call('HDEL', KEYS[4], expired_member)
end
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZSCORE', key, member) == false then
  return {0, ''}
end
redis.call('ZADD', key, expires, member)
redis.call('EXPIRE', key, ttl)
return {1, tostring(expires)}
"""

    _RELEASE_SCRIPT = """
local key = KEYS[1]
local retirement_key = KEYS[2]
local member = ARGV[1]
local now = tonumber(ARGV[2])
local expired = redis.call('ZRANGEBYSCORE', key, '-inf', now)
for _, expired_member in ipairs(expired) do
  redis.call('HDEL', KEYS[3], expired_member)
end
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
local removed = redis.call('ZREM', key, member)
if removed == 1 then
  redis.call('HDEL', KEYS[3], member)
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
  local expired = redis.call('ZRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
  for _, expired_member in ipairs(expired) do
    redis.call('HDEL', KEYS[3], expired_member)
  end
  redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
  local revoked = redis.call('ZCARD', key)
  local revoked_members = redis.call('ZRANGE', key, 0, -1)
  for _, revoked_member in ipairs(revoked_members) do
    redis.call('HDEL', KEYS[3], revoked_member)
  end
  redis.call('DEL', key)
  retirement.retired = true
  retirement.revoked_count = tonumber(retirement.revoked_count or 0) + revoked
elseif retirement.retired ~= true then
  local expired = redis.call('ZRANGEBYSCORE', key, '-inf', now)
  for _, expired_member in ipairs(expired) do
    redis.call('HDEL', KEYS[3], expired_member)
  end
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
    local expired = redis.call('ZRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
    for _, expired_member in ipairs(expired) do
      redis.call('HDEL', KEYS[3], expired_member)
    end
    redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. tostring(deadline))
    revoked_count = revoked_count + redis.call('ZCARD', key)
    local revoked_members = redis.call('ZRANGE', key, 0, -1)
    for _, revoked_member in ipairs(revoked_members) do
      redis.call('HDEL', KEYS[3], revoked_member)
    end
    redis.call('DEL', key)
    retired = true
    retirement.retired = true
    retirement.revoked_count = revoked_count
    redis.call('SET', retirement_key, cjson.encode(retirement))
  end
end
if not retired then
  local expired = redis.call('ZRANGEBYSCORE', key, '-inf', now)
  for _, expired_member in ipairs(expired) do
    redis.call('HDEL', KEYS[3], expired_member)
  end
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

    @property
    def _context_key(self):
        return f"{self.key_prefix}:{self.install_id}:contexts"

    @property
    def _reservation_key(self):
        return f"{self.key_prefix}:{self.install_id}:reservations"

    @property
    def _reservation_context_key(self):
        return f"{self.key_prefix}:{self.install_id}:reservation-contexts"

    @staticmethod
    def _decode_record(raw, label):
        try:
            record = json.loads(raw)
            context = json.loads(record["context"])
            if not isinstance(context, dict):
                raise TypeError
            return record, context
        except (TypeError, ValueError, KeyError) as exc:
            raise RuntimeDirectoryError(
                f"persisted task {label} is corrupt"
            ) from exc

    def _zscore_many(self, requests):
        """Read many scores in one Redis round trip when pipelines exist."""

        requests = list(requests)
        if not requests:
            return []
        pipeline_factory = getattr(self.redis, "pipeline", None)
        if not callable(pipeline_factory):
            return [self.redis.zscore(key, member) for key, member in requests]
        pipeline = pipeline_factory(transaction=False)
        for key, member in requests:
            pipeline.zscore(key, member)
        return pipeline.execute()

    def reserve(
        self,
        revision,
        root_uuid,
        context,
        active_revision,
        ttl_seconds=60.0,
    ):
        revision = _revision(revision)
        if active_revision is not None:
            _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        normalized = _task_context(context, revision, root_uuid)
        now = self.clock()
        expires_at = now + ttl_seconds
        result = self.redis.eval(
            self._RESERVE_SCRIPT,
            5,
            self._active_key,
            self._retirement_key(revision),
            self._reservation_key,
            self._reservation_context_key,
            self._context_key,
            revision,
            now,
            expires_at,
            root_uuid,
            _context_json(normalized),
        )
        code, value = int(result[0]), result[1]
        if code == 0:
            raise RuntimeDirectoryConflict(
                f"task reservation revision {revision} is not active "
                f"(active revision is {value})"
            )
        if code == 1:
            raise TaskLeaseRetired(revision, float(value))
        if code == 3:
            raise RuntimeDirectoryConflict(
                "task reservation changed for an existing root_uuid"
            )
        if code == 4:
            raise RuntimeDirectoryConflict(
                "task reservation root_uuid is already admitted"
            )
        expires_at = float(value)
        reserved_at = float(result[2]) if len(result) > 2 else now
        return _record_payload(normalized, reserved_at, expires_at, "pending")

    def acquire(
        self,
        revision,
        root_uuid,
        active_revision,
        ttl_seconds=60.0,
        context=None,
    ):
        revision = _revision(revision)
        if active_revision is not None:
            _revision(active_revision)
        root_uuid = _root_uuid(root_uuid)
        ttl_seconds = _ttl(ttl_seconds)
        normalized = _task_context(context, revision, root_uuid)
        now = self.clock()
        expires_at = now + ttl_seconds
        code, value = self.redis.eval(
            self._ACQUIRE_SCRIPT,
            6,
            self._active_key,
            self._retirement_key(revision),
            self._key(revision),
            self._context_key,
            self._reservation_key,
            self._reservation_context_key,
            revision,
            now,
            expires_at,
            root_uuid,
            max(1, int(math.ceil(ttl_seconds)) + 60),
            _context_json(normalized),
        )
        code = int(code)
        if code == 0:
            raise RuntimeDirectoryConflict(
                f"task lease revision {revision} is not active (active revision is {value})"
            )
        if code == 1:
            raise TaskLeaseRetired(revision, float(value))
        if code == 3:
            raise RuntimeDirectoryConflict(
                "task execution context changed for an existing root_uuid"
            )
        if code == 4:
            raise RuntimeDirectoryConflict(
                "task execution context does not match its reservation"
            )
        expires_at = float(value)
        valid_for_seconds = _valid_for_seconds(
            expires_at, self.clock(), ttl_seconds
        )
        return _lease_payload(
            revision, root_uuid, expires_at, valid_for_seconds
        )

    def cancel_reservation(self, revision, root_uuid, decision_id=None):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        decision_id = str(decision_id or "").strip()
        result = self.redis.eval(
            self._CANCEL_RESERVATION_SCRIPT,
            2,
            self._reservation_key,
            self._reservation_context_key,
            revision,
            self.clock(),
            root_uuid,
            decision_id,
        )
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            raise RuntimeDirectoryError(
                "Redis task reservation cancellation returned an invalid result"
            )
        code, value = int(result[0]), str(result[1] or "")
        if code == 0:
            return {
                "revision": revision,
                "root_uuid": root_uuid,
                "cancelled": True,
                "already_cancelled": True,
                "decision_id": decision_id,
            }
        if code == 2:
            raise RuntimeDirectoryConflict(
                "task reservation revision does not match cancellation request"
            )
        if code == 3:
            raise RuntimeDirectoryConflict(
                "task reservation decision_id does not match cancellation request"
            )
        if code != 1:
            raise RuntimeDirectoryError(
                "Redis task reservation cancellation returned an unknown result"
            )
        return {
            "revision": revision,
            "root_uuid": root_uuid,
            "cancelled": True,
            "decision_id": value,
        }

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
            4,
            self._key(revision),
            self._retirement_key(revision),
            self._active_key,
            self._context_key,
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
            3,
            self._key(revision),
            self._retirement_key(revision),
            self._context_key,
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
            3,
            self._key(revision),
            self._retirement_key(revision),
            self._context_key,
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
            3,
            self._key(revision),
            self._retirement_key(revision),
            self._context_key,
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

    def list_reservations(self):
        now = self.clock()
        records = []
        stale = []
        persisted = self.redis.hgetall(self._reservation_context_key)
        scores = self._zscore_many(
            (self._reservation_key, root_uuid) for root_uuid in persisted
        )
        for (root_uuid, raw), score in zip(persisted.items(), scores):
            if score is None or float(score) <= now:
                stale.append(root_uuid)
                continue
            record, context = self._decode_record(raw, "reservation")
            records.append(_record_payload(
                context,
                float(record["reserved_at"]),
                float(score),
                "pending",
            ))
        if stale:
            self.redis.hdel(self._reservation_context_key, *stale)
            self.redis.zrem(self._reservation_key, *stale)
        return records

    def get_reservation(self, revision, root_uuid):
        revision = _revision(revision)
        root_uuid = _root_uuid(root_uuid)
        now = self.clock()
        pipeline_factory = getattr(self.redis, "pipeline", None)
        if callable(pipeline_factory):
            pipeline = pipeline_factory(transaction=False)
            pipeline.hget(self._reservation_context_key, root_uuid)
            pipeline.zscore(self._reservation_key, root_uuid)
            raw, score = pipeline.execute()
        else:
            raw = self.redis.hget(self._reservation_context_key, root_uuid)
            score = self.redis.zscore(self._reservation_key, root_uuid)
        if raw is None or score is None or float(score) <= now:
            if raw is not None or score is not None:
                self.redis.hdel(self._reservation_context_key, root_uuid)
                self.redis.zrem(self._reservation_key, root_uuid)
            return None
        record, context = self._decode_record(raw, "reservation")
        if int(context.get("runtime_directory_revision") or 0) != revision:
            return None
        return _record_payload(
            context,
            float(record["reserved_at"]),
            float(score),
            "pending",
        )

    def list_active(self):
        now = self.clock()
        records = []
        stale = []
        persisted = self.redis.hgetall(self._context_key)
        decoded = []
        score_requests = []
        for root_uuid, raw in persisted.items():
            record, context = self._decode_record(raw, "execution context")
            revision = _revision(context.get("runtime_directory_revision"))
            decoded.append((root_uuid, record, context))
            score_requests.append((self._key(revision), root_uuid))
        scores = self._zscore_many(score_requests)
        for (root_uuid, record, context), score in zip(decoded, scores):
            if score is None or float(score) <= now:
                stale.append(root_uuid)
                continue
            records.append(_record_payload(
                context,
                float(record["admitted_at"]),
                float(score),
                "active",
            ))
        if stale:
            self.redis.hdel(self._context_key, *stale)
        return records


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
