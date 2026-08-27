"""Shared persistence for task barriers.

The store owns only barrier identity, branch payloads, expiry, and read-only
state.  DAG-specific validation and task deserialization remain with the
caller.
"""

import math


class TaskBarrierError(RuntimeError):
    """Raised when task-barrier state cannot be read or changed safely."""


class TaskBarrierStore:
    _ARRIVE_SCRIPT = """
local key = KEYS[1]
local branch = ARGV[1]
local payload = ARGV[2]
local ttl = tonumber(ARGV[3])
local required = tonumber(ARGV[4])
redis.call('HSET', key, branch, payload)
redis.call('EXPIRE', key, ttl)
if redis.call('HLEN', key) < required then
  return {}
end
return redis.call('HGETALL', key)
"""

    def __init__(
        self,
        redis_client,
        ttl_seconds,
        key_prefix="dayu:dag:joint_service",
    ):
        try:
            ttl_seconds = max(1, int(math.ceil(float(ttl_seconds))))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("task barrier ttl_seconds must be numeric") from exc
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.key_prefix = str(key_prefix).rstrip(":")

    @staticmethod
    def _identity(value, label):
        value = str(value or "").strip()
        if not value:
            raise TaskBarrierError(f"task barrier {label} is required")
        return value

    @staticmethod
    def _text(value):
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def key(self, root_uuid, barrier):
        root_uuid = self._identity(root_uuid, "root_uuid")
        barrier = self._identity(barrier, "name")
        return f"{self.key_prefix}:{root_uuid}:{barrier}"

    def arrive(self, root_uuid, barrier, branch, payload, required_count):
        branch = self._identity(branch, "branch")
        try:
            required_count = int(required_count)
        except (TypeError, ValueError) as exc:
            raise TaskBarrierError("task barrier required_count must be an integer") from exc
        if required_count < 1:
            raise TaskBarrierError("task barrier required_count must be positive")
        try:
            return self.redis.eval(
                self._ARRIVE_SCRIPT,
                1,
                self.key(root_uuid, barrier),
                branch,
                payload,
                self.ttl_seconds,
                required_count,
            )
        except Exception as exc:
            raise TaskBarrierError(f"failed to retain task barrier: {exc}") from exc

    def complete(self, root_uuid, barrier):
        try:
            self.redis.delete(self.key(root_uuid, barrier))
        except Exception as exc:
            raise TaskBarrierError(f"failed to complete task barrier: {exc}") from exc

    def snapshot(self, barriers):
        """Read exact known barrier keys without scanning the Redis namespace."""

        requests = []
        for item in barriers or []:
            if not isinstance(item, dict):
                raise TaskBarrierError("task barrier snapshot entries must be objects")
            root_uuid = self._identity(item.get("root_uuid"), "root_uuid")
            barrier = self._identity(item.get("barrier"), "name")
            expected = sorted({
                self._identity(branch, "branch")
                for branch in item.get("expected_branches", [])
                if str(branch or "").strip()
            })
            required_count = item.get("required_count", len(expected))
            try:
                required_count = int(required_count)
            except (TypeError, ValueError) as exc:
                raise TaskBarrierError(
                    "task barrier required_count must be an integer"
                ) from exc
            if required_count < 1:
                continue
            requests.append((root_uuid, barrier, expected, required_count))

        if not requests:
            return []

        try:
            pipeline_factory = getattr(self.redis, "pipeline", None)
            if callable(pipeline_factory):
                pipeline = pipeline_factory(transaction=False)
                for root_uuid, barrier, _, _ in requests:
                    pipeline.hkeys(self.key(root_uuid, barrier))
                    pipeline.ttl(self.key(root_uuid, barrier))
                values = pipeline.execute()
                observations = [
                    (values[index * 2], values[index * 2 + 1])
                    for index in range(len(requests))
                ]
            else:
                observations = [
                    (
                        self.redis.hkeys(self.key(root_uuid, barrier)),
                        self.redis.ttl(self.key(root_uuid, barrier)),
                    )
                    for root_uuid, barrier, _, _ in requests
                ]
        except Exception as exc:
            raise TaskBarrierError(f"failed to read task barriers: {exc}") from exc

        result = []
        for request, observation in zip(requests, observations):
            root_uuid, barrier, expected, required_count = request
            fields, ttl = observation
            arrived = sorted({self._text(field) for field in fields or []})
            if not arrived:
                continue
            try:
                ttl = int(ttl)
            except (TypeError, ValueError):
                ttl = -1
            result.append({
                "root_uuid": root_uuid,
                "barrier": barrier,
                "arrived_branches": arrived,
                "expected_branches": expected,
                "required_count": required_count,
                "ready": len(arrived) >= required_count,
                "expires_in_seconds": ttl if ttl >= 0 else None,
            })
        return result
