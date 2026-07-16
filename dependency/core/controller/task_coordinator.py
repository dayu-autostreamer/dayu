import math

import redis

from core.lib.common import Context, LOGGER
from core.lib.content import Task
from core.lib.runtime import RuntimeContext, RuntimeEndpoint, RuntimeResolver


class TaskCoordinationError(RuntimeError):
    """Raised when a parallel DAG barrier cannot safely retain ownership."""


class TaskCoordinator:
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

    def __init__(self, runtime_context=None, redis_endpoint=None):
        runtime_context = runtime_context or RuntimeContext.get_default()
        self.max_connections = int(Context.get_parameter('MAX_REDIS_CONNECTIONS', '10', direct=False))
        self.storage_timeout = max(1, int(math.ceil(runtime_context.lease_ttl_seconds)))
        endpoint = RuntimeEndpoint.from_value(redis_endpoint, component="redis") if redis_endpoint else None
        endpoint = endpoint or RuntimeResolver(runtime_context).resolve(
            "redis", target_node=runtime_context.cloud_node or None
        )
        if not endpoint.fqdn or not endpoint.port:
            raise ValueError("redis bootstrap endpoint requires fqdn and port")
        self.pool = redis.ConnectionPool(
            host=endpoint.connection_host,
            port=endpoint.port,
            max_connections=self.max_connections,
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        self.joint_service_key_prefix = 'dayu:dag:joint_service'

    def _get_joint_service_key(self, root_uuid, joint_service_name):
        return f"{self.joint_service_key_prefix}:{root_uuid}:{joint_service_name}"

    def arrive(self, task, joint_service_name, required_count):
        """Idempotently retain one predecessor and return a ready barrier.

        The predecessor service is the stable branch identity. Re-delivering
        the same branch overwrites its previous value instead of incrementing
        the barrier. The hash remains in Redis until ``complete`` is called
        after the merged task has been acknowledged downstream.
        """
        branch = str(task.get_past_flow_index() or "")
        if not branch:
            raise TaskCoordinationError("parallel task has no predecessor identity")
        try:
            result = self.redis.eval(
                self._ARRIVE_SCRIPT,
                1,
                self._get_joint_service_key(task.get_root_uuid(), joint_service_name),
                branch,
                task.serialize(),
                self.storage_timeout,
                required_count,
            )
        except Exception as exc:
            raise TaskCoordinationError(f"failed to retain parallel task: {exc}") from exc

        if not result:
            return None

        try:
            tasks = [Task.deserialize(result[index + 1]) for index in range(0, len(result), 2)]
        except Exception as exc:
            raise TaskCoordinationError(f"parallel task barrier is corrupt: {exc}") from exc

        predecessors = {item.get_past_flow_index() for item in tasks}
        joint_services = {item.get_flow_index() for item in tasks}
        if len(tasks) != required_count or len(predecessors) != required_count:
            raise TaskCoordinationError(
                f"parallel task barrier expected {required_count} unique predecessors, "
                f"got {sorted(predecessors)}"
            )
        if joint_services != {joint_service_name}:
            raise TaskCoordinationError(
                f"parallel task barrier targets conflicting services: {sorted(joint_services)}"
            )

        LOGGER.debug(
            f"Parallel task barrier ready: root={task.get_root_uuid()} "
            f"service={joint_service_name} predecessors={sorted(predecessors)}"
        )
        return tasks

    def complete(self, root_uuid, joint_service_name):
        """Commit a barrier only after its merged task is owned downstream."""
        try:
            self.redis.delete(self._get_joint_service_key(root_uuid, joint_service_name))
        except Exception as exc:
            raise TaskCoordinationError(f"failed to complete parallel task barrier: {exc}") from exc
