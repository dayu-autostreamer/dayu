import math

import redis

from core.lib.common import Context, LOGGER
from core.lib.content import Task
from core.lib.runtime import (
    RuntimeContext,
    RuntimeEndpoint,
    RuntimeResolver,
    TaskBarrierError,
    TaskBarrierStore,
)


class TaskCoordinationError(RuntimeError):
    """Raised when a parallel DAG barrier cannot safely retain ownership."""


class TaskCoordinator:
    def __init__(self, runtime_context=None, redis_endpoint=None, barrier_store=None):
        runtime_context = runtime_context or RuntimeContext.get_default()
        self.max_connections = int(Context.get_parameter('MAX_REDIS_CONNECTIONS', '10', direct=False))
        self.storage_timeout = max(1, int(math.ceil(runtime_context.lease_ttl_seconds)))
        self.joint_service_key_prefix = 'dayu:dag:joint_service'
        if barrier_store is not None:
            self.barrier_store = barrier_store
            self.redis = barrier_store.redis
            self.pool = getattr(barrier_store, 'pool', None)
            return
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
        self.barrier_store = TaskBarrierStore(
            self.redis,
            ttl_seconds=self.storage_timeout,
            key_prefix=self.joint_service_key_prefix,
        )

    def _get_joint_service_key(self, root_uuid, joint_service_name):
        return self.barrier_store.key(root_uuid, joint_service_name)

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
            result = self.barrier_store.arrive(
                task.get_root_uuid(),
                joint_service_name,
                branch,
                task.serialize(),
                required_count,
            )
        except TaskBarrierError as exc:
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
            self.barrier_store.complete(root_uuid, joint_service_name)
        except TaskBarrierError as exc:
            raise TaskCoordinationError(f"failed to complete parallel task barrier: {exc}") from exc
