import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task
from core.lib.runtime import RuntimeResolver
from core.lib.scheduling import service_waiting_count

from .base_visualizer import BaseVisualizer

__all__ = ("ServiceQueueLengthVisualizer",)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias="service_queue_length")
class ServiceQueueLengthVisualizer(BaseVisualizer, abc.ABC):
    @staticmethod
    def _extract_queue_length(resource, device_name, service_name):
        if not isinstance(resource, dict):
            return 0

        device_resource = resource.get(device_name)
        if not isinstance(device_resource, dict):
            return 0

        return service_waiting_count(device_resource, service_name)

    @staticmethod
    def _format_replica_label(device_name, pod_name):
        if not pod_name:
            return device_name or "unknown-replica"
        if len(pod_name) <= 36:
            return f"{device_name}/{pod_name}" if device_name else pod_name
        return f"{device_name}/{pod_name[:16]}...{pod_name[-12:]}" if device_name else pod_name

    @staticmethod
    def _list_service_replicas(service_name, task):
        replicas = []
        seen = set()
        for route in RuntimeResolver.list_routes(task, component='processor', logical_service=service_name):
            pod_name = route.endpoint_pod_uid or route.runtime_id
            key = (route.target_node, pod_name)
            if key in seen:
                continue
            seen.add(key)
            replicas.append({"device": route.target_node, "pod_name": pod_name})

        replicas.sort(key=lambda item: (item.get("device", ""), item.get("pod_name", "")))
        return replicas

    def __call__(self, task: Task, resource=None, **_):
        result = {}

        for service_name in self.variables:
            records = []
            for replica in self._list_service_replicas(service_name, task):
                device_name = replica.get("device", "")
                pod_name = replica.get("pod_name", "")
                records.append(
                    {
                        "device": device_name,
                        "pod_name": pod_name,
                        "replica_label": self._format_replica_label(device_name, pod_name),
                        "queue_length": self._extract_queue_length(resource, device_name, service_name),
                    }
                )
            result[service_name] = records

        return result
