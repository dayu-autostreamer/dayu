import abc

from core.lib.common import ClassFactory, ClassType, KubeConfig, ServiceConfig, SystemConstant
from core.lib.content import Task
from core.lib.network import (
    http_request,
    NodeInfo,
    PortInfo,
    NetworkAPIPath,
    NetworkAPIMethod,
    merge_address,
)

from .base_visualizer import BaseVisualizer

__all__ = ("ServiceQueueLengthVisualizer",)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias="service_queue_length")
class ServiceQueueLengthVisualizer(BaseVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resource_url = None

    def get_resource_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        except AssertionError:
            self.resource_url = None
            return

        self.resource_url = merge_address(
            NodeInfo.hostname2ip(cloud_hostname),
            port=scheduler_port,
            path=NetworkAPIPath.SCHEDULER_GET_RESOURCE,
        )

    def request_resource_info(self):
        self.get_resource_url()
        return (
            http_request(self.resource_url, method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE)
            if self.resource_url
            else None
        )

    @staticmethod
    def _extract_queue_length(resource, device_name, service_name):
        if not isinstance(resource, dict):
            return 0

        device_resource = resource.get(device_name)
        if not isinstance(device_resource, dict):
            return 0

        queue_lengths = device_resource.get("queue_length")
        if not isinstance(queue_lengths, dict):
            return 0

        value = queue_lengths.get(service_name, 0)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _format_replica_label(device_name, pod_name):
        if not pod_name:
            return device_name or "unknown-replica"
        if len(pod_name) <= 36:
            return f"{device_name}/{pod_name}" if device_name else pod_name
        return f"{device_name}/{pod_name[:16]}...{pod_name[-12:]}" if device_name else pod_name

    @staticmethod
    def _list_service_replicas(service_name):
        try:
            KubeConfig.force_refresh()
        except Exception:
            return []

        replicas = []
        seen = set()
        for node_name in KubeConfig.get_nodes_for_service(service_name):
            for pod_name in KubeConfig.get_pods_on_node(node_name):
                if ServiceConfig.map_pod_name_to_service(pod_name) != service_name:
                    continue
                key = (node_name, pod_name)
                if key in seen:
                    continue
                seen.add(key)
                replicas.append(
                    {
                        "device": node_name,
                        "pod_name": pod_name,
                    }
                )

        replicas.sort(key=lambda item: (item.get("device", ""), item.get("pod_name", "")))
        return replicas

    def __call__(self, task: Task):
        _ = task
        resource = self.request_resource_info()
        result = {}

        for service_name in self.variables:
            records = []
            for replica in self._list_service_replicas(service_name):
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
