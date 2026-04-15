import abc

from core.lib.common import ClassFactory, ClassType, KubeConfig, ServiceConfig
from core.lib.network import NodeInfo
from .base_monitor import BaseMonitor

__all__ = ('ModelMemoryMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='model_memory')
class ModelMemoryMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'model_memory'

        self.local_device = NodeInfo.get_local_device()

    def get_model_memory(self):
        KubeConfig.force_refresh()
        pods_list = KubeConfig.get_pods_on_node(self.local_device)
        try:
            pod_memory_from_spec = KubeConfig.get_pod_memory_from_spec(pods_list)
        except Exception:
            pod_memory_from_spec = {}

        try:
            pod_memory_from_metrics = KubeConfig.get_pod_memory_from_metrics(pods_list)
        except Exception:
            pod_memory_from_metrics = {}

        service_memory_dict = {}
        for pod in set(pod_memory_from_spec) | set(pod_memory_from_metrics):
            service_name = ServiceConfig.map_pod_name_to_service(pod)
            if service_name is None:
                continue

            # Prefer the stable pod-spec request/limit value and only fall back
            # to live metrics when the spec does not expose a memory setting.
            memory_bytes = pod_memory_from_spec.get(pod)
            if memory_bytes is None:
                memory_bytes = pod_memory_from_metrics.get(pod, 0)

            memory_gb = float(memory_bytes) / 1e9
            service_memory_dict[service_name] = max(service_memory_dict.get(service_name, 0.0), memory_gb)
        return service_memory_dict

    def get_parameter_value(self):
        return self.get_model_memory()
