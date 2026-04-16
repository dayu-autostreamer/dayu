import abc

from core.lib.common import ClassFactory, ClassType, KubeConfig, ServiceConfig
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, NodeInfo, PortInfo, merge_address, http_request
from .base_monitor import BaseMonitor

__all__ = ('ModelMemoryMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='model_memory')
class ModelMemoryMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'model_memory'

        self.local_device = NodeInfo.get_local_device()
        self._service_memory_gb_max = {}

    def get_processor_memory(self):
        """Query local processor RSS as a fallback when Kubernetes memory is missing."""
        processor_memory_dict = {}
        service_ports_dict = PortInfo.get_service_ports_dict(self.local_device)
        for service, port in service_ports_dict.items():
            processor_address = merge_address(
                NodeInfo.hostname2ip(self.local_device),
                port=port,
                path=NetworkAPIPath.PROCESSOR_MODEL_MEMORY,
            )
            response = http_request(
                processor_address,
                method=NetworkAPIMethod.PROCESSOR_MODEL_MEMORY,
                timeout=2,
            )
            if response:
                processor_memory_dict[service] = float(response) / 1e9
        return processor_memory_dict

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

            # `model_memory` is consumed as a conservative placement footprint.
            # Pod specs can understate actual RSS when requests are small, while
            # live metrics may be unavailable during startup. Use the larger
            # available value and keep a running maximum per service to avoid
            # treating transient low-RSS samples as a smaller static model size.
            memory_candidates = [
                value for value in (
                    pod_memory_from_spec.get(pod),
                    pod_memory_from_metrics.get(pod),
                )
                if value is not None
            ]
            if not memory_candidates:
                continue

            memory_gb = float(max(memory_candidates)) / 1e9
            service_memory_dict[service_name] = max(service_memory_dict.get(service_name, 0.0), memory_gb)

        try:
            processor_memory_dict = self.get_processor_memory()
        except Exception:
            processor_memory_dict = {}

        for service_name, memory_gb in processor_memory_dict.items():
            service_memory_dict[service_name] = max(
                service_memory_dict.get(service_name, 0.0),
                float(memory_gb),
            )

        for service_name, memory_gb in service_memory_dict.items():
            self._service_memory_gb_max[service_name] = max(
                self._service_memory_gb_max.get(service_name, 0.0),
                float(memory_gb),
            )
        return dict(self._service_memory_gb_max)

    def get_parameter_value(self):
        return self.get_model_memory()
