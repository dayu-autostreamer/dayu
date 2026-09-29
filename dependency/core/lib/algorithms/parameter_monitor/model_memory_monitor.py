import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request
from .base_monitor import BaseMonitor

__all__ = ('ModelMemoryMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='model_memory')
class ModelMemoryMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'model_memory'

        self.local_device = system.local_device
        self._service_memory_gb_max = {}

    def get_processor_memory(self):
        """Query RSS directly from each exact local processor endpoint."""
        processor_memory_dict = {}
        for endpoint in self.system.runtime_routes(component='processor', target_node=self.local_device):
            service = endpoint.logical_service
            processor_address = endpoint.url(NetworkAPIPath.PROCESSOR_MODEL_MEMORY)
            response = http_request(
                processor_address,
                method=NetworkAPIMethod.PROCESSOR_MODEL_MEMORY,
                timeout=2,
            )
            if response:
                processor_memory_dict[service] = float(response) / 1e9
        return processor_memory_dict

    def get_model_memory(self):
        service_memory_dict = {}
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
