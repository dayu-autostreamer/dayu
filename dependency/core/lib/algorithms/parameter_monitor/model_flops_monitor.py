import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request
from .base_monitor import BaseMonitor

__all__ = ('ModelFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='model_flops')
class ModelFlopsMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'model_flops'

        self.local_device = system.local_device

    def get_model_flops(self):
        model_flops_dict = {}
        for endpoint in self.system.runtime_routes(component='processor', target_node=self.local_device):
            service = endpoint.logical_service
            processor_address = endpoint.url(NetworkAPIPath.PROCESSOR_MODEL_FLOPS)
            response = http_request(
                processor_address,
                method=NetworkAPIMethod.PROCESSOR_MODEL_FLOPS,
                timeout=2,
            )
            model_flops_dict[service] = response / 1e9 if response else 0
        return model_flops_dict

    def get_parameter_value(self):
        return self.get_model_flops()
