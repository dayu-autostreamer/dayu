import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, NodeInfo, PortInfo, merge_address, http_request
from .base_monitor import BaseMonitor

__all__ = ('ModelFlopsMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='model_flops')
class ModelFlopsMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'model_flops'

        self.local_device = NodeInfo.get_local_device()

    def get_model_flops(self):
        model_flops_dict = {}
        service_ports_dict = PortInfo.get_service_ports_dict(self.local_device)
        for service, port in service_ports_dict.items():
            processor_address = merge_address(NodeInfo.hostname2ip(self.local_device),
                                              port=port,
                                              path=NetworkAPIPath.PROCESSOR_MODEL_FLOPS)
            response = http_request(processor_address, method=NetworkAPIMethod.PROCESSOR_MODEL_FLOPS)
            model_flops_dict[service] = response / 1e9 if response else 0
        return model_flops_dict

    def get_parameter_value(self):
        return self.get_model_flops()
