import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, NodeInfo, PortInfo, merge_address, http_request
from .base_monitor import BaseMonitor

__all__ = ('QueueLengthMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='queue_length')
class QueueLengthMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'queue_length'

        self.local_device = NodeInfo.get_local_device()

    def get_queue_length(self):
        queue_length_dict = {}
        service_ports_dict = PortInfo.get_service_ports_dict(self.local_device)
        for service, port in service_ports_dict.items():
            processor_address = merge_address(NodeInfo.hostname2ip(self.local_device),
                                              port=port,
                                              path=NetworkAPIPath.PROCESSOR_QUEUE_LENGTH)
            response = http_request(processor_address, method=NetworkAPIMethod.PROCESSOR_QUEUE_LENGTH)
            queue_length_dict[service] = response if response else 0
        return queue_length_dict

    def get_parameter_value(self):
        return self.get_queue_length()
