import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request
from .base_monitor import BaseMonitor

__all__ = ('QueueLengthMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='queue_length')
class QueueLengthMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'queue_length'

        self.local_device = system.local_device

    def get_queue_length(self):
        queue_length_dict = {}
        for endpoint in self.system.runtime_routes(component='processor', target_node=self.local_device):
            service = endpoint.logical_service
            processor_address = endpoint.url(NetworkAPIPath.PROCESSOR_QUEUE_LENGTH)
            response = http_request(
                processor_address,
                method=NetworkAPIMethod.PROCESSOR_QUEUE_LENGTH,
                timeout=2,
            )
            queue_length_dict[service] = response if response else 0
        return queue_length_dict

    def get_parameter_value(self):
        return self.get_queue_length()
