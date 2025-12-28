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
        pod_memory_dict = KubeConfig.get_pod_memory_from_metrics(pods_list)

        service_memory_dict = {
            svc: memory
            for pod, memory in pod_memory_dict.items()
            if (svc := ServiceConfig.map_pod_name_to_service(pod)) is not None
        }
        return service_memory_dict

    def get_parameter_value(self):
        return self.get_model_memory() / 1e9
