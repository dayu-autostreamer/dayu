import abc
from core.lib.common import ClassFactory, ClassType, SystemConstant
from core.lib.network import http_request, NodeInfo, PortInfo, NetworkAPIPath, merge_address, NetworkAPIMethod

from .curve_visualizer import CurveVisualizer

__all__ = ('MemoryUsageVisualizer',)


@ClassFactory.register(ClassType.SYSTEM_VISUALIZER, alias='memory_usage')
class MemoryUsageVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.resource_url = None

    def request_resource_info(self):
        self.get_resource_url()

        return http_request(self.resource_url, method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE) if self.resource_url else None

    def get_resource_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        except AssertionError:
            return
        self.resource_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_GET_RESOURCE)

    def __call__(self):
        resource = self.request_resource_info()

        if self.variables:
            if not resource:
                return {device: 0 for device in self.variables}
            return {device: resource[device]['memory'] if device in resource else 0 for device in self.variables}

        else:
            if not resource:
                return {'no device':0}
            else:
                return {device:resource[device]['memory'] for device in resource}
