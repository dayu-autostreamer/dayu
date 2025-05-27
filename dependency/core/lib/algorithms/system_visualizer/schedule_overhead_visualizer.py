import abc
from core.lib.common import ClassFactory, ClassType, SystemConstant
from core.lib.network import http_request, NodeInfo, PortInfo, NetworkAPIPath, merge_address, NetworkAPIMethod

from .curve_visualizer import CurveVisualizer

__all__ = ('ScheduleOverheadVisualizer',)


@ClassFactory.register(ClassType.SYSTEM_VISUALIZER, alias='schedule_overhead')
class ScheduleOverheadVisualizer(CurveVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.resource_url = None

    def request_scheduling_overhead(self):
        self.get_scheduling_overhead_url()

        return http_request(self.resource_url,
                            method=NetworkAPIMethod.SCHEDULER_OVERHEAD) if self.resource_url else None

    def get_scheduling_overhead_url(self):
        cloud_hostname = NodeInfo.get_cloud_node()
        try:
            scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        except AssertionError:
            return
        self.resource_url = merge_address(NodeInfo.hostname2ip(cloud_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_OVERHEAD)

    def __call__(self):
        overhead = self.request_scheduling_overhead()

        return overhead * 1000 if overhead else 0
