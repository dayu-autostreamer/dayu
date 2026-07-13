import abc
from core.lib.common import ClassFactory, ClassType, TaskConstant
from core.lib.content import Task

from .topology_visualizer import TopologyVisualizer

__all__ = ('DAGOffloadingTopologyVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='dag_offloading')
class DAGOffloadingTopologyVisualizer(TopologyVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        result = task.get_dag_deployment_info()
        for service_name, node_info in result.items():
            service = node_info["service"]
            if service_name == TaskConstant.START.value:
                service["execute_device"] = task.get_source_device()
            service["data"] = service.pop("execute_device")

        return {self.variables[0]: result}
