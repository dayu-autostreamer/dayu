import abc
from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task
from core.lib.runtime import RuntimeResolver

from .topology_visualizer import TopologyVisualizer

__all__ = ('DAGDeploymentTopologyVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='dag_deployment')
class DAGDeploymentTopologyVisualizer(TopologyVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        result = task.get_dag_deployment_info()
        for node_info in result.values():
            service = node_info["service"]
            service.pop("execute_device")
            service_name = service["service_name"]
            routes = RuntimeResolver.list_routes(task, component='processor', logical_service=service_name)
            service["data"] = '\n'.join(sorted({route.target_node for route in routes}))

        return {self.variables[0]: result}
