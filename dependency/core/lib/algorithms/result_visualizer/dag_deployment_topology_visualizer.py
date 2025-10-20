import abc
from core.lib.common import ClassFactory, ClassType, KubeConfig, LOGGER
from core.lib.content import Task

from .topology_visualizer import TopologyVisualizer

__all__ = ('DAGDeploymentTopologyVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='dag_deployment')
class DAGDeploymentTopologyVisualizer(TopologyVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):
        LOGGER.debug('[DEBUG] start dag deployment topology visualizer')
        result = task.get_dag_deployment_info()
        LOGGER.debug('[DEBUG] got dag deployment info')
        LOGGER.debug(f'[DEBUG] result: {result}')
        for node_info in result.values():
            LOGGER.debug(f'[DEBUG] start node_info: {node_info["service"]["service_name"]}')
            service = node_info["service"]
            LOGGER.debug(f'[DEBUG] got service')
            service.pop("execute_device")
            LOGGER.debug(f'[DEBUG] pop execute device')
            service_name = service["service_name"]
            LOGGER.debug(f'[DEBUG] got service name')
            service["data"] = '\n'.join(KubeConfig.get_nodes_for_service(service_name))
            LOGGER.debug(f'[DEBUG] got service data')
            LOGGER.debug(f'[DEBUG] end node_info: {node_info["service"]["service_name"]}')
        LOGGER.debug('[DEBUG] end dag deployment topology visualizer')

        return {'topology': result}
