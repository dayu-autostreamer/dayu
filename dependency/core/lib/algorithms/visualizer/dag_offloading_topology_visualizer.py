import abc
from core.lib.common import ClassFactory, ClassType
from core.lib.content import Task
from core.lib.network import NodeInfo

from .topology_visualizer import TopologyVisualizer

__all__ = ('DAGOffloadingTopologyVisualizer',)


@ClassFactory.register(ClassType.VISUALIZER, alias='dag_offloading')
class DAGOffloadingTopologyVisualizer(TopologyVisualizer, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, task: Task):

        task.get_dag().get_start_node().service.set_execute_device(task.get_source_device())
        task.get_dag().get_end_node().service.set_execute_device(NodeInfo.get_cloud_node())
        result = task.get_dag_deployment_info()
        print(f'dag: {result}')
