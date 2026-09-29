from core.lib.common import ClassFactory, ClassType, TaskConstant
from core.lib.content import Task

from .gantt_visualizer_base import BaseGanttVisualizer

__all__ = ('ServiceGanttVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='service_gantt')
class ServiceGanttVisualizer(BaseGanttVisualizer):
    """Collect execute intervals grouped by DAG service."""

    def __init__(self, services=None, **kwargs):
        super().__init__(**kwargs)
        self.services = None if services is None else self._normalize_names(services)

    def __call__(self, task: Task):
        dag = task.get_dag()
        if dag is None:
            return self._wrap_payload([], [])

        excluded_services = {TaskConstant.START.value, TaskConstant.END.value}
        available_services = [
            service_name for service_name in dag.nodes
            if service_name not in excluded_services
        ]
        requested_services = available_services if self.services is None else self.services
        lanes = [
            service_name for service_name in requested_services
            if service_name in available_services
        ]

        segments = []
        for service_name in lanes:
            segment = self._build_segment(task, service_name, service_name)
            if segment is not None:
                segments.append(segment)

        return self._wrap_payload(lanes, segments)
