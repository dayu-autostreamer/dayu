from collections.abc import Mapping

from core.lib.common import ClassFactory, ClassType, TaskConstant
from core.lib.content import Task

from .gantt_visualizer_base import BaseGanttVisualizer

__all__ = ('ServiceDeviceGanttVisualizer',)


@ClassFactory.register(ClassType.RESULT_VISUALIZER, alias='service_device_gantt')
class ServiceDeviceGanttVisualizer(BaseGanttVisualizer):
    """Collect one service's execute intervals grouped by deployed device."""

    def __init__(self, service=None, **kwargs):
        super().__init__(**kwargs)
        self.service = str(service).strip() if service is not None else ''

    def _get_deployed_devices(self, task):
        deployment = task.get_deployment()
        if not isinstance(deployment, Mapping) or not self.service:
            return []

        # Current runtime snapshots use {service: [devices]}. Compare keys and
        # values with the DAG to disambiguate older {device: [services]}
        # snapshots when a device happens to have the same name as a service.
        dag = task.get_dag()
        dag_services = set(dag.nodes) if dag is not None else set()
        current_shape_score = sum(str(key).strip() in dag_services for key in deployment)
        reverse_shape_score = sum(
            service_name in dag_services
            for services in deployment.values()
            for service_name in self._normalize_names(services)
        )
        if self.service in deployment and current_shape_score >= reverse_shape_score:
            return self._normalize_names(deployment.get(self.service))

        devices = []
        for device, services in deployment.items():
            if self.service in self._normalize_names(services):
                device_name = str(device).strip()
                if device_name and device_name not in devices:
                    devices.append(device_name)
        return devices

    def __call__(self, task: Task):
        dag = task.get_dag()
        excluded_services = {TaskConstant.START.value, TaskConstant.END.value}
        if (
            not self.service
            or dag is None
            or self.service not in dag.nodes
            or self.service in excluded_services
        ):
            return self._wrap_payload([], [])

        service = task.get_service(self.service)
        execute_device = str(service.get_execute_device() or '').strip()
        lanes = self._get_deployed_devices(task)
        if execute_device and execute_device not in lanes:
            lanes.append(execute_device)

        segments = []
        if execute_device:
            segment = self._build_segment(task, self.service, execute_device)
            if segment is not None:
                segments.append(segment)

        return self._wrap_payload(lanes, segments)
