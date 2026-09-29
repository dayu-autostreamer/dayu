import math
from collections.abc import Mapping

from .base_visualizer import BaseVisualizer


class BaseGanttVisualizer(BaseVisualizer):
    """Build the task interval payload consumed by the frontend Gantt template."""

    @staticmethod
    def _normalize_names(values):
        if isinstance(values, str):
            values = [values]
        elif isinstance(values, set):
            values = sorted(values, key=str)
        elif not isinstance(values, (list, tuple)):
            return []

        names = []
        for value in values:
            name = str(value).strip()
            if name and name not in names:
                names.append(name)
        return names

    @staticmethod
    def _finite_timestamp(value):
        if isinstance(value, bool):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @classmethod
    def _build_segment(cls, task, service_name, lane):
        service = task.get_service(service_name)
        tmp_data = service.get_tmp_data()
        if not isinstance(tmp_data, Mapping):
            return None

        start_time = cls._finite_timestamp(tmp_data.get('execute_start'))
        end_time = cls._finite_timestamp(tmp_data.get('execute_end'))
        if start_time is None or end_time is None or end_time < start_time:
            return None

        return {
            'task_id': task.get_task_id(),
            'lane': lane,
            'service': service_name,
            'device': service.get_execute_device(),
            'start_time': start_time,
            'end_time': end_time,
        }

    def _wrap_payload(self, lanes, segments):
        if not self.variables:
            return {}
        return {
            self.variables[0]: {
                'lanes': list(lanes),
                'segments': list(segments),
            }
        }
