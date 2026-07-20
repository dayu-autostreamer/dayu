import abc

from core.lib.common import ClassFactory, ClassType
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request
from .base_monitor import BaseMonitor

__all__ = ('QueueStateMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='queue_state')
class QueueStateMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'queue_state'
        self.local_device = system.local_device

    @staticmethod
    def _non_negative_number(value, default=0.0):
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return default

    @classmethod
    def _normalize_queue_state(cls, response):
        state = dict(response) if isinstance(response, dict) else {}
        try:
            state['waiting_count'] = max(0, int(state.get('waiting_count', 0)))
        except (TypeError, ValueError):
            state['waiting_count'] = 0
        state['busy'] = bool(state.get('busy', False))
        state['running_elapsed_s'] = cls._non_negative_number(state.get('running_elapsed_s'))
        try:
            state['capacity'] = max(1, int(state.get('capacity', 1) or 1))
        except (TypeError, ValueError):
            state['capacity'] = 1
        try:
            state['sequence'] = max(0, int(state.get('sequence', 0)))
        except (TypeError, ValueError):
            state['sequence'] = 0
        state.setdefault('running_task', None)
        state.setdefault('observed_at', None)
        return state

    def get_parameter_value(self):
        queue_states = {}
        for endpoint in self.system.runtime_routes(component='processor', target_node=self.local_device):
            response = http_request(
                endpoint.url(NetworkAPIPath.PROCESSOR_QUEUE_STATE),
                method=NetworkAPIMethod.PROCESSOR_QUEUE_STATE,
                timeout=2,
            )
            queue_states[endpoint.logical_service] = self._normalize_queue_state(response)
        return queue_states
