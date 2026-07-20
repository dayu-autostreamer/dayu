import copy
import json
import threading
import time

from core.lib.common import LOGGER, Context
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request
from core.lib.runtime import RuntimeContext, RuntimeEndpoint


class Monitor:
    _RUNTIME_REQUEST_TIMEOUT_SECONDS = 2.0

    def __init__(self):

        self.resource_info = {}

        self.monitor_interval = Context.get_parameter('INTERVAL', direct=False)
        if (
            isinstance(self.monitor_interval, bool)
            or not isinstance(self.monitor_interval, (int, float))
            or self.monitor_interval <= 0
        ):
            raise ValueError('INTERVAL must be a positive number')
        self.monitor_interval = float(self.monitor_interval)
        self._next_monitor_deadline = time.monotonic() + self.monitor_interval

        self.runtime_context = RuntimeContext.get_default()
        self.scheduler_endpoint = self.runtime_context.resolve_static_endpoint('scheduler')
        self.scheduler_address = self.scheduler_endpoint.url(NetworkAPIPath.SCHEDULER_POST_RESOURCE)
        self.local_device = self.runtime_context.local_node
        self._directory_lock = threading.Lock()
        self._directory = {}
        self._directory_fetched_at = 0.0
        self._monitor_cycle_directory = None

        monitor_parameters_text = Context.get_parameter('MONITORS', direct=False)
        if not isinstance(monitor_parameters_text, (list, tuple)) or not all(
            isinstance(name, str) and name.strip() for name in monitor_parameters_text
        ):
            raise ValueError('MONITORS must be a list of non-empty monitor hook names')
        if len(set(monitor_parameters_text)) != len(monitor_parameters_text):
            raise ValueError('MONITORS must not contain duplicate hook names')
        self.monitor_parameters = []
        for mp_text in monitor_parameters_text:
            self.monitor_parameters.append(
                Context.get_algorithm('MON_PRAM', mp_text, system=self)
            )

    def runtime_routes(self, component=None, target_node=None, logical_service=None):
        directory = (
            copy.deepcopy(self._monitor_cycle_directory)
            if self._monitor_cycle_directory is not None
            else self._runtime_directory_snapshot()
        )
        routes = directory.get('routes') or []
        endpoints = [RuntimeEndpoint.from_value(route) for route in routes]
        matches = [
            endpoint for endpoint in endpoints
            if endpoint.matches(component, target_node, logical_service)
        ]
        for endpoint in matches:
            if endpoint.component in {'controller', 'processor'}:
                endpoint.validate_exact()
        return matches

    def _runtime_directory_snapshot(self):
        with self._directory_lock:
            now = time.time()
            if now - self._directory_fetched_at >= self.monitor_interval:
                directory = http_request(
                    self.scheduler_endpoint.url(NetworkAPIPath.SCHEDULER_RUNTIME_DIRECTORY),
                    method=NetworkAPIMethod.SCHEDULER_GET_RUNTIME_DIRECTORY,
                    timeout=self._RUNTIME_REQUEST_TIMEOUT_SECONDS,
                )
                self._directory = directory if isinstance(directory, dict) else {}
                self._directory_fetched_at = now
            return copy.deepcopy(self._directory or {})

    def runtime_directory_revision(self):
        directory = (
            self._monitor_cycle_directory
            if self._monitor_cycle_directory is not None
            else self._runtime_directory_snapshot()
        )
        try:
            revision = int(directory.get('revision') or 0)
        except (TypeError, ValueError):
            return 0
        return revision if revision > 0 else 0

    def monitor_resource(self):
        # Queue hooks and the revision attached to their resource payload must
        # come from the same immutable directory snapshot, even when polling
        # takes longer than one monitor interval.
        self._monitor_cycle_directory = self._runtime_directory_snapshot()
        threads = [mp() for mp in self.monitor_parameters]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def wait_for_monitor(self):
        current_ts = time.monotonic()
        wait_time = self._next_monitor_deadline - current_ts
        if wait_time > 0:
            LOGGER.debug(f'[Monitor Interval] Waiting {wait_time} seconds for next monitor cycle.')
            time.sleep(wait_time)
        current_ts = time.monotonic()
        elapsed_intervals = max(
            1,
            int((current_ts - self._next_monitor_deadline) // self.monitor_interval) + 1,
        )
        self._next_monitor_deadline += elapsed_intervals * self.monitor_interval

    def send_resource_state_to_scheduler(self):

        LOGGER.info(f'[Monitor Resource] info: {self.resource_info}')

        data = {'device': self.local_device, 'resource': self.resource_info}
        revision = self.runtime_directory_revision()
        if revision:
            data['runtime_directory_revision'] = revision

        try:
            http_request(self.scheduler_address,
                         method=NetworkAPIMethod.SCHEDULER_POST_RESOURCE,
                         timeout=self._RUNTIME_REQUEST_TIMEOUT_SECONDS,
                         data={'data': json.dumps(data)})
        finally:
            self._monitor_cycle_directory = None
