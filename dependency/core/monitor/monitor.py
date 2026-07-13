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
        self.last_monitor_ts = time.time()

        self.runtime_context = RuntimeContext.get_default()
        self.scheduler_endpoint = self.runtime_context.resolve_static_endpoint('scheduler')
        self.scheduler_address = self.scheduler_endpoint.url(NetworkAPIPath.SCHEDULER_POST_RESOURCE)
        self.local_device = self.runtime_context.local_node
        self._directory_lock = threading.Lock()
        self._directory = {}
        self._directory_fetched_at = 0.0

        monitor_parameters_text = Context.get_parameter('MONITORS', direct=False)
        self.monitor_parameters = []
        for mp_text in monitor_parameters_text:
            self.monitor_parameters.append(
                Context.get_algorithm('MON_PRAM', mp_text, system=self)
            )

    def runtime_routes(self, component=None, target_node=None, logical_service=None):
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
            routes = (self._directory or {}).get('routes') or []
        endpoints = [RuntimeEndpoint.from_value(route) for route in routes]
        matches = [
            endpoint for endpoint in endpoints
            if endpoint.matches(component, target_node, logical_service)
        ]
        for endpoint in matches:
            if endpoint.component in {'controller', 'processor'}:
                endpoint.validate_exact()
        return matches

    def monitor_resource(self):
        threads = [mp() for mp in self.monitor_parameters]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def wait_for_monitor(self):
        current_ts = time.time()
        if current_ts - self.last_monitor_ts < self.monitor_interval:
            wait_time = self.monitor_interval - (current_ts - self.last_monitor_ts)
            LOGGER.debug(f'[Monitor Interval] Waiting {wait_time} seconds for next monitor cycle.')
            time.sleep(wait_time)
        self.last_monitor_ts = current_ts

    def send_resource_state_to_scheduler(self):

        LOGGER.info(f'[Monitor Resource] info: {self.resource_info}')

        data = {'device': self.local_device, 'resource': self.resource_info}

        http_request(self.scheduler_address,
                     method=NetworkAPIMethod.SCHEDULER_POST_RESOURCE,
                     timeout=self._RUNTIME_REQUEST_TIMEOUT_SECONDS,
                     data={'data': json.dumps(data)})
