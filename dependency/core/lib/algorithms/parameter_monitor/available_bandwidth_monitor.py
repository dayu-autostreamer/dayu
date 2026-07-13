import abc
import threading
import json
from func_timeout import func_set_timeout as timeout

from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType, LOGGER, Context, NodeRoleConstant
from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request

__all__ = ('AvailableBandwidthMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='available_bandwidth')
class AvailableBandwidthMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'available_bandwidth'

        self.local_device = system.local_device
        self.permitted_device = ''

        context = system.runtime_context
        self.is_server = (
            self.local_device == context.cloud_node
            or context.node_role(self.local_device) == NodeRoleConstant.CLOUD.value
        )
        if self.is_server:
            self.iperf3_ports = [Context.get_parameter('GUNICORN_PORT')]
            self.run_iperf_server()
        else:
            monitor_endpoint = context.resolve_static_endpoint('monitor', target_node=context.cloud_node)
            self.iperf3_port = monitor_endpoint.port
            self.iperf3_server_ip = monitor_endpoint.fqdn
            try:
                self.request_for_bandwidth_permission()
            except Exception as e:
                LOGGER.warning(f'[Request Permission] Request bandwidth resource permission failed: {e}')
                LOGGER.exception(e)

    def run_iperf_server(self):
        for port in self.iperf3_ports:
            threading.Thread(target=self.iperf_server, args=(port,)).start()

    @staticmethod
    def iperf_server(port):
        import iperf3
        server = iperf3.Server()
        server.port = port
        LOGGER.debug(f'[Iperf3 Server] Running iperf3 server: {server.bind_address}:{server.port}')

        while True:
            try:
                result = server.run()
            except Exception as e:
                LOGGER.exception(e)
                continue

            if result.error:
                LOGGER.warning(result.error)

    @timeout(60)
    def request_for_bandwidth_permission(self):
        scheduler_address = self.system.scheduler_endpoint.url(NetworkAPIPath.SCHEDULER_GET_RESOURCE_LOCK)
        response = None
        while not response:
            response = http_request(scheduler_address,
                                    method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE_LOCK,
                                    timeout=2,
                                    data={'data': json.dumps(
                                        {'resource': 'available_bandwidth', 'device': self.local_device})})

        self.permitted_device = response['holder']

    def get_parameter_value(self):
        import iperf3
        if self.is_server:
            LOGGER.debug(f'Current device is the server ({self.local_device}), skip available bandwidth monitor..')
            return -1
        if self.local_device != self.permitted_device:
            LOGGER.debug(f'Current device is not the permitted device (current:{self.local_device},'
                         f' permitted:{self.permitted_device}), skip available bandwidth monitor..')
            return -1

        client = iperf3.Client()
        client.duration = 1
        client.server_hostname = self.iperf3_server_ip
        client.port = self.iperf3_port
        client.protocol = 'tcp'

        @timeout(2)
        def fetch_bandwidth_by_iperf3():
            return client.run()

        try:
            result_info = fetch_bandwidth_by_iperf3()

            if result_info.error:
                LOGGER.warning(f'resource monitor iperf3 error: {result_info.error}')
                bandwidth_result = 0
            else:
                bandwidth_result = result_info.sent_Mbps

        except Exception as e:
            LOGGER.exception(f'[Iperf3 Error] {e}')
            bandwidth_result = 0

        return bandwidth_result
