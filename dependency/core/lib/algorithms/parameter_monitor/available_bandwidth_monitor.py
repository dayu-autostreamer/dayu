import abc
import threading
import json
from func_timeout import func_set_timeout as timeout

from .base_monitor import BaseMonitor

from core.lib.common import ClassFactory, ClassType, LOGGER, Context, SystemConstant, NodeRoleConstant
from core.lib.network import NodeInfo, PortInfo, merge_address, NetworkAPIPath, NetworkAPIMethod, http_request

__all__ = ('AvailableBandwidthMonitor',)


@ClassFactory.register(ClassType.MON_PRAM, alias='available_bandwidth')
class AvailableBandwidthMonitor(BaseMonitor, abc.ABC):
    def __init__(self, system):
        super().__init__(system)
        self.name = 'available_bandwidth'

        self.local_device = NodeInfo.get_local_device()
        self.permitted_device = ''

        self.is_server = NodeInfo.get_node_role(NodeInfo.get_local_device()) == NodeRoleConstant.CLOUD.value
        if self.is_server:
            self.iperf3_ports = [Context.get_parameter('GUNICORN_PORT')]
            self.run_iperf_server()
        else:
            self.iperf3_port = PortInfo.get_component_port(SystemConstant.MONITOR.value)
            self.iperf3_server_ip = NodeInfo.hostname2ip(NodeInfo.get_cloud_node())
            self.request_for_bandwidth_permission()

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

    def request_for_bandwidth_permission(self):
        scheduler_hostname = NodeInfo.get_cloud_node()
        scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        scheduler_address = merge_address(NodeInfo.hostname2ip(scheduler_hostname),
                                          port=scheduler_port,
                                          path=NetworkAPIPath.SCHEDULER_GET_RESOURCE_LOCK)
        response = http_request(scheduler_address,
                                method=NetworkAPIMethod.SCHEDULER_GET_RESOURCE_LOCK,
                                data={'data': json.dumps(
                                    {'resource': 'available_bandwidth', 'device': self.local_device})})
        if not response:
            self.permitted_device = ''
        else:
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

        @timeout(2)
        def fetch_bandwidth_by_iperf3():
            result = client.run()
            return result

        client = iperf3.Client()
        client.duration = 1
        client.server_hostname = self.iperf3_server_ip
        client.port = self.iperf3_port
        client.protocol = 'tcp'

        try:
            result_info = fetch_bandwidth_by_iperf3()

            del client

            if result_info.error:
                LOGGER.warning(f'resource monitor iperf3 error: {result_info.error}')
                bandwidth_result = 0
            else:
                bandwidth_result = result_info.sent_Mbps

        except Exception as e:
            LOGGER.exception(f'[Iperf3 Error] {e}')
            bandwidth_result = 0

        return bandwidth_result
