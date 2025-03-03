import json
import time

from core.lib.common import LOGGER, Context, ClassFactory, ClassType, SystemConstant, NodeRoleConstant
from core.lib.network import NodeInfo, PortInfo, merge_address, NetworkAPIPath, NetworkAPIMethod, http_request


class Monitor:
    def __init__(self):

        self.resource_info = {}

        self.monitor_interval = Context.get_parameter('INTERVAL', direct=False)

        self.scheduler_hostname = NodeInfo.get_cloud_node()
        self.scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.scheduler_address = merge_address(NodeInfo.hostname2ip(self.scheduler_hostname),
                                               port=self.scheduler_port,
                                               path=NetworkAPIPath.SCHEDULER_POST_RESOURCE)

        self.local_device = NodeInfo.get_local_device()
        self.is_iperf3_server = NodeInfo.get_node_role(NodeInfo.get_local_device()) == NodeRoleConstant.CLOUD.value
        self.iperf3_server_ip = NodeInfo.hostname2ip(NodeInfo.get_cloud_node())

        self.iperf3_port = PortInfo.get_component_port(SystemConstant.MONITOR.value)
        self.iperf3_ports = [Context.get_parameter('GUNICORN_PORT')]

        monitor_parameters_text = Context.get_parameter('MONITORS', direct=False)
        self.monitor_parameters = []
        for mp_text in monitor_parameters_text:
            self.monitor_parameters.append(
                Context.get_algorithm('MON_PRAM', mp_text, system=self)
            )

    def monitor_resource(self):
        threads = [mp() for mp in self.monitor_parameters]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def wait_for_monitor(self):
        time.sleep(self.monitor_interval)

    def send_resource_state_to_scheduler(self):

        LOGGER.info(f'[Monitor Resource] info: {self.resource_info}')

        data = {'device': self.local_device, 'resource': self.resource_info}

        http_request(self.scheduler_address,
                     method=NetworkAPIMethod.SCHEDULER_POST_RESOURCE,
                     data={'data': json.dumps(data)})
