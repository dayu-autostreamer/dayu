import json
import time

from core.lib.common import LOGGER, Context, SystemConstant
from core.lib.network import NodeInfo, PortInfo, merge_address, NetworkAPIPath, NetworkAPIMethod, http_request


class Monitor:
    def __init__(self):

        self.resource_info = {}

        self.monitor_interval = Context.get_parameter('INTERVAL', direct=False)
        self.last_monitor_ts = time.time()

        self.scheduler_hostname = NodeInfo.get_cloud_node()
        self.scheduler_port = PortInfo.get_component_port(SystemConstant.SCHEDULER.value)
        self.scheduler_address = merge_address(NodeInfo.hostname2ip(self.scheduler_hostname),
                                               port=self.scheduler_port,
                                               path=NetworkAPIPath.SCHEDULER_POST_RESOURCE)

        self.local_device = NodeInfo.get_local_device()

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
        current_ts = time.time()
        if current_ts - self.last_monitor_ts < self.monitor_interval:
            time.sleep(self.monitor_interval - (current_ts - self.last_monitor_ts))
        self.last_monitor_ts = current_ts

    def send_resource_state_to_scheduler(self):

        LOGGER.info(f'[Monitor Resource] info: {self.resource_info}')

        data = {'device': self.local_device, 'resource': self.resource_info}

        http_request(self.scheduler_address,
                     method=NetworkAPIMethod.SCHEDULER_POST_RESOURCE,
                     data={'data': json.dumps(data)})
