import os
import time
import re

from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request, NodeInfo, PortInfo, merge_address
from core.lib.common import LOGGER, Context, SystemConstant
from script_helper import ScriptHelper


class DataSource:
    def __init__(self):

        self.source_label = ''
        self.source_open = False

        self.process_list = []

        self.command_headers = {
            'rtsp_video': 'bash modal_source.sh',
            'http_video': 'python3 modal_source.py',
        }

        self.backend_hostname = NodeInfo.get_cloud_node()
        self.backend_port = PortInfo.get_component_port(SystemConstant.BACKEND.value)
        self.backend_address = merge_address(NodeInfo.hostname2ip(self.backend_hostname),
                                             port=self.backend_port,
                                             path=NetworkAPIPath.BACKEND_DATASOURCE_STATE)

        self.inner_port = Context.get_parameter('GUNICORN_PORT')

        self.request_interval = Context.get_parameter('REQUEST_INTERVAL', direct=False)
        self.start_interval = Context.get_parameter('START_INTERVAL', direct=False)

        self.play_mode = Context.get_parameter('PLAY_MODE')

        if self.play_mode not in ['cycle', 'non-cycle']:
            raise ValueError(f'play_mode must be cycle or non-cycle, given {self.play_mode}')
        LOGGER.info(f'Play Mode: {self.play_mode}')

    def open_datasource(self, modal, label, mode, source_list):
        if self.source_open:
            return

        if mode not in self.command_headers:
            LOGGER.warning(f'Datasource Mode of "{mode}" does not exist. (Only {self.command_headers.keys()})')
            return

        LOGGER.info(f'Open Datasource: {modal}/{label}..')

        for index, source in enumerate(source_list):
            datasource_dir = os.path.join(Context.get_file_path(modal), source['dir'], mode)
            if not os.path.exists(datasource_dir):
                LOGGER.warning(f'Datasource directory "{datasource_dir}" does not exist.')
                return
            url = re.sub(r'(?<=:)\d+', str(self.inner_port), source['url'])
            url = re.sub(r'\d+\.\d+\.\d+\.\d+', '0.0.0.0', url)
            command = (f'{self.command_headers[mode].replace("modal", modal)} '
                       f'--root {datasource_dir} --address {url} --play_mode {self.play_mode}')
            process = ScriptHelper.start_script(command)
            self.process_list.append(process)
            if index < len(source_list) - 1:
                time.sleep(self.start_interval)

        self.source_label = label
        self.source_open = True

    def close_datasource(self):
        if not self.source_open:
            return

        LOGGER.info('Close Datasource..')

        for process in self.process_list:
            ScriptHelper.stop_script(process)

        self.process_list = []
        self.source_label = ''
        self.source_open = False

    def run(self):
        while True:
            response = http_request(self.backend_address, method=NetworkAPIMethod.BACKEND_DATASOURCE_STATE)
            if response:
                if response['state'] == 'open':

                    self.open_datasource(modal=response['source_type'],
                                         label=response['source_label'],
                                         mode=response['source_mode'],
                                         source_list=response['source_list'])
                else:
                    self.close_datasource()
            else:
                self.close_datasource()

            time.sleep(self.request_interval)
