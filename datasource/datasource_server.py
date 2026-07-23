import os
import time
import re

from core.lib.network import NetworkAPIPath, NetworkAPIMethod, http_request
from core.lib.common import LOGGER, Context
from core.lib.runtime import RuntimeContext, RuntimeResolver
from script_helper import ScriptHelper


class DataSource:
    def __init__(self):

        self.source_label = ''
        self.source_open = False

        self.process_list = []

        self.runtime_context = RuntimeContext.get_default()
        self.runtime_resolver = RuntimeResolver(self.runtime_context)
        self.backend_address = self.runtime_resolver.resolve_url(
            "backend",
            path=NetworkAPIPath.BACKEND_DATASOURCE_STATE,
            target_node=self.runtime_context.cloud_node or None,
        )

        self.inner_port = Context.get_parameter('GUNICORN_PORT')

        self.request_interval = Context.get_parameter('REQUEST_INTERVAL', direct=False)
        self.start_interval = Context.get_parameter('START_INTERVAL', direct=False)

        self.play_mode = Context.get_parameter('PLAY_MODE')

        if self.play_mode not in ['cycle', 'non-cycle']:
            raise ValueError(f'play_mode must be cycle or non-cycle, given {self.play_mode}')
        LOGGER.info(f'Play Mode: {self.play_mode}')

    def open_datasource(self, modal, label, mode, source_list):
        if self.source_open:
            if self.source_label == label:
                return
            self.close_datasource()

        if not os.path.exists(f'{mode}.py'):
            LOGGER.warning(f'Datasource Mode of "{mode}" does not exist. ({mode}.py)')
            return

        LOGGER.info(f'Open Datasource: {modal}/{label}..')

        for index, source in enumerate(source_list):
            datasource_dir = os.path.join(Context.get_file_path(modal), source['dir'], mode)
            if not os.path.exists(datasource_dir):
                LOGGER.warning(f'Datasource directory "{datasource_dir}" does not exist.')
                return
            url = re.sub(r'(?<=:)\d+', str(self.inner_port), source['url'])
            url = re.sub(r'\d+\.\d+\.\d+\.\d+', '127.0.0.1', url)
            command = (f'python3 {mode}.py '
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
