import threading

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form

from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.lib.common import Context, SystemConstant
from core.lib.common import LOGGER, FileOps
from core.lib.network import NodeInfo, PortInfo, http_request, get_merge_address, NetworkAPIMethod, NetworkAPIPath
from core.lib.content import Task

from .processor import Processor


class ProcessorServer:
    def __init__(self):
        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.PROCESSOR_PROCESS,
                     self.process_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_PROCESS]
                     ),
        ], log_level='trace', timeout=6000)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        self.processor = Context.get_algorithm('PROCESSOR')

        self.task_queue = Context.get_algorithm('PRO_QUEUE')

        self.local_device = NodeInfo.get_local_device()
        self.processor_port = Context.get_parameter('GUNICORN_PORT')
        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.controller_address = get_merge_address(NodeInfo.hostname2ip(self.local_device),
                                                    port=self.controller_port,
                                                    path=NetworkAPIPath.CONTROLLER_RETURN)

        threading.Thread(target=self.loop_process).start()

    async def process_service(self, backtask: BackgroundTasks, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        cur_task = Task.deserialize(data)
        backtask.add_task(self.process_service_background, data, file_data)
        LOGGER.debug(f'[Monitor Task] (Process Request) '
                     f'Source: {cur_task.get_source_id()} / Task: {cur_task.get_task_id()} ')

    def process_service_background(self, data, file_data):
        cur_task = Task.deserialize(data)
        FileOps.save_data_file(cur_task, file_data)
        self.task_queue.put(cur_task)
        LOGGER.debug(f'[Task Queue] Queue Size (receive request): {self.task_queue.size()}')
        LOGGER.debug(f'[Monitor Task] (Process Request Background) '
                     f'Source: {cur_task.get_source_id()} / Task: {cur_task.get_task_id()} ')

    def loop_process(self):
        LOGGER.info('Start processing loop..')
        while True:
            if self.task_queue.empty():
                continue
            task = self.task_queue.get()
            if not task:
                continue

            LOGGER.debug(f'[Task Queue] Queue Size (loop): {self.task_queue.size()}')

            LOGGER.debug(f'[Monitor Task] (Process start) Source: {task.get_source_id()} / Task: {task.get_task_id()} ')
            new_task = self.processor(task)
            LOGGER.debug(f'[Monitor Task] (Process end) Source: {task.get_source_id()} / Task: {task.get_task_id()} ')
            if new_task:
                self.send_result_back_to_controller(new_task)
            FileOps.remove_data_file(task)

    def send_result_back_to_controller(self, task):

        http_request(url=self.controller_address, method=NetworkAPIMethod.CONTROLLER_RETURN,
                     data={'data': Task.serialize(task)})
