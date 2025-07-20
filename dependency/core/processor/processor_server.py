import threading

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form

from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.lib.common import Context, SystemConstant
from core.lib.common import LOGGER, FileOps
from core.lib.network import NodeInfo, PortInfo, http_request, merge_address, NetworkAPIMethod, NetworkAPIPath
from core.lib.content import Task
from core.lib.estimation import TimeEstimator


class ProcessorServer:
    def __init__(self):
        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.PROCESSOR_PROCESS,
                     self.process_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_PROCESS]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_PROCESS_RETURN,
                     self.process_return_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_PROCESS_RETURN]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_QUEUE_LENGTH,
                     self.query_queue_length,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_QUEUE_LENGTH]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_MODEL_FLOPS,
                     self.query_model_flops,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_MODEL_FLOPS]
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
        self.controller_address = merge_address(NodeInfo.hostname2ip(self.local_device),
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

    async def process_return_service(self, file: UploadFile = File(...),
                                     data: str = Form(...)):
        file_data = await file.read()
        cur_task = Task.deserialize(data)
        LOGGER.info(f'[Process Return Background] Process task: source {cur_task.get_source_id()}  / '
                    f'task {cur_task.get_task_id()}')
        FileOps.save_data_file(cur_task, file_data)

        new_task = self.processor(cur_task)
        LOGGER.debug(f'[Processor Return completed] content length: {len(new_task.get_current_content())}')
        FileOps.remove_data_file(cur_task)
        if new_task:
            return new_task.serialize()
        return None

    async def query_queue_length(self):
        return self.task_queue.size()

    async def query_model_flops(self):
        return self.processor.flops

    def loop_process(self):
        LOGGER.info('Start processing loop..')
        while True:
            if self.task_queue.empty():
                continue
            task = self.task_queue.get()
            if not task:
                continue
            LOGGER.debug(f'[Task Queue] Queue Size (loop): {self.task_queue.size()}')

            try:
                new_task = self.process_task_service(task)
            except Exception as e:
                LOGGER.critical("[Processor Error] Processor encountered error when processing data.")
                LOGGER.exception(e)
                continue

            if new_task:
                self.send_result_back_to_controller(new_task)
            FileOps.remove_data_file(task)

    def process_task_service(self, task: Task):
        LOGGER.debug(f'[Monitor Task] (Process start) Source: {task.get_source_id()} / Task: {task.get_task_id()} ')

        TimeEstimator.record_dag_ts(task, is_end=False, sub_tag='real_execute')
        new_task = self.processor(task)
        duration = TimeEstimator.record_dag_ts(new_task, is_end=True, sub_tag='real_execute')
        new_task.save_real_execute_time(duration)

        LOGGER.debug(f'[Monitor Task] (Process end) Source: {task.get_source_id()} / Task: {task.get_task_id()} ')
        LOGGER.info(f'[Process Task] Source: {task.get_source_id()} / Task: {task.get_task_id()} Duration: {duration} ')

        return new_task

    def send_result_back_to_controller(self, task):

        http_request(url=self.controller_address, method=NetworkAPIMethod.CONTROLLER_RETURN,
                     data={'data': task.serialize()})
