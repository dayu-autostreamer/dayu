from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form

from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.lib.network import NetworkAPIPath, NetworkAPIMethod
from core.lib.common import FileOps
from core.lib.common import Context,LOGGER
from core.lib.content import Task

from .controller import Controller


class ControllerServer:
    def __init__(self):
        self.controller = Controller()

        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.CONTROLLER_TASK,
                     self.submit_task,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.CONTROLLER_TASK]
                     ),
            APIRoute(NetworkAPIPath.CONTROLLER_RETURN,
                     self.process_return,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.CONTROLLER_RETURN]
                     ),
        ], log_level='trace', timeout=6000)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        self.is_delete_temp_files = Context.get_parameter('DELETE_TEMP_FILES', direct=False)

    async def submit_task(self, backtask: BackgroundTasks, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        backtask.add_task(self.submit_task_background, data, file_data)

    async def process_return(self, backtask: BackgroundTasks,  data: str = Form(...)):
        backtask.add_task(self.process_return_background, data)

    def submit_task_background(self, data, file_data):
        """deal with tasks submitted by the generator or other controllers"""
        cur_task = Task.deserialize(data)
        FileOps.save_data_file(cur_task, file_data)
        # record end time of transmitting
        self.controller.record_transmit_ts(cur_task, is_end=True)

        action = self.controller.submit_task(cur_task)

        # for execute action, the file is remained
        # so that task returned from processor don't need to carry with file.
        LOGGER.debug(f'[DEBUG for delete temp files] action: {action}, delete_temp_file:{self.is_delete_temp_files}')
        if self.is_delete_temp_files and not action == 'execute':
            FileOps.remove_data_file(cur_task)

    def process_return_background(self, data):
        """deal with tasks returned by the processor"""
        cur_task = Task.deserialize(data)
        # record end time of executing
        self.controller.record_execute_ts(cur_task, is_end=True)

        actions = self.controller.process_return(cur_task)

        # for execute action, the file is remained
        # so that task returned from processor don't need to carry with file;
        # for wait action of joint node, the file is remained
        # so that joint task merged from waiting tasks has file to transmit.
        LOGGER.debug(f'[DEBUG for delete temp files] action: {actions}, delete_temp_file:{self.is_delete_temp_files}')
        if self.is_delete_temp_files and 'execute' not in actions and 'wait' not in actions:
            FileOps.remove_data_file(cur_task)
