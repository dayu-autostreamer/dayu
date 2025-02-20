import json

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form

from fastapi.routing import APIRoute
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.lib.network import NetworkAPIPath, NetworkAPIMethod
from core.lib.common import FileOps

from .controller import Controller


class ControllerServer:
    # TODO: 修改api名字
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

    async def submit_task(self, backtask: BackgroundTasks, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        backtask.add_task(self.submit_task_background, data, file_data)

    async def process_return(self, backtask: BackgroundTasks,  data: str = Form(...)):
        backtask.add_task(self.process_return_background, data)

    def submit_task_background(self, data, file_data):
        self.controller.set_current_task(data)
        FileOps.save_data_file(self.controller.cur_task, file_data)

        self.controller.record_transmit_ts(is_end=True)
        action = self.controller.submit_task()
        if action == 'transmit':
            FileOps.remove_data_file(self.controller.cur_task)

    def process_return_background(self, data):
        self.controller.set_current_task(data)

        self.controller.record_execute_ts(is_end=True)
        self.controller.process_return()
        action = self.controller.submit_task()
        if action == 'transmit':
            FileOps.remove_data_file(self.controller.cur_task)
