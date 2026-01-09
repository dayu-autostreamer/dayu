from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Form
from fastapi.routing import APIRoute
from contextlib import asynccontextmanager
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from core.lib.network import NetworkAPIPath, NetworkAPIMethod
from core.lib.common import FileOps, Context, FileCleaner
from core.lib.content import Task

from .controller import Controller


class ControllerServer:
    def __init__(self):
        self.controller = Controller()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            FileOps.clear_temp_directory()
            app.state.file_cleaner = None
            is_delete_temp_files = Context.get_parameter('DELETE_TEMP_FILES', direct=False)

            if is_delete_temp_files:
                cleaner = FileCleaner(
                    folder=Context.get_temporary_file_path(''),
                    poll_seconds=30,
                    ttl_seconds=120,
                    recursive=False,
                    max_delete_per_round=200
                )
                cleaner.start()
                app.state.file_cleaner = cleaner

            try:
                yield
            finally:
                # Shutdown
                FileOps.clear_temp_directory()
                cleaner = getattr(app.state, "file_cleaner", None)
                if cleaner:
                    cleaner.stop(join=True, timeout=3.0)

        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.CONTROLLER_CHECK,
                     self.check_processor_health,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.CONTROLLER_CHECK]
                     ),
            APIRoute(NetworkAPIPath.CONTROLLER_TASK,
                     self.submit_task,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.CONTROLLER_TASK]
                     ),
            APIRoute(NetworkAPIPath.CONTROLLER_RETURN,
                     self.process_return,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.CONTROLLER_RETURN]
                     ), ],
            log_level='trace',
            timeout=6000,
            lifespan=lifespan)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        FileOps.clear_temp_directory()
        self.is_delete_temp_files = Context.get_parameter('DELETE_TEMP_FILES', direct=False)
        if self.is_delete_temp_files:
            self.file_cleaner = FileCleaner(folder=Context.get_temporary_file_path(''),
                                            poll_seconds=30, ttl_seconds=120, recursive=False,
                                            max_delete_per_round=200)
            self.file_cleaner.start()

    async def check_processor_health(self):
        """check if processor is healthy"""
        return {'status': 'ok'} if self.controller.check_processor_health() else {'status': 'not ok'}

    async def submit_task(self, backtask: BackgroundTasks, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        backtask.add_task(self.submit_task_background, data, file_data)

    async def process_return(self, backtask: BackgroundTasks, data: str = Form(...)):
        backtask.add_task(self.process_return_background, data)

    def submit_task_background(self, data, file_data):
        """deal with tasks submitted by the generator or other controllers"""
        cur_task = Task.deserialize(data)
        FileOps.save_task_file_in_temp(cur_task, file_data)
        # record end time of transmitting
        self.controller.record_transmit_ts(cur_task, is_end=True)

        self.controller.submit_task(cur_task)

    def process_return_background(self, data):
        """deal with tasks returned by the processor"""
        cur_task = Task.deserialize(data)
        # record end time of executing
        self.controller.record_execute_ts(cur_task, is_end=True)

        self.controller.process_return(cur_task)
