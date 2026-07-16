import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from core.lib.common import FileOps, FileCleaner
from core.lib.content import Task
from core.lib.network import NetworkAPIMethod, NetworkAPIPath, task_ack

from .controller import Controller


class ControllerServer:
    def __init__(self):
        self.controller = Controller()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            cleaner = FileCleaner(
                folder=FileOps.get_task_temp_directory(),
                poll_seconds=30,
                ttl_seconds=self.controller.runtime_context.lease_ttl_seconds,
                recursive=False,
                max_delete_per_round=200,
            )
            cleaner.start()

            try:
                yield
            finally:
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
                     ),
            APIRoute(NetworkAPIPath.CONTROLLER_CLEAR_PROCESSOR_QUEUES,
                     self.clear_processor_queues,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.CONTROLLER_CLEAR_PROCESSOR_QUEUES]
                     ), ],
            log_level='trace',
            timeout=6000,
            lifespan=lifespan)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

    async def check_processor_health(self, data: str = Form("{}")):
        """check if processor is healthy"""
        try:
            request = json.loads(data) if isinstance(data, str) and data else {}
        except Exception as exc:
            return {'status': 'not ok', 'error': f'invalid processor health request: {exc}'}
        return {'status': 'ok'} if self.controller.check_processor_health(request) else {'status': 'not ok'}

    async def submit_task(self, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        return await run_in_threadpool(self.accept_task, data, file_data)

    async def process_return(self, data: str = Form(...)):
        return await run_in_threadpool(self.accept_result, data)

    async def clear_processor_queues(self, data: str = Form("{}")):
        try:
            request = json.loads(data) if data else {}
        except Exception as exc:
            return {
                "ok": False,
                "error": f"invalid processor queue clear request: {exc}",
            }
        return self.controller.clear_processor_queues(request)

    def accept_task(self, data, file_data):
        """Acknowledge only after this controller has transferred ownership."""
        cur_task = Task.deserialize(data)
        FileOps.save_task_file_in_temp(cur_task, file_data)
        self.controller.record_transmit_ts(cur_task, is_end=True)
        if not self.controller.submit_task(cur_task):
            raise HTTPException(status_code=503, detail="downstream task ownership was not acknowledged")
        return task_ack(cur_task)

    def accept_result(self, data):
        """Retain or forward a Processor result before acknowledging it."""
        cur_task = Task.deserialize(data)
        if not FileOps.touch_task_file_in_temp(cur_task):
            raise HTTPException(status_code=503, detail="task artifact is unavailable")
        self.controller.record_execute_ts(cur_task, is_end=True)
        if not self.controller.process_return(cur_task):
            raise HTTPException(status_code=503, detail="downstream task ownership was not acknowledged")
        return task_ack(cur_task)
