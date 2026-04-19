import json
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
        self.processor = Context.get_algorithm('PROCESSOR')

        self.app = FastAPI(routes=[
            APIRoute(NetworkAPIPath.PROCESSOR_HEALTH,
                     self.health_check,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_HEALTH]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_PROCESS,
                     self.process_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_PROCESS]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_PROCESS_LOCAL,
                     self.process_local_service,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_PROCESS_LOCAL]
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
            APIRoute(NetworkAPIPath.PROCESSOR_CLEAR_QUEUE,
                     self.clear_queue,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_CLEAR_QUEUE]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_MODEL_FLOPS,
                     self.query_model_flops,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_MODEL_FLOPS]
                     ),
            APIRoute(NetworkAPIPath.PROCESSOR_MODEL_MEMORY,
                     self.query_model_memory,
                     response_class=JSONResponse,
                     methods=[NetworkAPIMethod.PROCESSOR_MODEL_MEMORY]
                     ),
        ], log_level='trace', timeout=6000)

        self.app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        self.task_queue = Context.get_algorithm('PRO_QUEUE')

        self.local_device = NodeInfo.get_local_device()
        self.processor_port = Context.get_parameter('GUNICORN_PORT')
        self.controller_port = PortInfo.get_component_port(SystemConstant.CONTROLLER.value)
        self.controller_address = merge_address(NodeInfo.hostname2ip(self.local_device),
                                                port=self.controller_port,
                                                path=NetworkAPIPath.CONTROLLER_RETURN)

        threading.Thread(target=self.loop_process, name="ProcessorLoop", daemon=True).start()

    async def health_check(self):
        return {'status': 'ok'}

    async def process_service(self, backtask: BackgroundTasks, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        backtask.add_task(self.process_service_background, data, file_data)

    def process_service_background(self, data, file_data):
        cur_task = Task.deserialize(data)
        FileOps.save_task_file_in_temp(cur_task, file_data)
        self.task_queue.put(cur_task)
        LOGGER.debug(f'[Task Queue] Queue Size (receive request): {self.task_queue.size()}')
        LOGGER.debug(f'[Monitor Task] (Process Request Background) '
                     f'Source: {cur_task.get_source_id()} / Task: {cur_task.get_task_id()} ')

    async def process_local_service(self, backtask: BackgroundTasks, data: str = Form(...)):
        """
            Process local services without transmitting files.
        """
        backtask.add_task(self.process_local_service_background, data)

    def process_local_service_background(self, data):
        cur_task = Task.deserialize(data)
        self.task_queue.put(cur_task)

    async def process_return_service(self, file: UploadFile = File(...),
                                     data: str = Form(...)):
        file_data = await file.read()
        cur_task = Task.deserialize(data)
        LOGGER.info(f'[Process Return Background] Process task: source {cur_task.get_source_id()}  / '
                    f'task {cur_task.get_task_id()}')
        FileOps.save_task_file_in_temp(cur_task, file_data)

        new_task = self.processor(cur_task)
        LOGGER.debug(f'[Processor Return completed] content length: {len(new_task.get_current_content())}')
        FileOps.remove_task_file_in_temp(cur_task)
        if new_task:
            return new_task.serialize()
        return None

    async def query_queue_length(self):
        return self.task_queue.size()

    @staticmethod
    def _normalize_queue_clear_limit(value):
        if value is None:
            return None
        try:
            limit = int(value)
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None

    @staticmethod
    def _task_drop_record(task):
        def _call(name, default=None):
            method = getattr(task, name, None)
            if not callable(method):
                return default
            try:
                return method()
            except Exception:
                return default

        return {
            "source_id": _call("get_source_id"),
            "task_id": _call("get_task_id"),
            "flow_index": _call("get_flow_index"),
            "file_path": _call("get_file_path"),
        }

    async def clear_queue(self, data: str = Form("{}")):
        try:
            payload = json.loads(data) if data else {}
        except Exception as exc:
            return {
                "ok": False,
                "error": f"invalid queue clear request: {exc}",
            }
        if not isinstance(payload, dict):
            payload = {}

        max_count = self._normalize_queue_clear_limit(payload.get("max_count"))
        dry_run = bool(payload.get("dry_run", False))
        reason = str(payload.get("reason") or "manual_queue_clear")

        if dry_run:
            peek = getattr(self.task_queue, "get_all_without_drop", None)
            if not callable(peek):
                return {
                    "ok": False,
                    "error": "queue does not support dry_run preview",
                }
            queued_tasks = peek()
            dropped_tasks = queued_tasks[:max_count] if max_count is not None else queued_tasks
        else:
            drain = getattr(self.task_queue, "drain", None)
            if callable(drain):
                dropped_tasks = drain(max_count=max_count)
            else:
                dropped_tasks = []
                while max_count is None or len(dropped_tasks) < max_count:
                    task = self.task_queue.get()
                    if task is None:
                        break
                    dropped_tasks.append(task)
            for task in dropped_tasks:
                try:
                    FileOps.remove_task_file_in_temp(task)
                except Exception as exc:
                    LOGGER.debug(
                        f"[Task Queue] Failed to remove temp file for dropped task: {exc}"
                    )

        dropped_records = [self._task_drop_record(task) for task in dropped_tasks]
        LOGGER.warning(
            f"[Task Queue] Cleared queued tasks: reason={reason}, dry_run={dry_run}, "
            f"dropped={len(dropped_records)}, remaining={self.task_queue.size()}"
        )
        return {
            "ok": True,
            "device": self.local_device,
            "service": Context.get_parameter("PROCESSOR_SERVICE_NAME", default="unknown"),
            "dry_run": dry_run,
            "cleared_count": 0 if dry_run else len(dropped_records),
            "matched_count": len(dropped_records),
            "remaining_count": self.task_queue.size(),
            "dropped_tasks": dropped_records,
        }

    async def query_model_flops(self):
        return self.processor.flops

    async def query_model_memory(self):
        import os
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss

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
