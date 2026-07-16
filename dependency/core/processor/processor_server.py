import json
import threading
import time
from collections import deque

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from core.lib.common import Context, FileOps, LOGGER
from core.lib.content import Task
from core.lib.estimation import TimeEstimator
from core.lib.network import deliver_task, NetworkAPIMethod, NetworkAPIPath, task_ack
from core.lib.runtime import RuntimeContext, RuntimeResolver


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

        self.runtime_context = RuntimeContext.get_default()
        self.runtime_resolver = RuntimeResolver(self.runtime_context)
        self.local_device = self.runtime_context.local_node
        self.processor_port = Context.get_parameter('GUNICORN_PORT')
        self._accepted_tasks = {}
        self._accepted_task_expirations = deque()
        self._accepted_tasks_lock = threading.Lock()

        threading.Thread(target=self.loop_process, name="ProcessorLoop", daemon=True).start()

    async def health_check(self):
        return {'status': 'ok'}

    async def process_service(self, file: UploadFile = File(...), data: str = Form(...)):
        file_data = await file.read()
        return await run_in_threadpool(self.accept_task, data, file_data)

    def accept_task(self, data, file_data):
        cur_task = Task.deserialize(data)
        FileOps.save_task_file_in_temp(cur_task, file_data)
        self._enqueue_task_once(cur_task)
        LOGGER.debug(f'[Task Queue] Queue Size (receive request): {self.task_queue.size()}')
        LOGGER.debug(f'[Monitor Task] (Process Request Accepted) '
                     f'Source: {cur_task.get_source_id()} / Task: {cur_task.get_task_id()} ')
        return task_ack(cur_task)

    async def process_local_service(self, data: str = Form(...)):
        """
            Process local services without transmitting files.
        """
        return await run_in_threadpool(self.accept_local_task, data)

    def accept_local_task(self, data):
        cur_task = Task.deserialize(data)
        if not FileOps.touch_task_file_in_temp(cur_task):
            raise HTTPException(status_code=503, detail="task artifact is unavailable")
        self._enqueue_task_once(cur_task)
        return task_ack(cur_task)

    def _enqueue_task_once(self, task):
        now = time.monotonic()
        task_uuid = task.get_task_uuid()
        with self._accepted_tasks_lock:
            while self._accepted_task_expirations and self._accepted_task_expirations[0][0] <= now:
                expires_at, accepted_uuid = self._accepted_task_expirations.popleft()
                if self._accepted_tasks.get(accepted_uuid) == expires_at:
                    self._accepted_tasks.pop(accepted_uuid, None)
            if task_uuid in self._accepted_tasks:
                return False
            self.task_queue.put(task)
            expires_at = now + self.runtime_context.lease_ttl_seconds
            self._accepted_tasks[task_uuid] = expires_at
            self._accepted_task_expirations.append((expires_at, task_uuid))
            return True

    async def process_return_service(self, file: UploadFile = File(...),
                                     data: str = Form(...)):
        file_data = await file.read()
        return await run_in_threadpool(self.process_return_task, data, file_data)

    def process_return_task(self, data, file_data):
        cur_task = Task.deserialize(data)
        LOGGER.info(f'[Process Return] Process task: source {cur_task.get_source_id()}  / '
                    f'task {cur_task.get_task_id()}')
        FileOps.save_task_file_in_temp(cur_task, file_data)

        new_task = self.processor(cur_task)
        current_content = new_task.get_current_content() if new_task else None
        output_labels = []
        if isinstance(current_content, dict) and isinstance(current_content.get('outputs'), dict):
            output_labels = list(current_content['outputs'].keys())
        LOGGER.debug(f'[Processor Return completed] output labels: {output_labels}')
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
            task = self.task_queue.get()
            if not task:
                time.sleep(0.01)
                continue
            LOGGER.debug(f'[Task Queue] Queue Size (loop): {self.task_queue.size()}')

            try:
                new_task = self.process_task_service(task)
            except Exception as e:
                LOGGER.critical("[Processor Error] Processor encountered error when processing data.")
                LOGGER.exception(e)
                self.task_queue.put(task)
                time.sleep(0.1)
                continue

            if new_task is None:
                self.task_queue.put(task)
                time.sleep(0.1)
                continue
            self.send_result_back_to_controller(new_task)

    def process_task_service(self, task: Task):
        LOGGER.debug(f'[Monitor Task] (Process start) Source: {task.get_source_id()} / Task: {task.get_task_id()} ')

        if not FileOps.touch_task_file_in_temp(task):
            LOGGER.warning(f'[Task Artifact] File unavailable. '
                           f'Source: {task.get_source_id()} / Task: {task.get_task_id()}')
            return None
        TimeEstimator.record_dag_ts(task, is_end=False, sub_tag='real_execute')
        new_task = self.processor(task)
        if new_task is None:
            LOGGER.warning(f'[Monitor Task] Processor returned no task. '
                           f'Source: {task.get_source_id()} / Task: {task.get_task_id()}')
            return None
        duration = TimeEstimator.record_dag_ts(new_task, is_end=True, sub_tag='real_execute')
        new_task.save_real_execute_time(duration)

        LOGGER.debug(f'[Monitor Task] (Process end) Source: {task.get_source_id()} / Task: {task.get_task_id()} ')
        LOGGER.info(f'[Process Task] Source: {task.get_source_id()} / Task: {task.get_task_id()} Duration: {duration} ')

        return new_task

    def send_result_back_to_controller(self, task):
        while True:
            try:
                controller_address = self.runtime_resolver.resolve_url(
                    "controller",
                    path=NetworkAPIPath.CONTROLLER_RETURN,
                    task=task,
                    target_node=self.local_device,
                    exact=True,
                )
                return deliver_task(
                    url=controller_address,
                    method=NetworkAPIMethod.CONTROLLER_RETURN,
                    task=task,
                    persistent=True,
                )
            except Exception as exc:
                LOGGER.warning(
                    f'[Task Delivery] Retain processed result after delivery setup failure. '
                    f'Source: {task.get_source_id()} / Task: {task.get_task_id()} / Error: {exc}'
                )
                time.sleep(0.5)
