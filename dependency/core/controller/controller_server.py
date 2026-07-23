import json
import os
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from core.lib.common import FileOps, FileCleaner, LOGGER
from core.lib.content import Task
from core.lib.network import NetworkAPIMethod, NetworkAPIPath, task_ack

from .controller import Controller


class ControllerServer:
    def __init__(self):
        self.controller = Controller()
        self._init_inbox()

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
            self._start_inbox_workers()

            try:
                yield
            finally:
                self._stop_inbox_workers(timeout=3.0)
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

    def _init_inbox(self):
        """Create the local ownership boundary for Controller deliveries.

        A Controller acknowledges a task after it has stored the artifact and
        accepted one idempotent inbox record.  Forwarding is deliberately
        performed by background workers: making the caller wait for the full
        downstream chain creates nested Controller requests, exhausts request
        thread pools, and can replay a completed fork after a timeout.
        """
        self._inbox = queue.Queue()
        self._accepted_work = {}
        self._accepted_work_expirations = deque()
        self._accepted_work_lock = threading.Lock()
        self._inbox_stop = threading.Event()
        self._inbox_workers = []
        try:
            configured_workers = int(os.getenv("CONTROLLER_FORWARD_WORKERS", "8"))
        except (TypeError, ValueError):
            configured_workers = 8
        self._inbox_worker_count = max(1, configured_workers)

    def _prune_accepted_work_locked(self, now):
        while (
            self._accepted_work_expirations
            and self._accepted_work_expirations[0][0] <= now
        ):
            expires_at, work_key = self._accepted_work_expirations.popleft()
            if self._accepted_work.get(work_key) == expires_at:
                self._accepted_work.pop(work_key, None)

    def _accept_work_once(self, kind, task, prepare):
        work_key = (str(kind), task.get_task_uuid())
        now = time.monotonic()
        with self._accepted_work_lock:
            self._prune_accepted_work_locked(now)
            if work_key in self._accepted_work:
                return False

            # Ownership is not acknowledged until the immutable artifact is
            # available locally.  Keep this operation inside the claim lock so
            # concurrent retries cannot enqueue the same stage twice.
            prepare()
            expires_at = now + self.controller.runtime_context.lease_ttl_seconds
            self._accepted_work[work_key] = expires_at
            self._accepted_work_expirations.append((expires_at, work_key))
            self._inbox.put((str(kind), task))
            return True

    def _start_inbox_workers(self):
        if self._inbox_workers:
            return
        self._inbox_stop.clear()
        for worker_index in range(self._inbox_worker_count):
            worker = threading.Thread(
                target=self._inbox_worker_loop,
                name=f"ControllerForward-{worker_index}",
                daemon=True,
            )
            worker.start()
            self._inbox_workers.append(worker)

    def _stop_inbox_workers(self, timeout=3.0):
        self._inbox_stop.set()
        workers = list(self._inbox_workers)
        for _ in workers:
            self._inbox.put(None)
        deadline = time.monotonic() + max(0.0, float(timeout))
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        self._inbox_workers = []

    def _inbox_worker_loop(self):
        while not self._inbox_stop.is_set():
            try:
                work = self._inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if work is None:
                self._inbox.task_done()
                break

            kind, task = work
            try:
                while not self._inbox_stop.is_set():
                    try:
                        accepted = (
                            self.controller.submit_task(task)
                            if kind == "task"
                            else self.controller.process_return(task)
                        )
                        if accepted:
                            break
                        LOGGER.warning(
                            "[Controller Inbox] Downstream did not accept retained work; "
                            f"retrying kind={kind} source={task.get_source_id()} "
                            f"task={task.get_task_id()} service={task.get_flow_index()}"
                        )
                    except Exception:
                        LOGGER.exception(
                            "[Controller Inbox] Retained work failed; retrying "
                            f"kind={kind} source={task.get_source_id()} "
                            f"task={task.get_task_id()} service={task.get_flow_index()}"
                        )
                    self._inbox_stop.wait(0.5)
            finally:
                self._inbox.task_done()

    def accept_task(self, data, file_data):
        """Accept one remote task into the local idempotent forwarding inbox."""
        cur_task = Task.deserialize(data)
        accepted = self._accept_work_once(
            "task",
            cur_task,
            lambda: FileOps.save_task_file_in_temp(cur_task, file_data),
        )
        if accepted:
            self.controller.record_transmit_ts(cur_task, is_end=True)
        return task_ack(cur_task)

    def accept_result(self, data):
        """Accept one Processor result into the local forwarding inbox."""
        cur_task = Task.deserialize(data)
        def prepare_result():
            if not FileOps.touch_task_file_in_temp(cur_task):
                raise HTTPException(status_code=503, detail="task artifact is unavailable")

        accepted = self._accept_work_once("result", cur_task, prepare_result)
        if accepted:
            self.controller.record_execute_ts(cur_task, is_end=True)
        return task_ack(cur_task)
