import queue
import threading
import time
from pathlib import Path

from core.lib.common import (
    ClassType,
    ClassFactory,
    Context,
    LOGGER,
)
from core.lib.algorithms.data_getter.base_getter import DataGetterStatus

from .generator import Generator


@ClassFactory.register(ClassType.GENERATOR, alias='video')
class VideoGenerator(Generator):
    def __init__(self, source_id: int, source_url: str,
                 source_metadata: dict, dag: dict):
        super().__init__(source_id, source_metadata, dag)
        self.video_data_source = source_url

        self.frame_filter = Context.get_algorithm('GEN_FILTER')
        self.frame_process = Context.get_algorithm('GEN_PROCESS')
        self.frame_compress = Context.get_algorithm('GEN_COMPRESS')
        self.getter_filter = Context.get_algorithm('GEN_GETTER_FILTER')

        self.cumulative_scheduling_frame_count = 0

        try:
            task_offer_limit = int(
                Context.get_parameter('TASK_OFFER_LIMIT', 0, direct=False) or 0
            )
        except (TypeError, ValueError):
            task_offer_limit = 0
        self.task_offer_limit = max(0, task_offer_limit)
        self._offered_task_count = 0
        self._offer_limit_reached_logged = False

        self._source_exhausted = False
        async_submission = Context.get_parameter(
            'ASYNC_TASK_SUBMISSION', False, direct=False
        )
        self.async_task_submission = str(async_submission).strip().lower() in (
            '1', 'true', 'yes', 'on'
        )
        try:
            submission_queue_depth = int(Context.get_parameter(
                'TASK_SUBMISSION_QUEUE_DEPTH', 16, direct=False
            ) or 16)
        except (TypeError, ValueError):
            submission_queue_depth = 16
        try:
            submission_workers = int(Context.get_parameter(
                'TASK_SUBMISSION_WORKERS', 1, direct=False
            ) or 1)
        except (TypeError, ValueError):
            submission_workers = 1
        self.task_submission_workers = max(1, min(16, submission_workers))
        self._submission_queue = queue.Queue(
            maxsize=max(1, submission_queue_depth)
        )
        self._submission_worker_threads = []
        if self.async_task_submission:
            for index in range(self.task_submission_workers):
                worker = threading.Thread(
                    target=self._submission_worker,
                    name=(
                        f'dayu-task-submitter-{self.source_id}-{index}'
                    ),
                    daemon=True,
                )
                worker.start()
                self._submission_worker_threads.append(worker)

    def _submission_worker(self):
        while True:
            cur_task, file_content = self._submission_queue.get()
            try:
                submitted = self._submit_task_to_controller(
                    cur_task,
                    file_content=file_content,
                )
                if not submitted:
                    LOGGER.error(
                        f'[Async Submission] Generated task rejected: '
                        f'source={cur_task.get_source_id()} '
                        f'task={cur_task.get_task_id()}'
                    )
            except BaseException as exc:
                LOGGER.exception(exc)
            finally:
                self._submission_queue.task_done()

    def submit_task_to_controller(self, cur_task):
        # The application task starts immediately before lease admission and
        # Controller submission; source decode/encode is not application
        # latency.
        self.record_total_start_ts(cur_task)
        if not self.async_task_submission:
            return super().submit_task_to_controller(cur_task)

        # Snapshot the immutable artifact before the getter releases its local
        # file.  The bounded queue preserves backpressure; any wait here is
        # included in the task's SLO and is detected by the evaluation
        # runner's end-to-end validity gate.
        file_content = Path(cur_task.get_file_path()).read_bytes()
        self._submission_queue.put((cur_task, file_content))
        return True

    def run(self):
        # Let the selected extension initialize any dimensions it deliberately
        # keeps fixed. The host itself chooses no configuration, offloading, or
        # deployment value, and still obtains a schedulable plan before data is
        # ingested.
        self.after_schedule_operation(self, None)

        initial_schedule_pending = True
        pending_task_identity = None
        pending_schedule_ready = False
        while True:
            if self._runtime_schedule_refresh_required.is_set():
                initial_schedule_pending = True
                pending_task_identity = None
                pending_schedule_ready = False
                self._runtime_schedule_refresh_required.clear()

            if (
                self.task_offer_limit > 0
                and self._offered_task_count >= self.task_offer_limit
            ):
                if not self._offer_limit_reached_logged:
                    LOGGER.info(
                        '[Evaluation Input] Offered-task limit reached: '
                        f'{self._offered_task_count}; generator is quiescent.'
                    )
                    self._offer_limit_reached_logged = True
                time.sleep(0.5)
                continue

            if self._source_exhausted and pending_task_identity is not None:
                if self.cancel_schedule_reservation(pending_task_identity):
                    pending_task_identity = None
                    pending_schedule_ready = False

            # Skip this round when the getter filter decides not to ingest data.
            getter_allowed = self.getter_filter(self)
            if not getter_allowed:
                LOGGER.info('[Filter Getter] step to next round of getter.')
                time.sleep(0.5)
                continue

            if self._source_exhausted:
                reset_ready = getattr(
                    self.data_getter,
                    'datasource_reset_ready',
                    None,
                )
                if (
                    callable(reset_ready)
                    and reset_ready(self)
                ):
                    self._source_exhausted = False
                    initial_schedule_pending = True
                    LOGGER.info(
                        '[Camera Simulation] datasource reset observed; '
                        'start a new finite playback epoch.'
                    )
                else:
                    # The query is still open on a finite source that has
                    # already ended. Do not create more phantom decisions.
                    time.sleep(0.5)
                    continue

            # Reserve the root identity before scheduling so the policy can
            # reason about the exact task that may consume its decision. Source
            # data is fetched afterwards because existing policies can change
            # buffer size, frame rate, resolution, encoding, and DAG routing.
            if pending_task_identity is None:
                pending_task_identity = self.create_task_identity()
                pending_schedule_ready = False

            # Refresh scheduling policy periodically after enough frames have
            # been processed since the last scheduling decision.
            scheduling_threshold = self.request_scheduling_interval * self.raw_meta_data.get('fps', 0)
            should_schedule = (
                not pending_schedule_ready
                and (
                    initial_schedule_pending
                    or self.request_scheduling_interval <= 0
                    or self.cumulative_scheduling_frame_count > scheduling_threshold
                )
            )
            if should_schedule:
                LOGGER.debug('[Scheduling Request] Request a task-aware scheduling policy.')
                if not self.request_schedule_policy(pending_task_identity):
                    LOGGER.debug('[Runtime Directory] Updated scheduling policy is not routable; postpone ingestion.')
                    time.sleep(0.5)
                    continue
                self.cumulative_scheduling_frame_count = 0
                initial_schedule_pending = False
                pending_schedule_ready = True
            elif not pending_schedule_ready:
                # The periodic threshold did not request a fresh plan, so this
                # task deliberately reuses the last accepted routable plan.
                pending_schedule_ready = self.runtime_routes_ready()
                if not pending_schedule_ready:
                    time.sleep(0.5)
                    continue

            # Ingest the next chunk/frame from the source.
            ingested = self.data_getter(self, pending_task_identity)
            if ingested is DataGetterStatus.EXHAUSTED:
                # The plan was reserved before source materialization. Revoke
                # it explicitly so future-state/profile logic never counts a
                # task that cannot exist. The finite source remains quiescent
                # until a query close/reset/open boundary is observed.
                if self.cancel_schedule_reservation(pending_task_identity):
                    pending_task_identity = None
                    pending_schedule_ready = False
                self._source_exhausted = True
                continue

            # Successful getters in the existing hook ecosystem may return
            # either True or None. False can mean a materialized task was
            # rejected by a retired runtime; it is not datasource exhaustion.
            self._offered_task_count += 1
            pending_task_identity = None
            pending_schedule_ready = False
