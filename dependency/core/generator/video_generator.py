import time

from core.lib.common import ClassType, ClassFactory, Context, LOGGER

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

    def submit_task_to_controller(self, cur_task):
        self.record_total_start_ts(cur_task)
        return super().submit_task_to_controller(cur_task)

    def run(self):
        # Start with the default local scheduling view. Once runtime services are
        # healthy, request a fresh scheduler decision before ingesting data.
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

            # Skip this round when the getter filter decides not to ingest data.
            if not self.getter_filter(self):
                LOGGER.info('[Filter Getter] step to next round of getter.')
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
            # An exhausted/closed HTTP datasource returns ``False``.  Retain
            # the same reserved identity so the next scheduling request is a
            # replay of the pending decision instead of advancing the online
            # policy for a task that never exists.  Successful getters in the
            # existing hook ecosystem may return either ``True`` or ``None``.
            if ingested is not False:
                pending_task_identity = None
                pending_schedule_ready = False
