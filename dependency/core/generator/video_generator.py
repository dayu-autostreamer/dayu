import time

from core.lib.common import ClassType, ClassFactory, Context, LOGGER, KubeConfig, HealthChecker

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
        super().submit_task_to_controller(cur_task)

    def run(self):
        # Start with the default local scheduling view. Once runtime services are
        # healthy, request a fresh scheduler decision before ingesting data.
        self.after_schedule_operation(self, None)

        services_ready = False
        initial_schedule_pending = True
        while True:
            if not KubeConfig.check_services_running():
                services_ready = False
                initial_schedule_pending = True
                LOGGER.debug("Services not in running state, wait for service deployment..")
                time.sleep(0.5)
                continue

            if not services_ready:
                if not HealthChecker.check_processors_health():
                    LOGGER.debug("Processors are not healthy yet, wait before requesting scheduler decisions..")
                    time.sleep(0.5)
                    continue
                services_ready = True
                initial_schedule_pending = True

            if initial_schedule_pending:
                LOGGER.debug('[Scheduling Request] Request an initial scheduling policy after services become healthy.')
                self.request_schedule_policy()
                self.cumulative_scheduling_frame_count = 0
                initial_schedule_pending = False

            # Skip this round when the getter filter decides not to ingest data.
            if not self.getter_filter(self):
                LOGGER.info('[Filter Getter] step to next round of getter.')
                time.sleep(0.5)
                continue

            # Refresh scheduling policy periodically after enough frames have
            # been processed since the last scheduling decision.
            scheduling_threshold = self.request_scheduling_interval * self.raw_meta_data.get('fps', 0)
            if self.cumulative_scheduling_frame_count > scheduling_threshold:
                LOGGER.debug(f'[Scheduling Request] Request a Scheduling policy from scheduler.')
                self.request_schedule_policy()
                self.cumulative_scheduling_frame_count = 0

            # Ingest the next chunk/frame from the source.
            self.data_getter(self)
