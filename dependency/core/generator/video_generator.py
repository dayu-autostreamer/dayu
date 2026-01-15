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
        # initialize with default schedule policy
        self.after_schedule_operation(self, None)

        service_running_flag = False
        while True:
            if not KubeConfig.check_services_running():
                service_running_flag = False
                LOGGER.debug("Services not in running state, wait for service deployment..")
                time.sleep(0.5)
                continue
            if not service_running_flag:
                if HealthChecker.check_processors_health():
                    service_running_flag = True
                else:
                    LOGGER.debug("Services not in running state, wait for service deployment..")
                    time.sleep(0.5)
                    continue
            # skip getter according to some specific requirements
            if not self.getter_filter(self):
                LOGGER.info('[Filter Getter] step to next round of getter.')
                continue

            # get data from source
            self.data_getter(self)

            # request schedule policy for subsequent tasks
            if self.cumulative_scheduling_frame_count > \
                    self.request_scheduling_interval * self.raw_meta_data.get('fps', 0):
                LOGGER.debug(f'[Scheduling Request] Request a Scheduling policy from scheduler.')
                self.request_schedule_policy()
                self.cumulative_scheduling_frame_count = 0
