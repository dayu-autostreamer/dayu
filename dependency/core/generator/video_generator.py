from .generator import Generator
from core.lib.content import Task

from core.lib.common import ClassType, ClassFactory, Context, LOGGER


@ClassFactory.register(ClassType.GENERATOR, alias='video')
class VideoGenerator(Generator):
    def __init__(self, source_id: int, source_url: str,
                 source_metadata: dict, pipeline: list):
        super().__init__(source_id, source_metadata, pipeline)

        self.task_id = 0
        self.video_data_source = source_url

        self.frame_filter = Context.get_algorithm('GEN_FILTER')
        self.frame_process = Context.get_algorithm('GEN_PROCESS')
        self.frame_compress = Context.get_algorithm('GEN_COMPRESS')
        self.getter_filter = Context.get_algorithm('GEN_GETTER_FILTER')

    def submit_task_to_controller(self, compressed_path, hash_codes):
        self.current_task = Task(source_id=self.source_id,
                                 task_id=self.task_id,
                                 source_device=self.local_device,
                                 pipeline=self.task_pipeline,
                                 metadata=self.meta_data,
                                 raw_metadata=self.raw_meta_data,
                                 content=self.task_content,
                                 hash_data=hash_codes,
                                 file_path=compressed_path
                                 )
        self.record_total_start_ts()
        super().submit_task_to_controller(compressed_path, hash_codes)

    def run(self):
        # initialize with default schedule policy
        self.after_schedule_operation(self, None)

        while True:
            if not self.getter_filter(self):
                LOGGER.info('[Filter Getter] step to next round of getter.')
                continue
            self.data_getter(self)

            self.task_id += 1

            self.request_schedule_policy()
