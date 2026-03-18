import abc
import json
import time
import copy

from .base_getter import BaseDataGetter

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context, Counter, NameMaintainer
from core.lib.network import http_request

__all__ = ('HttpVideoGetter',)


@ClassFactory.register(ClassType.GEN_GETTER, alias='http_video')
class HttpVideoGetter(BaseDataGetter, abc.ABC):
    """
    get video data from http (fastapi)
    preprocessed video data with accuracy information
    """

    def __init__(self):
        self.file_name = None
        self.hash_codes = None

        self.file_suffix = 'mp4'

        self.time_record = 0

    def request_source_data(self, system, task_id):
        data = {
            'source_id': system.source_id,
            'task_id': task_id,
            'meta_data': system.meta_data,
            'raw_meta_data': system.raw_meta_data,
            'gen_filter_name': Context.get_parameter('GEN_FILTER_NAME'),
            'gen_process_name': Context.get_parameter('GEN_PROCESS_NAME'),
            'gen_compress_name': Context.get_parameter('GEN_COMPRESS_NAME')
        }

        response = None
        self.hash_codes = None
        while not self.hash_codes or not response:
            self.hash_codes = http_request(system.video_data_source + '/source', method='GET',
                                           data={'data': json.dumps(data)})
            if self.hash_codes == []:
                return False

            if self.hash_codes:
                response = http_request(system.video_data_source + '/file', method='GET', no_decode=True)
            else:
                time.sleep(1)

        self.file_name = NameMaintainer.get_task_data_file_name(system.source_id, task_id, self.file_suffix)

        with open(self.file_name, 'wb') as f:
            f.write(response.content)
        return True

    @staticmethod
    def compute_cost_time(system, cost, actual_buffer_size=None):
        buffer_size = actual_buffer_size if actual_buffer_size is not None else system.meta_data['buffer_size']
        return max(1 / system.meta_data['fps'] * buffer_size - cost, 0)

    def __call__(self, system):
        new_task_id = Counter.get_count('task_id')

        if not self.request_source_data(system, new_task_id):
            LOGGER.info(f'[Camera Simulation] source {system.source_id}: datasource exhausted, skip current round')
            time.sleep(1)
            return

        actual_buffer_size = len(self.hash_codes) if self.hash_codes else 0
        system.cumulative_scheduling_frame_count += (
            actual_buffer_size *
            system.raw_meta_data.get('fps', 0) /
            system.meta_data.get('fps', 1)
        )

        new_time_record = time.time()
        delay = new_time_record - self.time_record if self.time_record else 0
        self.time_record = new_time_record
        sleep_time = self.compute_cost_time(system, delay, actual_buffer_size=actual_buffer_size)
        LOGGER.info(f'[Camera Simulation] source {system.source_id}: sleep {sleep_time}s')
        time.sleep(sleep_time)

        new_task = system.generate_task(new_task_id, copy.deepcopy(system.task_dag),
                                        copy.deepcopy(system.service_deployment),
                                        copy.deepcopy(system.meta_data),
                                        self.file_name, self.hash_codes)
        system.submit_task_to_controller(new_task)

        FileOps.remove_file(self.file_name)
