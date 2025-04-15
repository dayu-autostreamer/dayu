import abc
import json
import time

from .base_getter import BaseDataGetter

from core.lib.common import ClassFactory, ClassType, LOGGER, FileOps, Context, Counter, NameMaintainer
from core.lib.network import http_request
from core.lib.estimation import TimeEstimator

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

    @TimeEstimator.estimate_duration_time
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

            if self.hash_codes:
                response = http_request(system.video_data_source + '/file', method='GET', no_decode=True)

        self.file_name = NameMaintainer.get_task_data_file_name(system.source_id, task_id, self.file_suffix)

        with open(self.file_name, 'wb') as f:
            f.write(response.content)

    @staticmethod
    def compute_cost_time(system, cost):
        return max(1 / system.meta_data['fps'] * system.meta_data['buffer_size'] - cost, 0)

    def __call__(self, system):
        new_task_id = Counter.get_count('task_id')
        delay = self.request_source_data(system, new_task_id)

        sleep_time = self.compute_cost_time(system, delay)
        LOGGER.info(f'[Camera Simulation] source {system.source_id}: sleep {sleep_time}s')
        time.sleep(sleep_time)

        new_task = system.generate_task(new_task_id, system.task_dag, system.meta_data, self.file_name, self.hash_codes)
        system.submit_task_to_controller(new_task)

        FileOps.remove_file(self.file_name)
