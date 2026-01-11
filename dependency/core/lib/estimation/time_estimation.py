import time
from core.lib.content import Task
from core.lib.common import LOGGER, NameMaintainer


class Timer:
    def __init__(self, label=""):
        self.label = label
        self.elapsed_time = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.label:
            LOGGER.info(f"[{self.label}] Execution time: {self.elapsed_time:.4f}s")
        else:
            LOGGER.info(f"Execution time: {self.elapsed_time:.4f}s")

    def get_elapsed_time(self):
        return self.elapsed_time


class TimeEstimator:
    @staticmethod
    def record_dag_ts(task: Task, is_end: bool, sub_tag: str = 'transmit'):
        """
        record dag timestamp in system
        :param task: dag task
        :param is_end: if recording the end of current timestamp
        :param sub_tag: sub tag of ts record name (eg:transmit_time_service_1)
        :return: task: dag task with recorded time
                 duration: time estimation result

        """
        prefix = NameMaintainer.get_time_ticket_tag_prefix(task)
        duration, time_ticket = TimeEstimator.record_ts(task.get_tmp_data(),
                                                        f'{prefix}:{sub_tag}_time_{task.get_flow_index()}',
                                                        is_end=is_end)
        task.record_time_ticket_in_service(type_tag=sub_tag, is_end=is_end, time_ticket=time_ticket)
        return duration

    @staticmethod
    def record_task_ts(task: Task, tag: str, is_end: bool = False):
        """

        :param task: dag task
        :param tag: name of time ticket
        :param is_end: if recording the end of current timestamp
        :return: task: dag task with recorded time
                 duration: time estimation result
        """

        prefix = NameMaintainer.get_time_ticket_tag_prefix(task)
        duration, _ = TimeEstimator.record_ts(task.get_tmp_data(),
                                              tag=f'{prefix}:{tag}',
                                              is_end=is_end)
        return duration

    @staticmethod
    def record_ts(data: dict, tag: str, is_end: bool = False):
        """
        record timestamp in system
        :param data: time dictionary
        :param tag: name of time ticket
        :param is_end: if recording the end of current timestamp
        :return: data: time dictionary
                 duration: time estimation result

        """
        cur_time_ticket = time.time()
        if is_end:
            assert tag in data, f'record end timestamp of {tag}, but start timestamp does not exists!'
            start_time = data[tag]
            end_time = cur_time_ticket
            del data[tag]
            duration = end_time - start_time
        else:
            assert tag not in data, f'record start timestamp of {tag}, but start timestamp has existed in system!'
            data[tag] = cur_time_ticket
            duration = 0

        return duration, cur_time_ticket

    @staticmethod
    def erase_dag_ts(task: Task, is_end: bool, sub_tag: str = 'transmit') -> None:
        """
        erase task timestamp in system
        :param task: dag task
        :param is_end: if erasing the end of current timestamp
        :param sub_tag: name of time ticket
        :return: None
        """
        prefix = NameMaintainer.get_time_ticket_tag_prefix(task)
        TimeEstimator.erase_ts(task.get_tmp_data(),
                               f'{prefix}:{sub_tag}_time_{task.get_flow_index()}')
        task.erase_time_ticket_in_service(type_tag=sub_tag, is_end=is_end)

    @staticmethod
    def erase_ts(data: dict, tag: str) -> None:
        """
        erase timestamp in system
        :param data: time dictionary
        :param tag: name of time ticket
        :return: None
        """
        assert tag in data, f'erase timestamp of {tag}, but timestamp does not exists!'
        del data[tag]