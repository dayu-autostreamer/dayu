import json
import os

from core.lib.common import LOGGER


class CorrectRecord:

    def __init__(self,
                 task_id=None,
                 task_id_for_pre=None,
                 record_param=None
                 ):

        self.task_id = task_id
        self.task_id_for_pre = task_id_for_pre
        self.record_param = record_param

    def get_value(self):

        return {

            'task_id': self.task_id,
            'task_id_for_pre': self.task_id_for_pre,
            'record_param': self.record_param
        }

    @staticmethod
    def serialize(correct_record: 'CorrectRecord'):

        return json.dumps({
            'task_id': correct_record.task_id,
            'task_id_for_pre': correct_record.task_id_for_pre,
            'record_param': correct_record.record_param

        })

    @staticmethod
    def deserialize(data: str):

        data = json.loads(data)
        correct_record = CorrectRecord(
            task_id=data['task_id'],
            task_id_for_pre=data['task_id_for_pre'],
            record_param=data['record_param']
        )
        return correct_record

    @staticmethod
    def write_record(correct_record: 'CorrectRecord', file_path):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('')

        json_str = CorrectRecord.serialize(correct_record) + '\n'
        with open(file_path, 'a') as f:
            f.write(json_str)

    @staticmethod
    def read_record(file_path):

        record_list = []
        with open(file_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        correct_record = CorrectRecord.deserialize(stripped_line)
                        record_list.append(correct_record)
                    except json.JSONDecodeError:
                        LOGGER.warning(f"Could not decode correction record JSON from line: {stripped_line}")

        return record_list
