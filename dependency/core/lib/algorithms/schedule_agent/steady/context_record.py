import json
import os

from core.lib.common import LOGGER
from core.lib.content import Task


class ContextRecord:
    def __init__(self,
                 task: Task,
                 resource_table: dict = None):

        self.__task = task
        self.__resource_table = resource_table if resource_table else {}

    def set_task(self, task: Task):
        self.__task = task

    def set_resource_table(self, resource_table: dict):
        self.__resource_table = resource_table

    def get_task(self):
        return self.__task

    def get_resource_table(self):
        return self.__resource_table

    @staticmethod
    def serialize(context_record: 'ContextRecord'):
        return json.dumps({
            'task': Task.serialize(context_record.get_task()),
            'resource_table': context_record.get_resource_table()
        })

    @staticmethod
    def deserialize(data: str):
        data = json.loads(data)
        task = Task.deserialize(data['task'])
        context_record = ContextRecord(task=task,
                                       resource_table=data['resource_table'])
        return context_record

    @staticmethod
    def write_record(context_record: 'ContextRecord', file_path):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write('')

        json_str = ContextRecord.serialize(context_record) + '\n'
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
                        context_record = ContextRecord.deserialize(stripped_line)
                        record_list.append(context_record)
                    except json.JSONDecodeError:
                        LOGGER.warning(f"Could not decode context record JSON from line: {stripped_line}")

        return record_list
