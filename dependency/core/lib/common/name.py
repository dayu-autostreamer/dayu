class NameMaintainer:
    @staticmethod
    def get_time_ticket_tag_prefix(task):
        return f"dayu:source-{task.get_source_id()}-task-{task.get_task_id()}:{task.get_root_uuid()}"

    @staticmethod
    def get_task_data_file_name(source_id, task_id, file_suffix):
        return f'data_of_source_{source_id}_task_{task_id}.{file_suffix}'
