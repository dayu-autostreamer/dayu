import abc
import os
from .base_operation import BaseBSTOperation
from core.lib.common import ClassFactory, ClassType, EncodeOps
from core.lib.content import Task

__all__ = ('SteadyBSTOperation',)

@ClassFactory.register(ClassType.GEN_BSTO, alias='steady')
class SteadyBSTOperation(BaseBSTOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, new_task:Task):
        # 感知文件大小
        task = system.current_task
        tmp_data = task.get_tmp_data()
        compressed_file = new_task.get_file_path()
        file_size = os.path.getsize(compressed_file) / 1024 / 1024
        tmp_data['file_size'] = file_size
        task.set_tmp_data(tmp_data)


