import abc
import os


from .base_operation import BaseBSTOperation

from core.lib.common import ClassFactory, ClassType


__all__ = ('CEVASBSTOperation',)


@ClassFactory.register(ClassType.GEN_BSTO, alias='cevas')
class CEVASBSTOperation(BaseBSTOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, new_task):
        task = system.current_task
        tmp_data = task.get_tmp_data()

        compressed_file = new_task.get_file_path()
        file_size = os.path.getsize(compressed_file)/1024
        tmp_data['file_size'] = file_size
        task.set_tmp_data(tmp_data)
