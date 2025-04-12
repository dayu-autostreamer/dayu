import abc

from .base_operation import BaseBSTOperation

from core.lib.common import ClassFactory, ClassType, EncodeOps
from core.lib.content import Task

__all__ = ('ChameleonBSTOperation',)


@ClassFactory.register(ClassType.GEN_BSTO, alias='chameleon')
class ChameleonBSTOperation(BaseBSTOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, new_task:Task):
        import cv2
        compressed_file = new_task.get_file_path()
        hash_codes = new_task.get_hash_data()

        cap = cv2.VideoCapture(compressed_file)
        ret, frame = cap.read()
        if ret:
            system.temp_encoded_frame = EncodeOps.encode_image(frame)
            system.temp_hash_code = hash_codes[0]
        else:
            system.temp_encoded_frame = ''
