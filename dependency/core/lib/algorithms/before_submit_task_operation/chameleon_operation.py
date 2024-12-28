import abc

from .base_operation import BaseBSTOperation

from core.lib.common import ClassFactory, ClassType, EncodeOps

__all__ = ('ChameleonBSTOperation',)


@ClassFactory.register(ClassType.GEN_BSTO, alias='chameleon')
class ChameleonBSTOperation(BaseBSTOperation, abc.ABC):
    def __init__(self):
        pass

    def __call__(self, system, compressed_file, hash_codes):
        import cv2
        cap = cv2.VideoCapture(compressed_file)
        ret, frame = cap.read()
        if ret:
            system.temp_encoded_frame = EncodeOps.encode_image(frame)
            system.temp_hash_code = hash_codes[0]
        else:
            system.temp_encoded_frame = ''
