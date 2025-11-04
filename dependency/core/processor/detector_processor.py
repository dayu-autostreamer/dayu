import numpy as np
from typing import List
import cv2

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import LOGGER, Context, convert_ndarray_to_list
from core.lib.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.PROCESSOR, alias='detector_processor')
class DetectorProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.detector = Context.get_instance('Detector')

        self.frame_size = None

    def __call__(self, task: Task):
        data_file_path = Context.get_temporary_file_path(task.get_file_path())
        cap = cv2.VideoCapture(data_file_path)
        image_list = []
        success, frame = cap.read()
        while success:
            self.frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_list.append(frame)
            success, frame = cap.read()

        if len(image_list) == 0:
            LOGGER.critical('ERROR: image list length is 0')
            LOGGER.critical(f'Source: {task.get_source_id()}, Task: {task.get_task_id()}')
            LOGGER.critical(f'file_path: {Context.get_temporary_file_path(task.get_file_path())}')
            return None
        result = self.infer(image_list)
        task = self.get_scenario(result, task)
        task.set_current_content(convert_ndarray_to_list(result))

        return task

    def infer(self, images: List[np.ndarray]):
        assert self.detector, 'No detector defined!'

        LOGGER.debug(f'[Batch Size] Car detection batch: {len(images)}')

        with Timer(f'Detection / {len(images)} frame'):
            process_output = self.detector(images)

        return process_output

    @property
    def flops(self):
        return self.detector.flops
