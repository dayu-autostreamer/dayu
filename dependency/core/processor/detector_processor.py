import numpy as np
from typing import List
import cv2

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import LOGGER, Context, convert_ndarray_to_list
from core.lib.common import ClassFactory, ClassType, FileOps


@ClassFactory.register(ClassType.PROCESSOR, alias='detector_processor')
class DetectorProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.detector = Context.get_instance('Detector')

        self.frame_size = None

    def __call__(self, task: Task):
        data_file_path = FileOps.get_task_file_in_temp(task)
        cap = cv2.VideoCapture(data_file_path)
        image_list = []
        success, frame = cap.read()
        while success:
            self.frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_list.append(frame)
            success, frame = cap.read()
        release = getattr(cap, 'release', None)
        if callable(release):
            release()

        if len(image_list) == 0:
            LOGGER.warning(f'[Image list length is 0] Source: {task.get_source_id()} '
                            f'Task: {task.get_task_id()} '
                            f'file_path: {FileOps.get_task_file_in_temp(task)}')
            return None

        result = convert_ndarray_to_list(self.infer(image_list))
        bbox_records, _ = self.detection_to_bbox_records(result)
        profile = self.make_profile(
            frame_count=len(image_list),
        )
        result = self.make_content(
            task.get_flow_index(),
            {'bbox': bbox_records},
            profile,
        )
        self.save_scenario(result, task)
        task.set_current_content(result)

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
