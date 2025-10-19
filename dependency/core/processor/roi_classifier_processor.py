import cv2
from typing import List

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import Context, LOGGER, ClassFactory, ClassType


@ClassFactory.register(ClassType.PROCESSOR, alias='roi_classifier_processor')
class RoiClassifierProcessor(Processor):
    def __init__(self):
        super().__init__()
        # Expect applications to provide a ROI-aware Classifier under the name 'Classifier' when PROCESSOR_NAME is set
        self.classifier = Context.get_instance('Roi_Classifier')

    def __call__(self, task: Task):
        data_file_path = task.get_file_path()
        cap = cv2.VideoCapture(data_file_path)
        content = task.get_prev_content()
        if content is None:
            LOGGER.warning(f'content of source {task.get_source_id()} task {task.get_task_id()} is none!')
            return task
        # Reset classifier cache at the beginning of each task to avoid cross-task roi_id collision
        if hasattr(self.classifier, 'reset_cache'):
            try:
                self.classifier.reset_cache()
            except Exception:
                pass
        content_output: List[list] = []
        try:
            for bbox, prob, class_id, roi_id in content:
                ret, frame = cap.read()
                if not ret or frame is None:
                    content_output.append([[]])
                    continue
                height, width, _ = frame.shape
                rois = []
                roi_ids = []
                for (x_min, y_min, x_max, y_max), rid in zip(bbox, roi_id):
                    x_min = int(max(x_min, 0))
                    y_min = int(max(y_min, 0))
                    x_max = int(min(width, x_max))
                    y_max = int(min(height, y_max))
                    rois.append(frame[y_min:y_max, x_min:x_max])
                    roi_ids.append(int(rid))
                with Timer(f'ROI Classification / {len(rois)} bboxes'):
                    result = self.classifier(rois, roi_ids)
                content_output.append([result])
        except Exception as e:
            LOGGER.exception(e)

        task.set_current_content(content_output)
        return task

    @property
    def flops(self):
        return self.classifier.flops
