import cv2

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import Context, LOGGER, ClassFactory, ClassType


@ClassFactory.register(ClassType.PROCESSOR, alias='classifier_processor')
class ClassifierProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.classifier = Context.get_instance('Classifier')

    def __call__(self, task: Task):
        data_file_path = task.get_file_path()
        cap = cv2.VideoCapture(data_file_path)
        content = task.get_prev_content()
        if content is None:
            LOGGER.warning(f'content of source {task.get_source_id()} task {task.get_task_id()} is none!')
            return task
        content_output = []
        try:
            for bbox, prob, class_id in content:
                ret, frame = cap.read()
                height, width, _ = frame.shape
                if ret:
                    faces = []
                    for x_min, y_min, x_max, y_max in bbox:
                        x_min = int(max(x_min, 0))
                        y_min = int(max(y_min, 0))
                        x_max = int(min(width, x_max))
                        y_max = int(min(height, y_max))
                        faces.append(frame[y_min:y_max, x_min:x_max])
                    with Timer(f'Classification / {len(faces)} bboxes'):
                        result = self.classifier(faces)
                else:
                    result = []
                content_output.append([result])
        except Exception as e:
            pass

        task.set_current_content(content_output)

        return task
