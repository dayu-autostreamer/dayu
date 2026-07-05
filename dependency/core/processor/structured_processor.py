import cv2

from .processor import Processor

from core.lib.content import Task
from core.lib.estimation import Timer
from core.lib.common import Context, LOGGER, ClassFactory, ClassType, FileOps, convert_ndarray_to_list


@ClassFactory.register(ClassType.PROCESSOR, alias='structured_processor')
class StructuredProcessor(Processor):
    def __init__(self):
        super().__init__()

        self.application = Context.get_instance('Application')
        self.input_services = Context.get_parameter('INPUT_SERVICES', default='[]', direct=False) or []
        if isinstance(self.input_services, str):
            self.input_services = [self.input_services]

        self.frame_size = None

    def __call__(self, task: Task):
        data_file_path = FileOps.get_task_file_in_temp(task)
        image_list = self._load_frames(data_file_path)

        if len(image_list) == 0:
            LOGGER.warning(f'[Image list length is 0] Source: {task.get_source_id()} '
                            f'Task: {task.get_task_id()} '
                            f'file_path: {data_file_path}')
            return None

        payload = {
            'task': {
                'source_id': task.get_source_id(),
                'task_id': task.get_task_id(),
                'flow_index': task.get_flow_index(),
                'metadata': task.get_metadata(),
                'raw_metadata': task.get_raw_metadata(),
                'hash_data': task.get_hash_data(),
                'file_path': data_file_path,
            },
            'frames': image_list,
            'inputs': self._collect_inputs(task),
        }

        with Timer(f'Structured Application / {task.get_flow_index()} / {len(image_list)} frame'):
            result = self.application(payload)

        result = convert_ndarray_to_list(result)
        self.save_scenario(result, task)
        task.set_current_content(result)

        return task

    def _load_frames(self, data_file_path):
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
        return image_list

    def _collect_inputs(self, task: Task):
        input_services = self.input_services or task.get_dag().get_prev_nodes(task.get_flow_index())
        inputs = {}
        for service_name in input_services:
            try:
                inputs[service_name] = task.get_service(service_name).get_content_data()
            except KeyError:
                LOGGER.warning(f'Input service "{service_name}" does not exist in task DAG.')
        return inputs

    @property
    def flops(self):
        return getattr(self.application, 'flops', 0)
