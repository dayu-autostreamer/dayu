import cv2
from typing import List

from .processor import Processor

from core.lib.estimation import Timer
from core.lib.content import Task
from core.lib.common import Context, LOGGER, ClassFactory, ClassType, FileOps


@ClassFactory.register(ClassType.PROCESSOR, alias='roi_classifier_processor')
class RoiClassifierProcessor(Processor):
    def __init__(self):
        super().__init__()
        # Expect applications to provide a ROI-aware Classifier under the name 'Classifier' when PROCESSOR_NAME is set
        self.classifier = Context.get_instance('Roi_Classifier')

    def __call__(self, task: Task):
        data_file_path = FileOps.get_task_file_in_temp(task)
        content = task.get_prev_content()
        image_list = self._load_frames(data_file_path)
        if content is None:
            LOGGER.warning(f'content of source {task.get_source_id()} task {task.get_task_id()} is none!')
            result = self._empty_content(task, len(image_list))
            self.save_scenario(result, task)
            task.set_current_content(result)
            return task

        # Reset classifier cache at the beginning of each task to avoid cross-task roi_id collision
        self.classifier.reset_cache()

        output_records: List[dict] = []
        for bbox_record in self.output_records(content, 'bbox'):
            frame_index = int(bbox_record.get('frame_index', 0))
            frame = image_list[frame_index] if 0 <= frame_index < len(image_list) else None
            source_items = bbox_record.get('items') or []
            if frame is None:
                output_records.append({'frame_index': frame_index, 'items': []})
                continue
            valid_entries = []
            for index, source_item in enumerate(source_items):
                crop = self._crop(frame, source_item.get('bbox', []))
                if crop is not None:
                    valid_entries.append((index, crop, source_item.get('object_id', index)))
            if valid_entries:
                with Timer(f'ROI Classification / {len(valid_entries)} bboxes'):
                    labels = self.classifier(
                        [crop for _, crop, _ in valid_entries],
                        [self._roi_id(roi_id, index) for index, _, roi_id in valid_entries],
                    )
            else:
                labels = []
            labels_by_index = {
                source_index: labels[label_index]
                for label_index, (source_index, _, _) in enumerate(valid_entries)
                if label_index < len(labels)
            }
            items = []
            for index, source_item in enumerate(source_items):
                label = labels_by_index.get(index, '')
                items.append(self._text_item(label, source_item))
            output_records.append({
                'frame_index': frame_index,
                'items': items,
            })

        profile = self.make_profile(
            frame_count=len(image_list),
        )
        result = self.make_content(task.get_flow_index(), {'text': output_records}, profile)
        self.save_scenario(result, task)
        task.set_current_content(result)
        return task

    @property
    def flops(self):
        return self.classifier.flops

    @staticmethod
    def _load_frames(data_file_path):
        cap = cv2.VideoCapture(data_file_path)
        image_list = []
        success, frame = cap.read()
        while success:
            image_list.append(frame)
            success, frame = cap.read()
        release = getattr(cap, 'release', None)
        if callable(release):
            release()
        return image_list

    def _empty_content(self, task, frame_count):
        profile = self.make_profile(
            frame_count=frame_count,
        )
        return self.make_content(task.get_flow_index(), {'text': []}, profile)

    @staticmethod
    def _crop(frame, bbox):
        if frame is None or len(bbox) != 4:
            return None
        height, width, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox
        x_min = int(max(x_min, 0))
        y_min = int(max(y_min, 0))
        x_max = int(min(width, x_max))
        y_max = int(min(height, y_max))
        if x_max <= x_min or y_max <= y_min:
            return None
        return frame[y_min:y_max, x_min:x_max]

    @staticmethod
    def _text_item(label, source_item):
        if isinstance(label, dict):
            text = label.get('text') or label.get('label') or label.get('class') or ''
            score = label.get('score', label.get('confidence', None))
        else:
            text = label
            score = None
        item = {
            'text': str(text),
            'source_object_id': source_item.get('object_id'),
            'bbox': source_item.get('bbox', []),
        }
        if score is not None:
            item['score'] = float(score)
        return item

    @staticmethod
    def _roi_id(value, fallback):
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(fallback)
