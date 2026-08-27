from core.lib.content import Task
from core.lib.common import Context


class Processor:
    profile_fields = {
        'frame_count',
    }

    def __init__(self):
        self.scenario_extractors_text = Context.get_parameter('SCENARIOS_EXTRACTORS', direct=False)

        self.scenario_extractors = []
        for scenario_extractor_text in self.scenario_extractors_text:
            self.scenario_extractors.append(
                Context.get_algorithm('PRO_SCENARIO', scenario_extractor_text)
            )

    def __call__(self, task: Task):
        raise NotImplementedError

    @property
    def flops(self):
        raise NotImplementedError

    def save_scenario(self, result, task):
        scenarios = {}

        for scenario_extractor_text, scenario_extractor in zip(self.scenario_extractors_text, self.scenario_extractors):
            scenarios.update({scenario_extractor_text: scenario_extractor(result, task)})

        task.add_scenario(scenarios)

    @staticmethod
    def make_profile(frame_count=0):
        return {
            'frame_count': int(frame_count),
        }

    @staticmethod
    def make_content(service_name, outputs, profile):
        content = {
            'service': service_name,
            'outputs': outputs,
            'profile': profile,
        }
        Processor.validate_content(content)
        return content

    @staticmethod
    def output_records(content, label):
        Processor.validate_content(content)
        outputs = content['outputs']
        records = outputs.get(label, [])
        if not isinstance(records, list):
            raise ValueError(f"Processor content output '{label}' must be a list of records")
        return records

    @staticmethod
    def validate_content(content):
        if not isinstance(content, dict):
            raise ValueError('Processor content must be a dictionary')
        for key in ('service', 'outputs', 'profile'):
            if key not in content:
                raise ValueError(f"Processor content missing required field '{key}'")
        Processor.validate_outputs(content.get('outputs'))
        profile = content.get('profile')
        if not isinstance(profile, dict):
            raise ValueError("Processor content field 'profile' must be a dictionary")
        if set(profile) != Processor.profile_fields:
            raise ValueError(f'Processor content profile fields must be {sorted(Processor.profile_fields)}')

    @staticmethod
    def validate_outputs(outputs):
        if not isinstance(outputs, dict):
            raise ValueError("Processor outputs must be a dictionary")
        for output_label, records in outputs.items():
            if not isinstance(output_label, str) or not output_label:
                raise ValueError("Processor content output labels must be non-empty strings")
            if not isinstance(records, list):
                raise ValueError(f"Processor content output '{output_label}' must be a list of records")
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(f"Processor content output '{output_label}' records must be dictionaries")
                if 'frame_index' not in record:
                    raise ValueError(f"Processor content output '{output_label}' records require 'frame_index'")
                if not isinstance(record.get('items'), list):
                    raise ValueError(f"Processor content output '{output_label}' records require list 'items'")

    @staticmethod
    def output_items(content, label):
        items = []
        for record in Processor.output_records(content, label):
            if isinstance(record, dict):
                record_items = record.get('items') or []
                if isinstance(record_items, list):
                    items.extend(record_items)
        return items

    @staticmethod
    def detection_to_bbox_records(result):
        records = []
        total = 0
        for frame_index, frame_result in enumerate(result or []):
            if len(frame_result) < 4:
                bboxes, scores, labels, object_ids = [], [], [], []
            else:
                bboxes, scores, labels, object_ids = frame_result[:4]
            frame_items = []
            for index, bbox in enumerate(bboxes):
                score = scores[index] if index < len(scores) else 0.0
                label = labels[index] if index < len(labels) else ''
                object_id = object_ids[index] if index < len(object_ids) else index
                frame_items.append({
                    'bbox': [int(round(value)) for value in bbox],
                    'score': float(score),
                    'label': str(label),
                    'object_id': object_id,
                })
            total += len(frame_items)
            records.append({
                'frame_index': frame_index,
                'items': frame_items,
            })
        return records, total
