import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskIdentity:
    """Identity reserved before source data is materialized for one root task."""

    source_id: int
    task_id: int
    task_uuid: str
    root_uuid: str

    @classmethod
    def create(cls, source_id, task_id):
        task_uuid = str(uuid.uuid4())
        return cls(
            source_id=int(source_id),
            task_id=int(task_id),
            task_uuid=task_uuid,
            root_uuid=task_uuid,
        )

    def to_dict(self):
        return {
            'source_id': self.source_id,
            'task_id': self.task_id,
            'task_uuid': self.task_uuid,
            'root_uuid': self.root_uuid,
        }
