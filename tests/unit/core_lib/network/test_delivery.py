import importlib

import pytest


class FakeTask:
    def __init__(self, task_uuid="branch-1"):
        self.task_uuid = task_uuid

    def get_task_uuid(self):
        return self.task_uuid

    @staticmethod
    def get_source_id():
        return 0

    @staticmethod
    def get_task_id():
        return 7

    @staticmethod
    def get_flow_index():
        return "detector"

    @staticmethod
    def get_file_path():
        return "payload.mp4"

    def serialize(self):
        return f"serialized:{self.task_uuid}"


@pytest.mark.unit
def test_task_ack_and_delivery_require_exact_identity(monkeypatch):
    delivery = importlib.import_module("core.lib.network.delivery")
    task = FakeTask()

    assert delivery.task_ack(task) == {"accepted": True, "task_uuid": "branch-1"}

    responses = iter([
        None,
        {"accepted": True, "task_uuid": "other-branch"},
        {"accepted": True, "task_uuid": "branch-1"},
    ])
    monkeypatch.setattr(delivery, "http_request", lambda **kwargs: next(responses))

    assert delivery.deliver_task(url="http://receiver", method="POST", task=task) is False
    assert delivery.deliver_task(url="http://receiver", method="POST", task=task) is False
    assert delivery.deliver_task(url="http://receiver", method="POST", task=task) is True


@pytest.mark.unit
def test_persistent_file_delivery_replays_identical_bytes_until_ack(monkeypatch, tmp_path):
    delivery = importlib.import_module("core.lib.network.delivery")
    task = FakeTask()
    payload = tmp_path / "payload.mp4"
    payload.write_bytes(b"complete-video")
    requests = []
    payloads = []
    file_handles = []
    sleeps = []

    def fake_request(**kwargs):
        requests.append(kwargs)
        file_handle = kwargs["files"]["file"][1]
        file_handles.append(file_handle)
        payloads.append(file_handle.read())
        if len(requests) == 3:
            return delivery.task_ack(task)
        return None

    monkeypatch.setattr(delivery, "http_request", fake_request)
    monkeypatch.setattr(delivery.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert delivery.deliver_task(
        url="http://receiver",
        method="POST",
        task=task,
        file_path=payload,
        persistent=True,
    ) is True

    assert payloads == [
        b"complete-video",
        b"complete-video",
        b"complete-video",
    ]
    assert len({id(file_handle) for file_handle in file_handles}) == 3
    assert all(file_handle.closed for file_handle in file_handles)
    assert all(request["retry"] == 1 for request in requests)
    assert sleeps == [0.5, 0.5]


@pytest.mark.unit
def test_delivery_rejects_ambiguous_file_payload(tmp_path):
    delivery = importlib.import_module("core.lib.network.delivery")
    payload = tmp_path / "payload.mp4"
    payload.write_bytes(b"payload")

    with pytest.raises(ValueError, match="either file_path or file_content"):
        delivery.deliver_task(
            url="http://receiver",
            method="POST",
            task=FakeTask(),
            file_path=payload,
            file_content=b"payload",
        )
