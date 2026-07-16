import asyncio
import importlib
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


class FakeCleaner:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = 0
        self.stopped = []
        FakeCleaner.instances.append(self)

    def start(self):
        self.started += 1

    def stop(self, **kwargs):
        self.stopped.append(kwargs)


class FakeController:
    def __init__(self):
        self.runtime_context = SimpleNamespace(lease_ttl_seconds=3600.0)

    @staticmethod
    def check_processor_health(request=None):
        return False

    @staticmethod
    def record_transmit_ts(task, is_end=False):
        return None

    @staticmethod
    def record_execute_ts(task, is_end=False):
        return None

    @staticmethod
    def submit_task(task):
        return True

    @staticmethod
    def process_return(task):
        return True

    @staticmethod
    def clear_processor_queues(request):
        return {"ok": True, "request": request}


class FakeTask:
    @staticmethod
    def get_task_uuid():
        return "branch-1"


class FakeUploadFile:
    async def read(self):
        return b"payload"


@pytest.mark.unit
def test_controller_lifespan_only_owns_one_lease_ttl_cleaner(monkeypatch):
    module = importlib.import_module("core.controller.controller_server")
    FakeCleaner.instances = []
    monkeypatch.setattr(module, "Controller", FakeController)
    monkeypatch.setattr(module, "FileCleaner", FakeCleaner)
    monkeypatch.setattr(
        module.FileOps,
        "get_task_temp_directory",
        staticmethod(lambda: "/tmp/dayu"),
    )

    server = module.ControllerServer()
    assert FakeCleaner.instances == []
    assert not hasattr(server, "is_delete_temp_files")

    with TestClient(server.app) as client:
        assert client.post("/check").json() == {"status": "not ok"}

    assert len(FakeCleaner.instances) == 1
    cleaner = FakeCleaner.instances[0]
    assert cleaner.kwargs == {
        "folder": "/tmp/dayu",
        "poll_seconds": 30,
        "ttl_seconds": 3600.0,
        "recursive": False,
        "max_delete_per_round": 200,
    }
    assert cleaner.started == 1
    assert cleaner.stopped == [{"join": True, "timeout": 3.0}]


@pytest.mark.unit
def test_accept_task_publishes_file_and_returns_ack_without_branch_cleanup(monkeypatch):
    module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(module.ControllerServer)
    task = FakeTask()
    calls = []
    server.controller = SimpleNamespace(
        record_transmit_ts=lambda current, is_end: calls.append(("record", current, is_end)),
        submit_task=lambda current: calls.append(("submit", current)) or True,
    )
    monkeypatch.setattr(module.Task, "deserialize", staticmethod(lambda data: task))
    monkeypatch.setattr(
        module.FileOps,
        "save_task_file_in_temp",
        staticmethod(lambda current, payload: calls.append(("save", current, payload))),
    )
    monkeypatch.setattr(
        module.FileOps,
        "remove_task_file_in_temp",
        staticmethod(lambda current: (_ for _ in ()).throw(AssertionError("must not delete"))),
    )

    assert server.accept_task("serialized", b"payload") == {
        "accepted": True,
        "task_uuid": "branch-1",
    }
    assert calls == [
        ("save", task, b"payload"),
        ("record", task, True),
        ("submit", task),
    ]

    server.controller.submit_task = lambda current: False
    with pytest.raises(HTTPException) as exc_info:
        server.accept_task("serialized", b"payload")
    assert exc_info.value.status_code == 503


@pytest.mark.unit
def test_accept_result_refreshes_artifact_and_propagates_failure(monkeypatch):
    module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(module.ControllerServer)
    task = FakeTask()
    calls = []
    server.controller = SimpleNamespace(
        record_execute_ts=lambda current, is_end: calls.append(("record", current, is_end)),
        process_return=lambda current: calls.append(("return", current)) or True,
    )
    monkeypatch.setattr(module.Task, "deserialize", staticmethod(lambda data: task))
    monkeypatch.setattr(
        module.FileOps,
        "touch_task_file_in_temp",
        staticmethod(lambda current: calls.append(("touch", current)) or True),
    )

    assert server.accept_result("serialized") == {"accepted": True, "task_uuid": "branch-1"}
    assert calls == [("touch", task), ("record", task, True), ("return", task)]

    monkeypatch.setattr(module.FileOps, "touch_task_file_in_temp", staticmethod(lambda current: False))
    with pytest.raises(HTTPException) as exc_info:
        server.accept_result("serialized")
    assert exc_info.value.status_code == 503

    monkeypatch.setattr(module.FileOps, "touch_task_file_in_temp", staticmethod(lambda current: True))
    server.controller.process_return = lambda current: False
    with pytest.raises(HTTPException) as exc_info:
        server.accept_result("serialized")
    assert exc_info.value.status_code == 503


@pytest.mark.unit
def test_task_endpoints_wait_for_synchronous_ownership_ack(monkeypatch):
    module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(module.ControllerServer)
    server.accept_task = lambda data, payload: {"accepted": True, "task_uuid": "upload"}
    server.accept_result = lambda data: {"accepted": True, "task_uuid": "result"}

    assert asyncio.run(server.submit_task(FakeUploadFile(), "task")) == {
        "accepted": True,
        "task_uuid": "upload",
    }
    assert asyncio.run(server.process_return("result")) == {
        "accepted": True,
        "task_uuid": "result",
    }


@pytest.mark.unit
def test_controller_health_and_queue_clear_request_parsing():
    module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(module.ControllerServer)
    server.controller = FakeController()

    assert asyncio.run(server.check_processor_health("bad-json")) == {
        "status": "not ok",
        "error": "invalid processor health request: Expecting value: line 1 column 1 (char 0)",
    }
    assert asyncio.run(server.clear_processor_queues('{"dry_run": true}')) == {
        "ok": True,
        "request": {"dry_run": True},
    }
    assert asyncio.run(server.clear_processor_queues("[]"))["ok"] is True
    invalid = asyncio.run(server.clear_processor_queues("{"))
    assert invalid["ok"] is False
