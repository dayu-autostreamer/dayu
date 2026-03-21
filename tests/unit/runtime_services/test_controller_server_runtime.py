import asyncio
import importlib

import pytest
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
    def check_processor_health(self):
        return False

    @staticmethod
    def record_transmit_ts(task, is_end=False):
        return None

    @staticmethod
    def record_execute_ts(task, is_end=False):
        return None

    def submit_task(self, task):
        return None

    def process_return(self, task):
        return None


@pytest.mark.unit
def test_controller_server_initialization_starts_instance_cleaner_when_delete_temp_files_is_enabled(monkeypatch):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    FakeCleaner.instances = []
    clear_calls = []

    monkeypatch.setattr(controller_server_module, "Controller", FakeController)
    monkeypatch.setattr(controller_server_module, "FileCleaner", FakeCleaner)
    monkeypatch.setattr(controller_server_module.FileOps, "clear_temp_directory", staticmethod(lambda: clear_calls.append(True)))
    monkeypatch.setattr(
        controller_server_module.Context,
        "get_parameter",
        staticmethod(lambda name, default=None, direct=False: True if name == "DELETE_TEMP_FILES" else default),
    )
    monkeypatch.setattr(
        controller_server_module.Context,
        "get_temporary_file_path",
        staticmethod(lambda suffix: f"/tmp/dayu/{suffix}"),
    )

    server = controller_server_module.ControllerServer()

    assert clear_calls == [True]
    assert server.is_delete_temp_files is True
    assert len(FakeCleaner.instances) == 1
    assert FakeCleaner.instances[0].kwargs["folder"] == "/tmp/dayu/"
    assert FakeCleaner.instances[0].started == 1


@pytest.mark.unit
def test_controller_server_lifespan_creates_and_stops_app_cleaner(monkeypatch):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    FakeCleaner.instances = []
    clear_calls = []
    delete_flags = iter([False, True])

    monkeypatch.setattr(controller_server_module, "Controller", FakeController)
    monkeypatch.setattr(controller_server_module, "FileCleaner", FakeCleaner)
    monkeypatch.setattr(controller_server_module.FileOps, "clear_temp_directory", staticmethod(lambda: clear_calls.append(True)))
    monkeypatch.setattr(
        controller_server_module.Context,
        "get_parameter",
        staticmethod(lambda name, default=None, direct=False: next(delete_flags) if name == "DELETE_TEMP_FILES" else default),
    )
    monkeypatch.setattr(
        controller_server_module.Context,
        "get_temporary_file_path",
        staticmethod(lambda suffix: f"/tmp/dayu/{suffix}"),
    )

    server = controller_server_module.ControllerServer()

    with TestClient(server.app) as client:
        assert client.post("/check").json() == {"status": "not ok"}

    assert clear_calls == [True, True, True]
    assert len(FakeCleaner.instances) == 1
    assert FakeCleaner.instances[0].started == 1
    assert FakeCleaner.instances[0].stopped == [{"join": True, "timeout": 3.0}]


@pytest.mark.unit
def test_controller_server_endpoints_enqueue_background_handlers(monkeypatch):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(controller_server_module.ControllerServer)
    enqueued = []

    class FakeBackgroundTasks:
        def add_task(self, func, *args):
            enqueued.append((func.__name__, args))

    class FakeUploadFile:
        async def read(self):
            return b"payload"

    server.submit_task_background = lambda data, file_data: None
    server.process_return_background = lambda data: None

    asyncio.run(server.submit_task(FakeBackgroundTasks(), FakeUploadFile(), "serialized-task"))
    asyncio.run(server.process_return(FakeBackgroundTasks(), "serialized-return"))

    assert enqueued == [
        ("<lambda>", ("serialized-task", b"payload")),
        ("<lambda>", ("serialized-return",)),
    ]
