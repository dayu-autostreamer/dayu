import asyncio
import importlib
from types import SimpleNamespace

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
    def __init__(self):
        self.runtime_context = SimpleNamespace(lease_ttl_seconds=3600.0)

    def check_processor_health(self, request=None):
        return False

    @staticmethod
    def record_transmit_ts(task, is_end=False):
        return None

    @staticmethod
    def record_execute_ts(task, is_end=False):
        return None

    def submit_task(self, task):
        return "transmit"

    def process_return(self, task):
        return ["transmit"]


@pytest.mark.unit
def test_controller_server_initialization_defers_temp_cleanup_to_lifespan(monkeypatch):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    FakeCleaner.instances = []
    clear_calls = []

    monkeypatch.setattr(controller_server_module, "Controller", FakeController)
    monkeypatch.setattr(controller_server_module, "FileCleaner", FakeCleaner)
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "clear_task_temp_directory",
        staticmethod(lambda: clear_calls.append(True)),
    )
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "get_task_temp_directory",
        staticmethod(lambda: "/tmp/dayu/dayu"),
    )
    monkeypatch.setattr(
        controller_server_module.Context,
        "get_parameter",
        staticmethod(lambda name, default=None, direct=False: True if name == "DELETE_TEMP_FILES" else default),
    )

    server = controller_server_module.ControllerServer()

    assert server.app is not None
    assert server.is_delete_temp_files is True
    assert clear_calls == []
    assert FakeCleaner.instances == []


@pytest.mark.unit
def test_controller_server_lifespan_creates_and_stops_app_cleaner(monkeypatch):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    FakeCleaner.instances = []
    clear_calls = []

    monkeypatch.setattr(controller_server_module, "Controller", FakeController)
    monkeypatch.setattr(controller_server_module, "FileCleaner", FakeCleaner)
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "clear_task_temp_directory",
        staticmethod(lambda: clear_calls.append(True)),
    )
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "get_task_temp_directory",
        staticmethod(lambda: "/tmp/dayu/dayu"),
    )
    monkeypatch.setattr(
        controller_server_module.Context,
        "get_parameter",
        staticmethod(lambda name, default=None, direct=False: True if name == "DELETE_TEMP_FILES" else default),
    )

    server = controller_server_module.ControllerServer()

    with TestClient(server.app) as client:
        assert client.post("/check").json() == {"status": "not ok"}

    assert clear_calls == [True, True]
    assert len(FakeCleaner.instances) == 1
    assert FakeCleaner.instances[0].kwargs["folder"] == "/tmp/dayu/dayu"
    assert FakeCleaner.instances[0].kwargs["ttl_seconds"] == 3600.0
    assert FakeCleaner.instances[0].started == 1
    assert FakeCleaner.instances[0].stopped == [{"join": True, "timeout": 3.0}]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("delete_enabled", "action", "should_remove"),
    [
        (True, "transmit", True),
        (True, "execute", False),
        (False, "transmit", False),
    ],
)
def test_controller_submit_background_cleans_only_files_no_longer_needed(
    monkeypatch, delete_enabled, action, should_remove
):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(controller_server_module.ControllerServer)
    task = object()
    calls = []
    server.is_delete_temp_files = delete_enabled
    server.controller = SimpleNamespace(
        record_transmit_ts=lambda current_task, is_end: calls.append(("record", current_task, is_end)),
        submit_task=lambda current_task: action,
    )

    monkeypatch.setattr(controller_server_module.Task, "deserialize", staticmethod(lambda data: task))
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "save_task_file_in_temp",
        staticmethod(lambda current_task, data: calls.append(("save", current_task, data))),
    )
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "remove_task_file_in_temp",
        staticmethod(lambda current_task: calls.append(("remove", current_task))),
    )

    server.submit_task_background("task-data", b"file-data")

    assert ("save", task, b"file-data") in calls
    assert ("record", task, True) in calls
    assert (("remove", task) in calls) is should_remove


@pytest.mark.unit
@pytest.mark.parametrize(
    ("delete_enabled", "actions", "should_remove"),
    [
        (True, ["transmit"], True),
        (True, ["execute"], False),
        (True, ["wait"], False),
        (False, ["transmit"], False),
    ],
)
def test_controller_return_background_preserves_files_needed_by_execute_or_join(
    monkeypatch, delete_enabled, actions, should_remove
):
    controller_server_module = importlib.import_module("core.controller.controller_server")
    server = object.__new__(controller_server_module.ControllerServer)
    task = object()
    calls = []
    server.is_delete_temp_files = delete_enabled
    server.controller = SimpleNamespace(
        record_execute_ts=lambda current_task, is_end: calls.append(("record", current_task, is_end)),
        process_return=lambda current_task: actions,
    )

    monkeypatch.setattr(controller_server_module.Task, "deserialize", staticmethod(lambda data: task))
    monkeypatch.setattr(
        controller_server_module.FileOps,
        "remove_task_file_in_temp",
        staticmethod(lambda current_task: calls.append(("remove", current_task))),
    )

    server.process_return_background("task-data")

    assert ("record", task, True) in calls
    assert (("remove", task) in calls) is should_remove


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
