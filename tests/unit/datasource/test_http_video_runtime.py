import asyncio
import importlib
import runpy
import socket
import sys
from types import SimpleNamespace

from fastapi import BackgroundTasks
import pytest


@pytest.fixture
def http_video_module():
    module = importlib.import_module("http_video")
    module.sources.clear()
    return module


@pytest.mark.unit
def test_http_video_file_endpoint_registers_cleanup_task(monkeypatch, http_video_module, tmp_path):
    payload = tmp_path / "payload.mp4"
    payload.write_bytes(b"video")
    monkeypatch.setattr(http_video_module, "VideoDatasetPlayer", lambda root, mode: SimpleNamespace(is_end=False))

    source = http_video_module.VideoSource(str(tmp_path), "cycle")
    source.file_name = str(payload)

    background = BackgroundTasks()
    response = source.get_source_file(background)

    assert response.path == str(payload)
    assert len(background.tasks) == 1
    assert background.tasks[0].func is http_video_module.FileOps.remove_file
    assert background.tasks[0].args == (str(payload),)


@pytest.mark.unit
def test_http_video_admin_registration_and_remote_registration_cover_success_and_failure(monkeypatch, http_video_module):
    included = []
    warnings = []
    exceptions = []

    monkeypatch.setattr(http_video_module.app, "include_router", lambda router, prefix=None: included.append(prefix))
    monkeypatch.setattr(http_video_module, "VideoSource", lambda root, play_mode: SimpleNamespace(router=f"{root}:{play_mode}"))

    request = http_video_module.SourceRequest(root="/tmp/data", path="camera-a", play_mode="cycle")
    success = asyncio.run(http_video_module.add_source(request))
    duplicate = asyncio.run(http_video_module.add_source(request))

    assert success == {"status": "success"}
    assert duplicate == {"status": "error", "message": "Path already exists"}
    assert included == ["/camera-a"]

    http_video_module.server_port = 9100
    monkeypatch.setattr(
        http_video_module.requests,
        "post",
        lambda url, json=None: SimpleNamespace(json=lambda: {"status": "ok", "path": json["path"]}),
    )
    monkeypatch.setattr(http_video_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(http_video_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))

    http_video_module.register_source("/tmp/data", "camera-a", "cycle")
    assert warnings == []
    assert exceptions == []

    monkeypatch.setattr(
        http_video_module.requests,
        "post",
        lambda url, json=None: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    http_video_module.register_source("/tmp/data", "camera-b", "cycle")
    assert any("failed to register" in message for message in warnings)
    assert exceptions == ["network down"]


@pytest.mark.unit
def test_http_video_port_helpers_and_server_runtime_cover_socket_polling_and_uvicorn(monkeypatch, http_video_module):
    class DummySocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def connect_ex(self, address):
            return 0

    monkeypatch.setattr(http_video_module.socket, "socket", lambda *args, **kwargs: DummySocket())
    assert http_video_module.is_port_in_use(9000) is True

    port_checks = iter([False, True])
    sleeps = []
    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr(http_video_module, "is_port_in_use", lambda port: next(port_checks))
    monkeypatch.setattr(http_video_module.time, "time", lambda: next(times))
    monkeypatch.setattr(http_video_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    assert http_video_module.wait_for_port(9000, timeout=1) is True
    assert sleeps == [0.5]

    fake_loop_calls = []
    fake_server_calls = []

    class FakeLoop:
        def run_until_complete(self, result):
            fake_loop_calls.append(result)

    class FakeConfig:
        def __init__(self, app, host=None, port=None):
            self.app = app
            self.host = host
            self.port = port

    class FakeServer:
        def __init__(self, config):
            fake_server_calls.append((config.host, config.port))

        def serve(self):
            return "served"

    fake_loop = FakeLoop()
    set_loop = []
    monkeypatch.setattr(http_video_module.asyncio, "new_event_loop", lambda: fake_loop)
    monkeypatch.setattr(http_video_module.asyncio, "set_event_loop", lambda loop: set_loop.append(loop))
    monkeypatch.setattr(http_video_module.uvicorn, "Config", FakeConfig)
    monkeypatch.setattr(http_video_module.uvicorn, "Server", FakeServer)

    http_video_module.run_server(9050)

    assert set_loop == [fake_loop]
    assert fake_server_calls == [("0.0.0.0", 9050)]
    assert fake_loop_calls == ["served"]


@pytest.mark.unit
def test_http_video_wait_for_port_can_timeout(monkeypatch, http_video_module):
    monkeypatch.setattr(http_video_module, "is_port_in_use", lambda port: False)
    times = iter([0.0, 0.4, 0.8, 1.2])
    sleeps = []
    monkeypatch.setattr(http_video_module.time, "time", lambda: next(times))
    monkeypatch.setattr(http_video_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert http_video_module.wait_for_port(9001, timeout=1) is False
    assert sleeps == [0.5, 0.5]


@pytest.mark.unit
def test_http_video_module_entrypoint_covers_existing_and_new_server_paths(monkeypatch):
    register_calls = []
    thread_events = []

    class ExistingSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def connect_ex(self, address):
            return 0

    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: ExistingSocket())
    monkeypatch.setattr(
        importlib.import_module("requests"),
        "post",
        lambda url, json=None: SimpleNamespace(json=lambda: {"status": "ok", "path": json["path"]}),
    )
    monkeypatch.setattr(sys, "argv", ["http_video.py", "--root", "/tmp/data", "--address", "http://127.0.0.1:9000/camera-a", "--play_mode", "cycle"])
    runpy.run_module("http_video", run_name="__main__")

    class StartupSocket:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def connect_ex(self, address):
            StartupSocket.calls += 1
            return 1 if StartupSocket.calls == 1 else 0

    class DummyThread:
        def __init__(self, target=None, args=None, daemon=None):
            self.args = args or ()
            self.daemon = daemon

        def start(self):
            thread_events.append(("start", self.args, self.daemon))

        def join(self):
            thread_events.append(("join", self.args, self.daemon))

    times = iter([0.0, 0.1, 0.2])
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: StartupSocket())
    monkeypatch.setattr(importlib.import_module("threading"), "Thread", DummyThread)
    monkeypatch.setattr(importlib.import_module("time"), "time", lambda: next(times))
    monkeypatch.setattr(importlib.import_module("time"), "sleep", lambda seconds: None)
    monkeypatch.setattr(
        importlib.import_module("requests"),
        "post",
        lambda url, json=None: register_calls.append((url, json)) or SimpleNamespace(json=lambda: {"status": "ok"}),
    )
    monkeypatch.setattr(sys, "argv", ["http_video.py", "--root", "/tmp/data", "--address", "http://127.0.0.1:9100/camera-b", "--play_mode", "cycle"])
    runpy.run_module("http_video", run_name="__main__")

    assert register_calls[-1][1]["path"] == "camera-b"
    assert thread_events == [("start", (9100,), True), ("join", (9100,), True)]
