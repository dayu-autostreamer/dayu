import asyncio
import importlib
import json
from pathlib import Path
import runpy
import socket
import sys
import threading
from types import SimpleNamespace

from fastapi import BackgroundTasks
import pytest

from core.lib.common import FileNotMountedError


@pytest.fixture
def http_video_module(monkeypatch, tmp_path):
    monkeypatch.setenv("TEMP_PATH", str(tmp_path))
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
def test_http_video_requires_explicit_temporary_storage(
    monkeypatch,
    http_video_module,
    tmp_path,
):
    monkeypatch.delenv("TEMP_PATH")
    monkeypatch.setattr(
        http_video_module,
        "VideoDatasetPlayer",
        lambda root, mode: SimpleNamespace(is_end=False),
    )
    source = http_video_module.VideoSource(str(tmp_path), "non-cycle")
    source.source_id = 314159
    source.task_id = 2718
    source.meta_data = {"resolution": "720p"}
    source.raw_meta_data = {"resolution": "1080p"}
    source.frame_process = lambda system, frame, source_res, target_res: frame
    source.frame_compress = lambda system, frames, file_name: Path(file_name).write_text(
        frames[0],
        encoding="utf-8",
    )
    monkeypatch.setattr(source, "_configure_request", lambda data: None)
    monkeypatch.setattr(source, "_select_frames", lambda: (["frame"], [1]))
    cwd_artifact = Path.cwd() / "data_of_source_314159_task_2718.mp4"

    assert not cwd_artifact.exists()
    with pytest.raises(FileNotMountedError, match="Temporary directory is not mounted"):
        source.get_source_data("{}")
    assert not cwd_artifact.exists()


@pytest.mark.unit
def test_http_video_status_identifies_source_instance_and_exhaustion(
    monkeypatch, http_video_module, tmp_path
):
    player = SimpleNamespace(is_end=False)
    monkeypatch.setattr(
        http_video_module,
        "VideoDatasetPlayer",
        lambda root, mode: player,
    )

    first = http_video_module.VideoSource(str(tmp_path), "non-cycle")
    second = http_video_module.VideoSource(str(tmp_path), "non-cycle")

    first_status = first.get_source_status()
    second_status = second.get_source_status()
    assert first_status == {
        "instance_id": first.instance_id,
        "exhausted": False,
        "ready": True,
    }
    assert second_status["instance_id"] != first_status["instance_id"]

    player.is_end = True
    assert first.get_source_status() == {
        "instance_id": first.instance_id,
        "exhausted": True,
        "ready": True,
    }


@pytest.mark.unit
def test_http_video_on_demand_processing_preserves_segment_order(
    monkeypatch, http_video_module, tmp_path
):
    monkeypatch.chdir(tmp_path)
    class Player:
        def __init__(self):
            self.frames = iter(
                [
                    ("frame-0", 100),
                    ("frame-1", 101),
                    ("frame-2", 102),
                    (None, None),
                ]
            )
            self.is_end = False

        def read_frame(self):
            frame, index = next(self.frames)
            if frame is None:
                self.is_end = True
            return frame, index

    monkeypatch.setattr(
        http_video_module,
        "VideoDatasetPlayer",
        lambda root, mode: Player(),
    )

    def algorithm(kind, al_name=None):
        if kind == "GEN_FILTER":
            return lambda system, frame: True
        if kind == "GEN_PROCESS":
            return lambda system, frame, source, target: frame

        def compress(system, frames, file_name):
            Path(file_name).write_text(",".join(frames), encoding="utf-8")

        return compress

    monkeypatch.setattr(
        http_video_module.Context,
        "get_algorithm",
        staticmethod(algorithm),
    )
    source = http_video_module.VideoSource(str(tmp_path), "non-cycle")

    def request(task_id):
        return json.dumps(
            {
                "source_id": 7,
                "task_id": task_id,
                "meta_data": {
                    "buffer_size": 1,
                    "resolution": "720p",
                    "fps": 16,
                    "encoding": "mp4v",
                },
                "raw_meta_data": {"resolution": "1080p", "fps": 30},
                "gen_filter_name": "simple",
                "gen_process_name": "simple",
                "gen_compress_name": "simple",
            }
        )

    observed = []
    payloads = []
    for task_id in (1, 2, 3):
        response = source.get_source_data(request(task_id))
        observed.append(json.loads(response.body))
        payloads.append(Path(source.file_name).read_text(encoding="utf-8"))

    exhausted = source.get_source_data(request(4))

    assert observed == [[100], [101], [102]]
    assert payloads == ["frame-0", "frame-1", "frame-2"]
    assert json.loads(exhausted.body) == []
    assert source.get_source_status()["exhausted"] is True
    assert source.get_source_status()["ready"] is True


@pytest.mark.unit
def test_http_video_source_exhaustion_is_stable_and_side_effect_free(
    monkeypatch,
    http_video_module,
    tmp_path,
):
    player = SimpleNamespace(is_end=True)
    monkeypatch.setattr(
        http_video_module,
        "VideoDatasetPlayer",
        lambda root, mode: player,
    )
    source = http_video_module.VideoSource(str(tmp_path), "non-cycle")
    monkeypatch.setattr(
        source,
        "_configure_request",
        lambda data: pytest.fail("an exhausted source must not consume a request"),
    )

    for task_id in (1, 2):
        assert source.get_source_data(json.dumps({"task_id": task_id})) == []

    assert source.get_source_status() == {
        "instance_id": source.instance_id,
        "exhausted": True,
        "ready": True,
    }


@pytest.mark.unit
def test_http_video_source_serializes_stateful_source_requests(
    monkeypatch,
    http_video_module,
    tmp_path,
):
    monkeypatch.setattr(
        http_video_module,
        "VideoDatasetPlayer",
        lambda root, mode: SimpleNamespace(is_end=False),
    )
    source = http_video_module.VideoSource(str(tmp_path), "non-cycle")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    events = []
    responses = {}
    generated_paths = []

    def compress(system, frames, file_name):
        events.append(("compress", system.task_id))
        generated_paths.append(Path(file_name))
        Path(file_name).write_text(frames[0], encoding="utf-8")

    def configure(data):
        task_id = int(data["task_id"])
        events.append(("configure", task_id))
        if task_id == 1:
            first_entered.set()
            release_first.wait(timeout=1.0)
        else:
            second_entered.set()
        source.source_id = int(data["source_id"])
        source.task_id = task_id
        source.meta_data = dict(data["meta_data"])
        source.raw_meta_data = dict(data["raw_meta_data"])
        source.frame_process = lambda system, frame, source_res, target_res: frame
        source.frame_compress = compress

    def select_frames():
        events.append(("select", source.task_id))
        return [f"frame-{source.task_id}"], [source.task_id]

    monkeypatch.setattr(source, "_configure_request", configure)
    monkeypatch.setattr(source, "_select_frames", select_frames)

    def request(task_id):
        return json.dumps({
            "source_id": 7,
            "task_id": task_id,
            "meta_data": {
                "buffer_size": 1,
                "resolution": "720p",
            },
            "raw_meta_data": {"resolution": "1080p"},
        })

    def call(task_id):
        if task_id == 2:
            second_started.set()
        response = source.get_source_data(request(task_id))
        responses[task_id] = json.loads(response.body)

    first = threading.Thread(target=call, args=(1,))
    second = threading.Thread(target=call, args=(2,))
    first.start()
    assert first_entered.wait(timeout=1.0)
    second.start()
    assert second_started.wait(timeout=1.0)
    assert second_entered.wait(timeout=0.05) is False

    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert first.is_alive() is False
    assert second.is_alive() is False
    assert responses == {1: [1], 2: [2]}
    assert events == [
        ("configure", 1),
        ("select", 1),
        ("compress", 1),
        ("configure", 2),
        ("select", 2),
        ("compress", 2),
    ]
    assert generated_paths == [
        tmp_path / "data_of_source_7_task_1.mp4",
        tmp_path / "data_of_source_7_task_2.mp4",
    ]


@pytest.mark.unit
def test_http_video_applies_each_task_configuration_after_scheduling(
    monkeypatch, http_video_module, tmp_path
):
    events = []
    generated_paths = []

    class Player:
        def __init__(self):
            self.index = 0
            self.is_end = False

        def read_frame(self):
            if self.index >= 3:
                self.is_end = True
                return None, None
            current = self.index
            self.index += 1
            events.append(("decode", current))
            return f"frame-{current}", current

    monkeypatch.setattr(
        http_video_module,
        "VideoDatasetPlayer",
        lambda root, mode: Player(),
    )

    def algorithm(kind, al_name=None):
        if kind == "GEN_FILTER":
            return lambda system, frame: True
        if kind == "GEN_PROCESS":
            return lambda system, frame, source, target: (
                events.append(("process", frame, target)) or frame
            )

        def compress(system, frames, file_name):
            events.append(("encode", int(frames[0].split("-")[1])))
            generated_paths.append(Path(file_name))
            Path(file_name).write_text(frames[0], encoding="utf-8")

        return compress

    monkeypatch.setattr(
        http_video_module.Context,
        "get_algorithm",
        staticmethod(algorithm),
    )
    source = http_video_module.VideoSource(str(tmp_path), "non-cycle")
    def request(task_id, buffer_size, resolution):
        return json.dumps({
            "source_id": 9,
            "task_id": task_id,
            "meta_data": {
                "buffer_size": buffer_size,
                "resolution": resolution,
                "fps": 16,
                "encoding": "mp4v",
            },
            "raw_meta_data": {"resolution": "1080p", "fps": 30},
            "gen_filter_name": "simple",
            "gen_process_name": "simple",
            "gen_compress_name": "simple",
        })

    first = source.get_source_data(request(1, 1, "720p"))
    second = source.get_source_data(request(2, 2, "480p"))

    assert json.loads(first.body) == [0]
    assert json.loads(second.body) == [1, 2]
    assert [event for event in events if event[0] == "process"] == [
        ("process", "frame-0", "720p"),
        ("process", "frame-1", "480p"),
        ("process", "frame-2", "480p"),
    ]
    assert generated_paths == [
        tmp_path / "data_of_source_9_task_1.mp4",
        tmp_path / "data_of_source_9_task_2.mp4",
    ]
    assert source.get_source_status()["ready"] is True


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
