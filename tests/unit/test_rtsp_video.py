import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_rtsp_video_streams_clips_in_manifest_order(monkeypatch, tmp_path):
    rtsp_video_module = importlib.import_module("rtsp_video")

    clip_a = tmp_path / "data" / "clips" / "a.mp4"
    clip_b = tmp_path / "data" / "clips" / "b.mp4"
    clip_a.parent.mkdir(parents=True, exist_ok=True)
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")

    write_manifest(
        tmp_path / "rtsp_video" / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [
                {"path": "clips/b.mp4", "frame_count": 1},
                {"path": "clips/a.mp4", "frame_count": 1},
            ],
        },
    )

    monkeypatch.setattr(rtsp_video_module.RtspSource, "ensure_mediamtx", lambda self: None)

    commands = []

    class DummyProcess:
        def __init__(self, command):
            self.command = command

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        rtsp_video_module.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or DummyProcess(command),
    )

    source = rtsp_video_module.RtspSource(
        str(tmp_path / "rtsp_video"),
        "rtsp://127.0.0.1:8554/live",
        "non-cycle",
    )
    source.run()

    assert [Path(command[4]).name for command in commands] == ["b.mp4", "a.mp4"]


@pytest.mark.unit
def test_rtsp_video_skips_mediamtx_start_when_already_running(monkeypatch):
    rtsp_video_module = importlib.import_module("rtsp_video")

    monkeypatch.setattr(
        rtsp_video_module,
        "VideoDataset",
        lambda root: SimpleNamespace(get_clip_paths=lambda: []),
    )
    monkeypatch.setattr(
        rtsp_video_module.subprocess,
        "run",
        lambda command, stdout=None, stderr=None: SimpleNamespace(returncode=0),
    )

    popen_calls = []
    monkeypatch.setattr(
        rtsp_video_module.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append(args) or None,
    )

    source = rtsp_video_module.RtspSource("/tmp/data", "rtsp://127.0.0.1:8554/live", "non-cycle")
    source.ensure_mediamtx()

    assert popen_calls == []
    assert source.started_mediamtx is False


@pytest.mark.unit
def test_rtsp_video_cleanup_terminates_running_processes(monkeypatch):
    rtsp_video_module = importlib.import_module("rtsp_video")
    monkeypatch.setattr(
        rtsp_video_module,
        "VideoDataset",
        lambda root: SimpleNamespace(get_clip_paths=lambda: []),
    )

    class DummyProcess:
        def __init__(self, timeout=False):
            self.timeout = timeout
            self.terminated = False
            self.killed = False
            self.pid = 1234

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            if self.timeout:
                raise rtsp_video_module.subprocess.TimeoutExpired(cmd="ffmpeg", timeout=timeout)
            return 0

        def kill(self):
            self.killed = True

    current_process = DummyProcess(timeout=True)
    mediamtx_process = DummyProcess(timeout=False)

    source = rtsp_video_module.RtspSource("/tmp/data", "rtsp://127.0.0.1:8554/live", "non-cycle")
    source.current_process = current_process
    source.mediamtx_process = mediamtx_process
    source.started_mediamtx = True

    source.cleanup()

    assert current_process.terminated is True
    assert current_process.killed is True
    assert mediamtx_process.terminated is True
    assert source.running is False
    assert source.current_process is None
    assert source.mediamtx_process is None
