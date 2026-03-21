import importlib
import runpy
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.unit
def test_rtsp_video_starts_mediamtx_when_missing(monkeypatch):
    rtsp_video_module = importlib.import_module("rtsp_video")
    monkeypatch.setattr(
        rtsp_video_module,
        "VideoDataset",
        lambda root: SimpleNamespace(get_clip_paths=lambda: []),
    )
    monkeypatch.setattr(
        rtsp_video_module.subprocess,
        "run",
        lambda command, stdout=None, stderr=None: SimpleNamespace(returncode=1),
    )

    popen_calls = []
    sleep_calls = []
    fake_process = SimpleNamespace(pid=4321, poll=lambda: None)

    monkeypatch.setattr(rtsp_video_module.os, "getenv", lambda key, default=None: "/custom/rtsp")
    monkeypatch.setattr(
        rtsp_video_module.subprocess,
        "Popen",
        lambda command, stdout=None, stderr=None: popen_calls.append(command) or fake_process,
    )
    monkeypatch.setattr(rtsp_video_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    source = rtsp_video_module.RtspSource("/tmp/data", "rtsp://127.0.0.1:8554/live", "non-cycle")
    source.ensure_mediamtx()

    assert popen_calls == [["/custom/rtsp/mediamtx", "/custom/rtsp/mediamtx.yml"]]
    assert sleep_calls == [4]
    assert source.started_mediamtx is True
    assert source.mediamtx_process is fake_process


@pytest.mark.unit
def test_rtsp_video_stream_once_covers_running_guard_and_ffmpeg_warning(monkeypatch):
    rtsp_video_module = importlib.import_module("rtsp_video")
    warnings = []
    monkeypatch.setattr(rtsp_video_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(
        rtsp_video_module,
        "VideoDataset",
        lambda root: SimpleNamespace(get_clip_paths=lambda: ["clip-a.mp4"]),
    )

    source = rtsp_video_module.RtspSource("/tmp/data", "rtsp://127.0.0.1:8554/live", "cycle")
    source.running = False
    source.stream_once()
    assert source.current_process is None

    class DummyProcess:
        def wait(self):
            return 2

    popen_calls = []
    monkeypatch.setattr(
        rtsp_video_module.subprocess,
        "Popen",
        lambda command: popen_calls.append(command) or DummyProcess(),
    )

    source.running = True
    source.stream_once()

    assert popen_calls[0][-1] == "rtsp://127.0.0.1:8554/live"
    assert any("ffmpeg exited with code 2" in message for message in warnings)


@pytest.mark.unit
def test_rtsp_video_signal_handler_cleans_up_and_exits(monkeypatch):
    rtsp_video_module = importlib.import_module("rtsp_video")
    cleanup_calls = []
    monkeypatch.setattr(rtsp_video_module, "rtsp_source", SimpleNamespace(cleanup=lambda: cleanup_calls.append("cleanup")))
    monkeypatch.setattr(rtsp_video_module.sys, "exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))

    with pytest.raises(SystemExit) as exc_info:
        rtsp_video_module.handle_signal(None, None)

    assert exc_info.value.code == 0
    assert cleanup_calls == ["cleanup"]


@pytest.mark.unit
def test_rtsp_video_module_entrypoint_registers_signals_and_runs_cleanup(monkeypatch):
    signal_calls = []
    monkeypatch.setattr(
        importlib.import_module("argparse").ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(root="/tmp/data", address="rtsp://127.0.0.1:8554/live", play_mode="non-cycle"),
    )
    monkeypatch.setattr(signal, "signal", lambda signum, handler: signal_calls.append(signum))
    monkeypatch.setattr(
        importlib.import_module("video_dataset"),
        "VideoDataset",
        lambda root: SimpleNamespace(get_clip_paths=lambda: []),
    )
    monkeypatch.setattr(subprocess, "run", lambda command, stdout=None, stderr=None: SimpleNamespace(returncode=0))
    monkeypatch.setattr(sys, "argv", ["rtsp_video.py"])

    runpy.run_module("rtsp_video", run_name="__main__")

    assert signal_calls == [signal.SIGINT, signal.SIGTERM]
