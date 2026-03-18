import importlib
import json
from pathlib import Path

import pytest


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_rtsp_source_streams_clips_in_manifest_order(monkeypatch, tmp_path):
    rtsp_source_module = importlib.import_module("rtsp_source")

    clip_a = tmp_path / "data" / "clips" / "a.mp4"
    clip_b = tmp_path / "data" / "clips" / "b.mp4"
    clip_a.parent.mkdir(parents=True, exist_ok=True)
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")

    write_manifest(
        tmp_path / "rtsp_video" / "manifest.json",
        {
            "version": 1,
            "media_root": "../data",
            "sequence": [
                {"path": "clips/b.mp4", "frame_count": 1},
                {"path": "clips/a.mp4", "frame_count": 1},
            ],
        },
    )

    monkeypatch.setattr(rtsp_source_module.RtspSource, "ensure_mediamtx", lambda self: None)

    commands = []

    class DummyProcess:
        def __init__(self, command):
            self.command = command

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(
        rtsp_source_module.subprocess,
        "Popen",
        lambda command, **kwargs: commands.append(command) or DummyProcess(command),
    )

    source = rtsp_source_module.RtspSource(str(tmp_path / "rtsp_video"), "rtsp://127.0.0.1:8554/live", "non-cycle")
    source.run()

    assert [Path(command[4]).name for command in commands] == ["b.mp4", "a.mp4"]
