import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


video_dataset_module = importlib.import_module("video_dataset")


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_video_dataset_helpers_cover_extensions_probe_failures_and_missing_manifest(monkeypatch, tmp_path):
    assert video_dataset_module.is_video_file("clip.MP4") is True
    assert video_dataset_module.is_video_file("clip.txt") is False

    class ClosedCapture:
        def isOpened(self):
            return False

    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: ClosedCapture())
    with pytest.raises(ValueError, match='Failed to open video "broken.mp4"'):
        video_dataset_module.probe_video_frame_count("broken.mp4")

    class ZeroCapture:
        def isOpened(self):
            return True

        def get(self, key):
            return 0

        def release(self):
            return None

    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: ZeroCapture())
    monkeypatch.setattr(
        video_dataset_module.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("ffprobe")),
    )
    assert video_dataset_module.probe_video_frame_count("demo.mp4") == 0

    monkeypatch.setattr(video_dataset_module.subprocess, "check_output", lambda *args, **kwargs: "not-a-number")
    assert video_dataset_module.probe_video_frame_count("demo.mp4") == 0

    monkeypatch.setattr(
        video_dataset_module.subprocess,
        "check_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "ffprobe")),
    )
    assert video_dataset_module.probe_video_frame_count("demo.mp4") == 0

    class CountCapture:
        def isOpened(self):
            return True

        def get(self, key):
            return 12

        def release(self):
            return None

    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: CountCapture())
    assert video_dataset_module.probe_video_frame_count("direct.mp4") == 12

    with pytest.raises(ValueError, match="Missing manifest file"):
        video_dataset_module.VideoDataset(tmp_path / "missing_manifest")


@pytest.mark.unit
def test_video_dataset_builds_clip_paths_and_validates_frame_count_resolution(monkeypatch, tmp_path):
    data_dir = tmp_path / "data" / "clips"
    manifest_dir = tmp_path / "dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    clip_path = data_dir / "segment_a.mp4"
    clip_path.write_bytes(b"video")

    write_manifest(
        manifest_dir / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [{"path": "clips/segment_a.mp4", "frame_count": 4, "start_frame_index": 10}],
        },
    )

    dataset = video_dataset_module.VideoDataset(manifest_dir)
    assert dataset.total_frames == 4
    assert dataset.get_clip_paths() == [str(clip_path.resolve())]
    assert dataset.clips[0].end_frame_index == 13

    monkeypatch.setattr(video_dataset_module, "probe_video_frame_count", lambda path: 0)
    write_manifest(
        manifest_dir / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [{"path": "clips/segment_a.mp4"}],
        },
    )
    with pytest.raises(ValueError, match="Failed to resolve frame count"):
        video_dataset_module.VideoDataset(manifest_dir)


@pytest.mark.unit
def test_video_dataset_player_handles_capture_seek_failures_and_clip_advances(monkeypatch):
    clip_a = video_dataset_module.VideoClip("a", "clip-a.mp4", frame_count=2, start_frame_index=10)
    clip_b = video_dataset_module.VideoClip("b", "clip-b.mp4", frame_count=1, start_frame_index=20)

    player = object.__new__(video_dataset_module.VideoDatasetPlayer)
    player.dataset = SimpleNamespace(clips=[clip_a, clip_b])
    player.play_mode = "cycle"
    player.current_clip_index = 0
    player.current_frame_offset = 1
    player.capture = None
    player.is_end = False

    set_calls = []

    class OpenCapture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def set(self, key, value):
            set_calls.append((key, value))

        def release(self):
            return None

    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: OpenCapture())
    player._ensure_capture()
    assert set_calls == [(video_dataset_module.cv2.CAP_PROP_POS_FRAMES, 1)]

    player.capture = OpenCapture()
    player._advance_after_clip_end()
    assert player.current_clip_index == 1
    assert player.current_frame_offset == 0

    player.current_clip_index = 1
    player.current_frame_offset = 0
    player.capture = OpenCapture()
    assert player._advance_to_next_clip() is True
    assert player.current_clip_index == 0
    assert player.current_frame_offset == 0

    player.current_clip_index = 0
    player.capture = OpenCapture()
    assert player._advance_to_next_clip() is True
    assert player.current_clip_index == 1

    player.play_mode = "non-cycle"
    player.current_clip_index = 1
    player.capture = OpenCapture()
    assert player._advance_to_next_clip() is False
    assert player.is_end is True

    player = object.__new__(video_dataset_module.VideoDatasetPlayer)
    player.dataset = SimpleNamespace(clips=[clip_a])
    player.play_mode = "non-cycle"
    player.current_clip_index = 0
    player.current_frame_offset = 0
    player.is_end = False
    player.capture = OpenCapture()
    monkeypatch.setattr(player, "_advance_to_next_clip", lambda: False)
    assert player.read_frame() == (None, None)

    class ClosedCapture:
        def isOpened(self):
            return False

    player = object.__new__(video_dataset_module.VideoDatasetPlayer)
    player.dataset = SimpleNamespace(clips=[clip_a])
    player.play_mode = "cycle"
    player.current_clip_index = 0
    player.current_frame_offset = 0
    player.capture = None
    player.is_end = False
    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: ClosedCapture())
    with pytest.raises(ValueError, match='Failed to open video "clip-a.mp4"'):
        player._ensure_capture()


@pytest.mark.unit
def test_video_dataset_player_retries_next_clip_when_current_capture_returns_no_frame(monkeypatch):
    clip_a = video_dataset_module.VideoClip("a", "clip-a.mp4", frame_count=1, start_frame_index=10)
    clip_b = video_dataset_module.VideoClip("b", "clip-b.mp4", frame_count=1, start_frame_index=20)

    captures = {
        "clip-a.mp4": SimpleNamespace(
            isOpened=lambda: True,
            read=lambda: (False, None),
            set=lambda key, value: None,
            release=lambda: None,
        ),
        "clip-b.mp4": SimpleNamespace(
            isOpened=lambda: True,
            read=lambda: (True, "frame-b"),
            set=lambda key, value: None,
            release=lambda: None,
        ),
    }

    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: captures[path])

    player = object.__new__(video_dataset_module.VideoDatasetPlayer)
    player.dataset = SimpleNamespace(clips=[clip_a, clip_b])
    player.play_mode = "cycle"
    player.current_clip_index = 0
    player.current_frame_offset = 0
    player.capture = None
    player.is_end = False

    frame, frame_index = player.read_frame()

    assert (frame, frame_index) == ("frame-b", 20)
    assert player.current_clip_index == 0
    assert player.current_frame_offset == 0
