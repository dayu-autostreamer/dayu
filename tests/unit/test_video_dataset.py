import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def create_video(path: Path, intensities):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    assert writer.isOpened(), f"Failed to create test video at {path}"
    for intensity in intensities:
        frame = np.full((16, 16, 3), intensity, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_manifest(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
def test_video_dataset_auto_assigns_contiguous_frame_offsets(tmp_path):
    video_dataset_module = importlib.import_module("video_dataset")

    create_video(tmp_path / "data" / "clips" / "segment_a.mp4", [50, 60])
    create_video(tmp_path / "data" / "clips" / "segment_b.mp4", [150, 160, 170])
    write_manifest(
        tmp_path / "http_video" / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [
                {"path": "clips/segment_a.mp4", "frame_count": 2, "start_frame_index": 10},
                {"path": "clips/segment_b.mp4", "frame_count": 3},
            ],
        },
    )

    dataset = video_dataset_module.VideoDataset(tmp_path / "http_video")

    assert [clip.start_frame_index for clip in dataset.clips] == [10, 12]
    assert [clip.frame_count for clip in dataset.clips] == [2, 3]


@pytest.mark.unit
def test_video_dataset_rejects_invalid_manifest_metadata(tmp_path):
    video_dataset_module = importlib.import_module("video_dataset")
    manifest_dir = tmp_path / "http_video"
    data_dir = tmp_path / "data" / "clips"
    create_video(data_dir / "segment_a.mp4", [50])

    invalid_cases = [
        (
            {
                "version": 2,
                "video_root": "../data",
                "sequence": [{"path": "clips/segment_a.mp4", "frame_count": 1}],
            },
            'Unsupported manifest version',
        ),
        (
            {
                "version": 1,
                "type": "image_sequence",
                "video_root": "../data",
                "sequence": [{"path": "clips/segment_a.mp4", "frame_count": 1}],
            },
            'Unsupported manifest type',
        ),
        (
            {
                "version": 1,
                "video_root": "../missing",
                "sequence": [{"path": "clips/segment_a.mp4", "frame_count": 1}],
            },
            'does not exist',
        ),
        (
            {
                "version": 1,
                "video_root": "../data",
                "sequence": [],
            },
            'must provide a non-empty "sequence" list',
        ),
    ]

    for index, (payload, message) in enumerate(invalid_cases):
        write_manifest(manifest_dir / "manifest.json", payload)
        with pytest.raises(ValueError, match=message):
            video_dataset_module.VideoDataset(manifest_dir)


@pytest.mark.unit
def test_video_dataset_validates_clip_entries(tmp_path):
    video_dataset_module = importlib.import_module("video_dataset")
    data_dir = tmp_path / "data" / "clips"
    create_video(data_dir / "segment_a.mp4", [50])
    bad_file = data_dir / "notes.txt"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("not a video", encoding="utf-8")

    invalid_manifests = [
        (
            {"version": 1, "video_root": "../data", "sequence": ["bad-entry"]},
            "Invalid clip entry",
        ),
        (
            {"version": 1, "video_root": "../data", "sequence": [{}]},
            'missing "path"',
        ),
        (
            {"version": 1, "video_root": "../data", "sequence": [{"path": "clips/missing.mp4"}]},
            "does not exist",
        ),
        (
            {"version": 1, "video_root": "../data", "sequence": [{"path": "clips/notes.txt"}]},
            "not a supported video",
        ),
        (
            {
                "version": 1,
                "video_root": "../data",
                "sequence": [{"path": "clips/segment_a.mp4", "frame_count": 1, "start_frame_index": -1}],
            },
            "negative start_frame_index",
        ),
    ]

    for payload, message in invalid_manifests:
        write_manifest(tmp_path / "http_video" / "manifest.json", payload)
        with pytest.raises(ValueError, match=message):
            video_dataset_module.VideoDataset(tmp_path / "http_video")


@pytest.mark.unit
def test_probe_video_frame_count_falls_back_to_ffprobe(monkeypatch):
    video_dataset_module = importlib.import_module("video_dataset")

    class FakeCapture:
        def isOpened(self):
            return True

        def get(self, key):
            return 0

        def release(self):
            return None

    monkeypatch.setattr(video_dataset_module.cv2, "VideoCapture", lambda path: FakeCapture())
    monkeypatch.setattr(
        video_dataset_module.subprocess,
        "check_output",
        lambda command, stderr=None, text=True: "42",
    )

    assert video_dataset_module.probe_video_frame_count("demo.mp4") == 42


@pytest.mark.unit
def test_video_dataset_player_handles_non_cycle_and_cycle_modes(tmp_path):
    video_dataset_module = importlib.import_module("video_dataset")

    create_video(tmp_path / "data" / "clips" / "segment_a.mp4", [50])
    create_video(tmp_path / "data" / "clips" / "segment_b.mp4", [150])
    write_manifest(
        tmp_path / "http_video" / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [
                {"path": "clips/segment_a.mp4", "frame_count": 1, "start_frame_index": 10},
                {"path": "clips/segment_b.mp4", "frame_count": 1, "start_frame_index": 20},
            ],
        },
    )

    non_cycle_player = video_dataset_module.VideoDatasetPlayer(tmp_path / "http_video", "non-cycle")
    frame_indices = []
    while True:
        frame, frame_index = non_cycle_player.read_frame()
        if frame is None:
            break
        frame_indices.append(frame_index)
    assert frame_indices == [10, 20]
    assert non_cycle_player.is_end is True

    cycle_player = video_dataset_module.VideoDatasetPlayer(tmp_path / "http_video", "cycle")
    cycled_indices = [cycle_player.read_frame()[1] for _ in range(3)]
    cycle_player.close()
    assert cycled_indices == [10, 20, 10]
