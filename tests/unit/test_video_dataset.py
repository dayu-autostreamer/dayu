import importlib
import json
from pathlib import Path

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
