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


def decode_payload(response):
    if isinstance(response, list):
        return response
    return json.loads(response.body.decode("utf-8"))


def configure_algorithms(monkeypatch, http_video_module, tmp_path):
    compressed_batches = []

    class PassThroughFilter:
        def __call__(self, system, frame):
            return True

    class IdentityProcess:
        def __call__(self, system, frame, source_resolution, target_resolution):
            return frame

    class CaptureCompress:
        def __call__(self, system, frame_buffer, file_name):
            compressed_batches.append([int(frame[0, 0, 0]) for frame in frame_buffer])
            Path(file_name).write_bytes(b"test-payload")

    algorithms = {
        "GEN_FILTER": PassThroughFilter(),
        "GEN_PROCESS": IdentityProcess(),
        "GEN_COMPRESS": CaptureCompress(),
    }

    monkeypatch.setattr(
        http_video_module.Context,
        "get_algorithm",
        staticmethod(lambda algorithm_type, al_name=None: algorithms[algorithm_type]),
    )
    monkeypatch.setattr(
        http_video_module.NameMaintainer,
        "get_task_data_file_name",
        staticmethod(lambda source_id, task_id, file_suffix: str(tmp_path / f"task-{task_id}.{file_suffix}")),
    )

    return compressed_batches


def build_request(buffer_size):
    return json.dumps(
        {
            "source_id": 0,
            "task_id": 1,
            "meta_data": {
                "buffer_size": buffer_size,
                "resolution": "540p",
                "fps": 30,
                "encoding": "mp4v",
            },
            "raw_meta_data": {
                "resolution": "540p",
                "fps": 30,
                "encoding": "mp4v",
            },
            "gen_filter_name": "simple",
            "gen_process_name": "simple",
            "gen_compress_name": "simple",
        }
    )


def assert_batch_close(actual, expected, tolerance=15):
    assert len(actual) == len(expected)
    for current, target in zip(actual, expected):
        assert abs(current - target) <= tolerance


@pytest.mark.unit
def test_http_video_reads_shared_data_from_manifest(monkeypatch, tmp_path):
    http_video_module = importlib.import_module("http_video")
    compressed_batches = configure_algorithms(monkeypatch, http_video_module, tmp_path)

    create_video(tmp_path / "data" / "clips" / "000001.mp4", [50, 60])
    create_video(tmp_path / "data" / "clips" / "000002.mp4", [150, 160])
    write_manifest(
        tmp_path / "http_video" / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [
                {"path": "clips/000001.mp4", "frame_count": 2, "start_frame_index": 0},
                {"path": "clips/000002.mp4", "frame_count": 2},
            ],
        },
    )

    source = http_video_module.VideoSource(str(tmp_path / "http_video"), "cycle")
    response = source.get_source_data(build_request(buffer_size=4))

    assert decode_payload(response) == [0, 1, 2, 3]
    assert len(compressed_batches) == 1
    assert_batch_close(compressed_batches[0], [50, 60, 150, 160])


@pytest.mark.unit
def test_http_video_respects_explicit_ground_truth_offsets(monkeypatch, tmp_path):
    http_video_module = importlib.import_module("http_video")
    compressed_batches = configure_algorithms(monkeypatch, http_video_module, tmp_path)

    create_video(tmp_path / "data" / "clips" / "segment_a.mp4", [50, 60])
    create_video(tmp_path / "data" / "clips" / "segment_b.mp4", [150, 160])
    write_manifest(
        tmp_path / "http_video" / "manifest.json",
        {
            "version": 1,
            "video_root": "../data",
            "sequence": [
                {"path": "clips/segment_a.mp4", "frame_count": 2, "start_frame_index": 100},
                {"path": "clips/segment_b.mp4", "frame_count": 2, "start_frame_index": 200},
            ],
        },
    )

    source = http_video_module.VideoSource(str(tmp_path / "http_video"), "cycle")
    response = source.get_source_data(build_request(buffer_size=4))

    assert decode_payload(response) == [100, 101, 200, 201]
    assert len(compressed_batches) == 1
    assert_batch_close(compressed_batches[0], [50, 60, 150, 160])
