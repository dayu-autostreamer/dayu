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


def create_frame(path: Path, intensity):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((16, 16, 3), intensity, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame), f"Failed to create test frame at {path}"


def decode_payload(response):
    if isinstance(response, list):
        return response
    return json.loads(response.body.decode("utf-8"))


def configure_algorithms(monkeypatch, video_source_module, tmp_path):
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
        video_source_module.Context,
        "get_algorithm",
        staticmethod(lambda algorithm_type, al_name=None: algorithms[algorithm_type]),
    )
    monkeypatch.setattr(
        video_source_module.NameMaintainer,
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
def test_video_source_reads_discovered_video_files_in_natural_order(monkeypatch, tmp_path):
    video_source_module = importlib.import_module("video_source")
    compressed_batches = configure_algorithms(monkeypatch, video_source_module, tmp_path)

    create_video(tmp_path / "videos" / "10.mp4", [150, 160])
    create_video(tmp_path / "videos" / "2.mp4", [50, 60])

    source = video_source_module.VideoSource(str(tmp_path), "cycle")
    response = source.get_source_data(build_request(buffer_size=5))

    assert decode_payload(response) == [1, 2, 3, 0, 1]
    assert len(compressed_batches) == 1
    assert_batch_close(compressed_batches[0], [50, 60, 150, 160, 50])


@pytest.mark.unit
def test_video_source_manifest_overrides_play_order(monkeypatch, tmp_path):
    video_source_module = importlib.import_module("video_source")
    compressed_batches = configure_algorithms(monkeypatch, video_source_module, tmp_path)

    create_video(tmp_path / "videos" / "10.mp4", [150, 160])
    create_video(tmp_path / "videos" / "2.mp4", [50, 60])
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": ["videos/10.mp4", "videos/2.mp4"]}),
        encoding="utf-8",
    )

    source = video_source_module.VideoSource(str(tmp_path), "cycle")
    response = source.get_source_data(build_request(buffer_size=4))

    assert decode_payload(response) == [1, 2, 3, 0]
    assert len(compressed_batches) == 1
    assert_batch_close(compressed_batches[0], [150, 160, 50, 60])


@pytest.mark.unit
def test_video_source_keeps_legacy_frames_storage_compatible(monkeypatch, tmp_path):
    video_source_module = importlib.import_module("video_source")
    compressed_batches = configure_algorithms(monkeypatch, video_source_module, tmp_path)

    create_frame(tmp_path / "frames" / "10.jpg", 120)
    create_frame(tmp_path / "frames" / "2.jpg", 30)

    source = video_source_module.VideoSource(str(tmp_path), "cycle")
    response = source.get_source_data(build_request(buffer_size=3))

    assert isinstance(source.storage, video_source_module.FramesSourceStorage)
    assert decode_payload(response) == [1, 0, 1]
    assert len(compressed_batches) == 1
    assert_batch_close(compressed_batches[0], [30, 120, 30])
