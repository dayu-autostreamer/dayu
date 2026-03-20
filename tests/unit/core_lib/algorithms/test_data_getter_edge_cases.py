import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


http_getter_module = importlib.import_module("core.lib.algorithms.data_getter.http_video_getter")
rtsp_getter_module = importlib.import_module("core.lib.algorithms.data_getter.rtsp_video_getter")


@pytest.mark.unit
def test_http_video_getter_waits_for_hashes_before_fetching_payload(monkeypatch, tmp_path):
    getter = http_getter_module.HttpVideoGetter()
    system = SimpleNamespace(
        source_id=3,
        video_data_source="http://datasource",
        meta_data={"fps": 10, "buffer_size": 2},
        raw_meta_data={"fps": 20},
    )
    request_log = []
    sleep_calls = []

    monkeypatch.setattr(http_getter_module.Context, "get_parameter", staticmethod(lambda key: "simple"))
    monkeypatch.setattr(
        http_getter_module.NameMaintainer,
        "get_task_data_file_name",
        staticmethod(lambda source_id, task_id, file_suffix: str(tmp_path / "payload.mp4")),
    )
    monkeypatch.setattr(http_getter_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    class FakeResponse:
        def __init__(self, content):
            self.content = content

    responses = iter([None, ["hash-0"], FakeResponse(b"video-bytes")])

    def fake_http_request(url, method=None, **kwargs):
        request_log.append(url)
        return next(responses)

    monkeypatch.setattr(http_getter_module, "http_request", fake_http_request)

    assert getter.request_source_data(system, task_id=7) is True
    assert Path(getter.file_name).read_bytes() == b"video-bytes"
    assert sleep_calls == [1]
    assert request_log == [
        "http://datasource/source",
        "http://datasource/source",
        "http://datasource/file",
    ]


@pytest.mark.unit
def test_http_video_getter_call_skips_round_when_datasource_is_exhausted(monkeypatch):
    getter = http_getter_module.HttpVideoGetter()
    sleep_calls = []
    submitted = []
    system = SimpleNamespace(
        source_id=3,
        meta_data={"fps": 10, "buffer_size": 2},
        raw_meta_data={"fps": 20},
        cumulative_scheduling_frame_count=0,
        task_dag=None,
        service_deployment=None,
        submit_task_to_controller=lambda task: submitted.append(task),
    )

    monkeypatch.setattr(http_getter_module.Counter, "get_count", staticmethod(lambda name: 11))
    monkeypatch.setattr(getter, "request_source_data", lambda current_system, task_id: False)
    monkeypatch.setattr(http_getter_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    getter(system)

    assert sleep_calls == [1]
    assert system.cumulative_scheduling_frame_count == 0
    assert submitted == []


@pytest.mark.unit
def test_rtsp_video_getter_open_capture_sets_tcp_transport_options(monkeypatch):
    getter = rtsp_getter_module.RtspVideoGetter()
    capture_calls = []

    import cv2

    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda url, backend: capture_calls.append((url, backend)) or "capture",
    )

    capture = getter._open_capture("rtsp://camera")

    assert capture == "capture"
    assert capture_calls == [("rtsp://camera", cv2.CAP_FFMPEG)]
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000"


@pytest.mark.unit
def test_rtsp_video_getter_recovers_even_if_previous_capture_release_fails(monkeypatch):
    getter = rtsp_getter_module.RtspVideoGetter()

    class BrokenCapture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            raise RuntimeError("socket already closed")

    class HealthyCapture:
        def isOpened(self):
            return True

        def read(self):
            return True, np.ones((2, 2, 3), dtype=np.uint8)

        def release(self):
            return None

    captures = iter([HealthyCapture()])
    getter.data_source_capture = BrokenCapture()

    monkeypatch.setattr(getter, "_open_capture", lambda url: next(captures))
    monkeypatch.setattr(rtsp_getter_module.time, "sleep", lambda seconds: None)

    frame = getter.get_one_frame(SimpleNamespace(source_id=8, video_data_source="rtsp://camera"))
    assert frame.shape == (2, 2, 3)


@pytest.mark.unit
def test_rtsp_video_getter_call_retries_until_filtered_buffer_is_filled(monkeypatch):
    getter = rtsp_getter_module.RtspVideoGetter()
    frames = iter(
        [
            np.zeros((2, 2, 3), dtype=np.uint8),
            np.ones((2, 2, 3), dtype=np.uint8),
            np.full((2, 2, 3), 2, dtype=np.uint8),
        ]
    )
    filter_results = iter([False, True, True])
    started = []

    class DummyProcess:
        def __init__(self, target=None, args=None):
            self.target = target
            self.args = args or ()

        def start(self):
            started.append((self.target, self.args))

    monkeypatch.setattr(getter, "get_one_frame", lambda system: next(frames))
    monkeypatch.setattr(getter, "filter_frame", lambda system, frame: next(filter_results))
    monkeypatch.setattr(rtsp_getter_module.Counter, "get_count", staticmethod(lambda name: 6))
    monkeypatch.setattr(rtsp_getter_module.multiprocessing, "Process", DummyProcess)

    system = SimpleNamespace(
        source_id=4,
        meta_data={"buffer_size": 2, "fps": 10},
        raw_meta_data={"fps": 20},
        cumulative_scheduling_frame_count=0,
        task_dag={"detector": ["edge-a"]},
        service_deployment={"detector": ["edge-a"]},
    )

    getter(system)

    assert system.cumulative_scheduling_frame_count == 4
    assert len(started) == 1
    assert len(started[0][1][1]) == 2
    assert getter.frame_buffer == []
