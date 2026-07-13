import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from core.lib.runtime import RuntimeContext


http_getter_module = importlib.import_module("core.lib.algorithms.data_getter.http_video_getter")
rtsp_getter_module = importlib.import_module("core.lib.algorithms.data_getter.rtsp_video_getter")
v4l2_getter_module = importlib.import_module("core.lib.algorithms.data_getter.v4l2_video_getter")
scheduler_filter_module = importlib.import_module(
    "core.lib.algorithms.data_getter_filter.scheduler_permitted_getter_filter"
)


def action_routes(node="edge-a"):
    def route(component, port, service=""):
        slot = {"component": component, "target_node": node}
        if service:
            slot["logical_service"] = service
        return {"slot": slot, "runtime_id": f"{component}-{node}", "runtime_revision": 1, "endpoint": {
            "dns_name": f"{component}-{node}.dayu.svc", "port": port,
            "runtime_service_uid": f"rs-{component}", "service_uid": f"svc-{component}",
            "pod_uid": f"pod-{component}",
        }}
    return [route("controller", 9002), route("processor", 9004, "face")]


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


@pytest.mark.unit
def test_v4l2_video_getter_opens_device_with_v4l2_backend(monkeypatch):
    getter = v4l2_getter_module.V4L2VideoGetter()
    capture_calls = []

    import cv2

    def fake_capture(source, backend=None):
        capture_calls.append((source, backend))
        return "capture"

    monkeypatch.setattr(cv2, "VideoCapture", fake_capture)

    capture = getter._open_capture("/dev/video0")

    assert capture == "capture"
    assert capture_calls == [("/dev/video0", cv2.CAP_V4L2)]


@pytest.fixture
def scheduler_filter_env(monkeypatch):
    monkeypatch.setenv("DAYU_RUNTIME_BOOTSTRAP", json.dumps({
        "local_node": "edge-a", "cloud_node": "cloud-node",
        "endpoints": {"scheduler": {"fqdn": "scheduler.dayu.svc", "port": 9001}},
    }))
    RuntimeContext.reset_default()


@pytest.mark.unit
def test_scheduler_permitted_filter_fail_open_and_throttled_logging(scheduler_filter_env, monkeypatch):
    log_messages = []
    time_values = iter([100.0, 100.5, 102.0, 200.0, 200.2, 202.5, 300.0])

    monkeypatch.setattr(scheduler_filter_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(scheduler_filter_module.LOGGER, "info", lambda message: log_messages.append(("info", message)))
    monkeypatch.setattr(
        scheduler_filter_module.LOGGER,
        "warning",
        lambda message: log_messages.append(("warning", message)),
    )

    getter_filter = scheduler_filter_module.SchedulerPermittedDataGetterFilter(
        fail_open=False,
        timeout_s=0,
        log_interval_s=2,
    )

    assert getter_filter.timeout_s == 0.1
    assert getter_filter.scheduler_address == "http://scheduler.dayu.svc:9001/generation_admission"

    getter_filter._log_throttled("blocked")
    getter_filter._log_throttled("blocked-again")
    getter_filter._log_throttled("blocked-later")
    getter_filter._log_throttled("error", is_error=True)
    getter_filter._log_throttled("error-again", is_error=True)
    getter_filter._log_throttled("error-later", is_error=True)

    assert log_messages == [
        ("info", "blocked"),
        ("info", "blocked-later"),
        ("warning", "error"),
        ("warning", "error-later"),
    ]

    system = SimpleNamespace(source_id=7, local_device="edge-a", meta_data={}, raw_meta_data={},
                             runtime_routes=action_routes(), runtime_directory_revision=1)
    monkeypatch.setattr(scheduler_filter_module, "http_request", lambda *args, **kwargs: None)

    assert getter_filter(system) is False


@pytest.mark.unit
def test_scheduler_permitted_filter_executes_clear_queue_actions_once(scheduler_filter_env, monkeypatch):
    action_requests = []
    admission_payloads = []

    responses = iter(
        [
            {
                "generate": False,
                "reason": "queue_pressure",
                "actions": {
                    "type": "clear_processor_queues",
                    "command_id": "clear-1",
                    "target_devices": "edge-a",
                    "request": {"dry_run": False, "timeout_s": "bad-value", "reason": "pressure"},
                },
            },
            {"ok": True, "cleared_count": 2, "matched_count": 3, "remaining_count": 1},
            {
                "allow": True,
                "commands": [
                    {
                        "type": "clear_processor_queues",
                        "command_id": "clear-1",
                        "target_devices": ["edge-a"],
                    },
                    {"type": "unknown"},
                    "bad-action",
                ],
            },
        ]
    )

    def fake_http_request(url, method=None, **kwargs):
        if url.endswith("/generation_admission"):
            admission_payloads.append(kwargs["data"]["data"])
        else:
            action_requests.append((url, method, kwargs))
        return next(responses)

    monkeypatch.setattr(scheduler_filter_module, "http_request", fake_http_request)
    monkeypatch.setattr(scheduler_filter_module.LOGGER, "warning", lambda message: None)

    getter_filter = scheduler_filter_module.SchedulerPermittedDataGetterFilter(action_retry_interval_s=0)
    system = SimpleNamespace(
        source_id=9,
        local_device="edge-a",
        meta_data={"fps": 10},
        raw_meta_data={"fps": 30},
        runtime_routes=action_routes(),
        runtime_directory_revision=1,
    )

    assert getter_filter(system) is False
    assert getter_filter(system) is True

    assert action_requests == [
        (
            "http://controller-edge-a.dayu.svc:9002/processor_queues_clear",
            "POST",
            {
                "timeout": 1.0,
                "data": action_requests[0][2]["data"],
            },
        )
    ]
    assert "clear-1:edge-a" in getter_filter._completed_action_targets
    assert admission_payloads[0]
    assert '"completed_action_targets": []' in admission_payloads[0]
    assert '"completed_action_targets": ["clear-1:edge-a"]' in admission_payloads[1]


@pytest.mark.unit
def test_scheduler_permitted_filter_retries_failed_actions_after_interval(scheduler_filter_env, monkeypatch):
    errors = []
    request_count = 0
    time_values = iter([10.0, 12.0, 12.5, 20.0, 22.0])

    def fake_http_request(*args, **kwargs):
        nonlocal request_count
        request_count += 1
        return {"ok": False}

    monkeypatch.setattr(scheduler_filter_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(scheduler_filter_module, "http_request", fake_http_request)
    monkeypatch.setattr(scheduler_filter_module.LOGGER, "warning", lambda message: errors.append(message))

    getter_filter = scheduler_filter_module.SchedulerPermittedDataGetterFilter(
        action_retry_interval_s=5,
        log_interval_s=1,
    )
    action = {
        "type": "clear_processor_queues",
        "command_id": "clear-2",
        "target_devices": ["edge-a"],
        "request": {"timeout_s": 1},
    }

    system = SimpleNamespace(runtime_routes=action_routes(), runtime_directory_revision=1)

    getter_filter._execute_scheduler_actions({"actions": [action]}, system)
    getter_filter._execute_scheduler_actions({"actions": [action]}, system)
    getter_filter._execute_scheduler_actions({"actions": [action]}, system)

    assert request_count == 2
    assert len(errors) == 2


@pytest.mark.unit
def test_scheduler_permitted_filter_ignores_invalid_actions_and_logs_controller_failures(
    scheduler_filter_env,
    monkeypatch,
):
    errors = []
    request_urls = []

    def failing_request(url, *args, **kwargs):
        request_urls.append(url)
        raise RuntimeError("controller unavailable")

    monkeypatch.setattr(scheduler_filter_module, "http_request", failing_request)
    monkeypatch.setattr(scheduler_filter_module.LOGGER, "warning", lambda message: errors.append(message))

    getter_filter = scheduler_filter_module.SchedulerPermittedDataGetterFilter()
    system = SimpleNamespace(runtime_routes=action_routes(), runtime_directory_revision=1)

    getter_filter._execute_scheduler_actions({"actions": "not-a-list"}, system)
    getter_filter._execute_scheduler_actions({"actions": [{"type": "clear_processor_queues", "target_devices": 5}]}, system)
    getter_filter._execute_scheduler_actions(
        {
            "actions": [
                {
                    "type": "clear_processor_queues",
                    "command_id": "clear-3",
                    "target_devices": ["edge-a"],
                    "request": "not-a-dict",
                }
            ]
        }, system
    )

    assert request_urls == ["http://controller-edge-a.dayu.svc:9002/processor_queues_clear"]
    assert len(errors) == 1
    assert "Failed to execute scheduler action clear-3" in errors[0]
