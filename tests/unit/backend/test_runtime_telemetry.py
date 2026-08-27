import copy
import math
import threading
from types import SimpleNamespace

import pytest

from runtime_telemetry import RuntimeTelemetryCache


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def empty_resource_usage(status):
    return {
        "cpu": {
            "status": status,
            "usage_millicores": None,
            "reference_millicores": None,
            "utilization_percent": None,
            "basis": "",
        },
        "memory": {
            "status": status,
            "usage_bytes": None,
            "reference_bytes": None,
            "utilization_percent": None,
            "basis": "",
        },
    }


def directory(revision=1, install_id="install-a", pod_suffix="r1"):
    routes = []
    for index, service in enumerate(("detect", "track"), start=1):
        routes.append(SimpleNamespace(
            slot=SimpleNamespace(
                component="processor",
                logical_service=service,
                target_node=f"edge-{index}",
            ),
            runtime_id=f"processor-{service}-{pod_suffix}",
            pod_name=f"processor-{service}-{pod_suffix}",
            pod_uid=f"uid-{service}-{pod_suffix}",
        ))
    routes.append(SimpleNamespace(
        slot=SimpleNamespace(
            component="scheduler", logical_service="", target_node="cloud-a",
        ),
        runtime_id="scheduler-r1",
        pod_name="scheduler-r1",
        pod_uid="uid-scheduler-r1",
    ))
    return SimpleNamespace(
        install_id=install_id,
        revision=revision,
        routes=tuple(routes),
    )


def scheduler_request(resource_values=None, overhead_values=None, calls=None):
    resource_values = iter(resource_values or ({},))
    overhead_values = iter(overhead_values or (0.0,))

    def request(url, method=None, timeout=None):
        if calls is not None:
            calls.append((url, method, timeout))
        if url.endswith("/resource"):
            return next(resource_values)
        return next(overhead_values)

    return request


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True, "invalid"])
def test_runtime_telemetry_rejects_invalid_internal_timing_values(value):
    with pytest.raises(ValueError, match="finite positive number"):
        RuntimeTelemetryCache(
            request=lambda *args, **kwargs: None,
            runtime_metrics=lambda *args, **kwargs: {},
            interval_seconds=value,
        )


def test_bind_immediately_exposes_exact_processor_placeholders_without_sampling():
    calls = []
    cache = RuntimeTelemetryCache(
        request=lambda *args, **kwargs: calls.append((args, kwargs)),
        runtime_metrics=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    cache.bind("http://scheduler/resource", directory())
    snapshot = cache.snapshot(logical_service="detect")

    assert calls == []
    assert snapshot["install_id"] == "install-a"
    assert snapshot["directory_revision"] == 1
    assert snapshot["runtime_metrics"] == {
        "processor-detect-r1": {
            "name": "processor-detect-r1",
            "uid": "uid-detect-r1",
            "node": "edge-1",
            "node_info": {},
            "phase": "",
            "ready": None,
            "pod_ip": "",
            "created_at": "",
            "resources": {},
            "usage": {},
            "resource_usage": empty_resource_usage("collecting"),
            "runtime_id": "processor-detect-r1",
            "logical_service": "detect",
        },
    }
    assert snapshot["runtime_metrics_sampled_at"] is None


def test_runtime_telemetry_batches_all_processors_and_retains_independent_lkg_fields():
    clock = FakeClock()
    scheduler_calls = []
    metric_calls = []
    metric_responses = iter([
        {
            "processor-detect-r1": {
                "uid": "uid-detect-r1", "node": "edge-a", "usage": {},
            },
            "processor-track-r1": {
                "uid": "uid-track-r1", "node": "edge-b", "usage": {},
            },
        },
        RuntimeError("metrics unavailable"),
    ])

    def runtime_metrics(refs, request_timeout_seconds):
        metric_calls.append((copy.deepcopy(refs), request_timeout_seconds))
        response = next(metric_responses)
        if isinstance(response, Exception):
            raise response
        return response

    cache = RuntimeTelemetryCache(
        request=scheduler_request(
            resource_values=({"edge-a": {"available_bandwidth": 12.5}}, None, None),
            overhead_values=(0.025, None, None),
            calls=scheduler_calls,
        ),
        runtime_metrics=runtime_metrics,
        interval_seconds=2,
        metrics_interval_seconds=10,
        scheduler_request_timeout_seconds=1.5,
        kubernetes_request_timeout_seconds=4.0,
        clock=clock,
    )
    cache.bind("http://scheduler/resource", directory())

    assert cache._sample_once() is True
    first = cache.snapshot()
    assert first["resource"] == {"edge-a": {"available_bandwidth": 12.5}}
    assert first["scheduling_overhead"] == 0.025
    assert set(first["runtime_metrics"]) == {
        "processor-detect-r1", "processor-track-r1",
    }
    assert first["runtime_metrics"]["processor-detect-r1"]["logical_service"] == "detect"
    assert metric_calls == [([
        {"name": "processor-detect-r1", "uid": "uid-detect-r1"},
        {"name": "processor-track-r1", "uid": "uid-track-r1"},
    ], 4.0)]
    assert {call[2] for call in scheduler_calls} == {1.5}

    # Scheduler is sampled at its faster cadence, while Kubernetes waits for
    # its own period. Failed reads retain values but expose resource freshness.
    clock.value = 2.0
    assert cache._sample_once() is False
    assert len(metric_calls) == 1
    clock.value = 10.0
    assert cache._sample_once() is False
    second = cache.snapshot()
    expected = copy.deepcopy(first)
    expected["resource_stale"] = True
    assert second == expected
    assert len(metric_calls) == 2

    detect = cache.snapshot(logical_service="detect")
    assert set(detect["runtime_metrics"]) == {"processor-detect-r1"}
    detect["runtime_metrics"]["processor-detect-r1"]["node"] = "mutated"
    assert cache.snapshot("detect")["runtime_metrics"]["processor-detect-r1"]["node"] == "edge-a"


def test_runtime_telemetry_discards_inflight_sample_after_rebind():
    request_started = threading.Event()
    release_request = threading.Event()

    def runtime_metrics(refs, request_timeout_seconds):
        request_started.set()
        assert release_request.wait(1)
        return {
            refs[0]["name"]: {
                "uid": refs[0]["uid"], "node": "edge-old", "usage": {},
            },
        }

    cache = RuntimeTelemetryCache(
        request=scheduler_request(resource_values=({}, {}), overhead_values=(0, 0)),
        runtime_metrics=runtime_metrics,
    )
    cache.bind("http://old-scheduler/resource", directory())
    worker = threading.Thread(target=cache._sample_once)
    worker.start()
    assert request_started.wait(1)

    cache.bind(
        "http://new-scheduler/resource",
        directory(revision=1, install_id="install-b", pod_suffix="new"),
    )
    release_request.set()
    worker.join(timeout=1)

    assert worker.is_alive() is False
    assert cache.snapshot() == {
        "install_id": "install-b",
        "directory_revision": 1,
        "resource": None,
        "scheduling_overhead": None,
        "runtime_metrics": {
            "processor-detect-new": {
                "name": "processor-detect-new",
                "uid": "uid-detect-new",
                "node": "edge-1",
                "node_info": {},
                "phase": "",
                "ready": None,
                "pod_ip": "",
                "created_at": "",
                "resources": {},
                "usage": {},
                "resource_usage": empty_resource_usage("collecting"),
                "runtime_id": "processor-detect-new",
                "logical_service": "detect",
            },
            "processor-track-new": {
                "name": "processor-track-new",
                "uid": "uid-track-new",
                "node": "edge-2",
                "node_info": {},
                "phase": "",
                "ready": None,
                "pod_ip": "",
                "created_at": "",
                "resources": {},
                "usage": {},
                "resource_usage": empty_resource_usage("collecting"),
                "runtime_id": "processor-track-new",
                "logical_service": "track",
            },
        },
        "scheduler_sampled_at": None,
        "resource_sampled_at": None,
        "resource_stale": False,
        "runtime_metrics_sampled_at": None,
    }


def test_processor_rebind_retains_scheduler_lkg_but_clears_old_pod_metrics():
    cache = RuntimeTelemetryCache(
        request=scheduler_request(
            resource_values=({"edge-a": {}},), overhead_values=(0.1,),
        ),
        runtime_metrics=lambda refs, request_timeout_seconds: {
            ref["name"]: {"uid": ref["uid"], "usage": {}} for ref in refs
        },
    )
    cache.bind("http://scheduler/resource", directory())
    assert cache._sample_once() is True

    cache.bind(
        "http://scheduler/resource",
        directory(revision=2, pod_suffix="r2"),
    )
    snapshot = cache.snapshot()
    assert snapshot["resource"] == {"edge-a": {}}
    assert snapshot["scheduling_overhead"] == 0.1
    assert set(snapshot["runtime_metrics"]) == {
        "processor-detect-r2", "processor-track-r2",
    }
    assert snapshot["runtime_metrics"]["processor-detect-r2"]["node"] == "edge-1"
    assert snapshot["runtime_metrics"]["processor-detect-r2"]["ready"] is None
    assert snapshot["directory_revision"] == 2

    cache.unbind()
    assert cache.snapshot()["resource"] is None
    assert cache.snapshot()["runtime_metrics"] == {}


def test_runtime_telemetry_is_single_flight_and_close_is_bounded():
    request_started = threading.Event()
    release_request = threading.Event()

    def runtime_metrics(refs, request_timeout_seconds):
        request_started.set()
        assert release_request.wait(1)
        return {}

    cache = RuntimeTelemetryCache(
        request=scheduler_request(resource_values=({},), overhead_values=(0,)),
        runtime_metrics=runtime_metrics,
        interval_seconds=10,
    )
    cache.bind("http://scheduler/resource", directory())
    cache.start()
    assert request_started.wait(1)
    worker = cache._thread

    cache.start()
    assert cache._thread is worker
    assert cache._sample_once() is False

    release_request.set()
    cache.close(join_timeout=1)
    assert worker.is_alive() is False


def test_runtime_telemetry_retains_resource_values_per_metric_and_marks_them_stale():
    clock = FakeClock()
    responses = iter([
        {
            "processor-detect-r1": {
                "uid": "uid-detect-r1",
                "resource_usage": {
                    "cpu": {
                        "status": "available",
                        "usage_millicores": 100.0,
                        "reference_millicores": 1000.0,
                        "utilization_percent": 10.0,
                        "basis": "node_allocatable",
                    },
                    "memory": {
                        "status": "available",
                        "usage_bytes": 100,
                        "reference_bytes": 1000,
                        "utilization_percent": 10.0,
                        "basis": "node_allocatable",
                    },
                },
            },
        },
        {
            "processor-detect-r1": {
                "uid": "uid-detect-r1",
                "resource_usage": {
                    "cpu": {
                        "status": "available",
                        "usage_millicores": 200.0,
                        "reference_millicores": 1000.0,
                        "utilization_percent": 20.0,
                        "basis": "node_allocatable",
                    },
                    "memory": {
                        "status": "error",
                        "usage_bytes": None,
                        "reference_bytes": 1000,
                        "utilization_percent": None,
                        "basis": "node_allocatable",
                    },
                },
            },
        },
        RuntimeError("Pod list unavailable"),
    ])

    def runtime_metrics(refs, request_timeout_seconds):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    cache = RuntimeTelemetryCache(
        request=scheduler_request(
            resource_values=({}, {}, {}),
            overhead_values=(0, 0, 0),
        ),
        runtime_metrics=runtime_metrics,
        metrics_interval_seconds=10,
        clock=clock,
    )
    cache.bind("http://scheduler/resource", directory())

    assert cache._sample_once() is True
    clock.value = 10
    assert cache._sample_once() is True
    second = cache.snapshot("detect")
    usage = second["runtime_metrics"]["processor-detect-r1"]["resource_usage"]
    assert usage["cpu"]["status"] == "available"
    assert usage["cpu"]["usage_millicores"] == 200.0
    assert usage["memory"]["status"] == "stale"
    assert usage["memory"]["usage_bytes"] == 100
    last_successful_sample = second["runtime_metrics_sampled_at"]

    clock.value = 20
    assert cache._sample_once() is True
    third = cache.snapshot("detect")
    usage = third["runtime_metrics"]["processor-detect-r1"]["resource_usage"]
    assert usage["cpu"]["status"] == "stale"
    assert usage["cpu"]["usage_millicores"] == 200.0
    assert usage["memory"]["status"] == "stale"
    assert third["runtime_metrics_sampled_at"] == last_successful_sample


def test_runtime_telemetry_first_failed_attempt_is_unavailable_not_collecting():
    cache = RuntimeTelemetryCache(
        request=scheduler_request(resource_values=({},), overhead_values=(0,)),
        runtime_metrics=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Pod list unavailable")
        ),
    )
    cache.bind("http://scheduler/resource", directory())

    assert cache._sample_once() is True
    snapshot = cache.snapshot("detect")
    usage = snapshot["runtime_metrics"]["processor-detect-r1"]["resource_usage"]
    assert usage["cpu"]["status"] == "unavailable"
    assert usage["memory"]["status"] == "unavailable"
    assert snapshot["runtime_metrics_sampled_at"] is None


def test_runtime_telemetry_successful_sample_with_missing_exact_pod_is_unavailable():
    cache = RuntimeTelemetryCache(
        request=scheduler_request(resource_values=({},), overhead_values=(0,)),
        runtime_metrics=lambda refs, request_timeout_seconds: {},
    )
    cache.bind("http://scheduler/resource", directory())

    assert cache._sample_once() is True
    snapshot = cache.snapshot("detect")
    usage = snapshot["runtime_metrics"]["processor-detect-r1"]["resource_usage"]
    assert usage["cpu"]["status"] == "unavailable"
    assert usage["memory"]["status"] == "unavailable"
    assert snapshot["runtime_metrics_sampled_at"] is not None


def test_scheduler_resource_freshness_is_independent_from_overhead_success():
    clock = FakeClock()
    cache = RuntimeTelemetryCache(
        request=scheduler_request(
            resource_values=({"edge-probe": {"available_bandwidth": 12.5}}, None),
            overhead_values=(0.1, 0.2),
        ),
        runtime_metrics=lambda refs, request_timeout_seconds: {},
        interval_seconds=2,
        metrics_interval_seconds=10,
        clock=clock,
    )
    cache.bind("http://scheduler/resource", directory())

    assert cache._sample_once() is True
    first = cache.snapshot()
    assert first["resource_stale"] is False
    assert first["resource_sampled_at"] is not None

    clock.value = 2
    assert cache._sample_once() is True
    second = cache.snapshot()
    assert second["resource"] == first["resource"]
    assert second["resource_sampled_at"] == first["resource_sampled_at"]
    assert second["resource_stale"] is True
    assert second["scheduling_overhead"] == 0.2
