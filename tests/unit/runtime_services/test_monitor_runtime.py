import importlib
import json

import pytest

from core.lib.runtime import RuntimeContext


monitor_module = importlib.import_module("core.monitor.monitor")
monitor_server_module = importlib.import_module("core.monitor.monitor_server")


class FakeMonitorWorker:
    def __init__(self, system, name, value, calls):
        self.system = system
        self.name = name
        self.value = value
        self.calls = calls

    def start(self):
        self.calls.append(("start", self.name))
        self.system.resource_info[self.name] = self.value

    def join(self):
        self.calls.append(("join", self.name))


@pytest.mark.unit
def test_monitor_initializes_workers_waits_by_interval_and_posts_resource_state(monkeypatch):
    worker_calls = []
    sleeps = []
    requests = []
    timestamps = iter([10.0, 12.0, 13.0])

    def fake_get_parameter(name, direct=False):
        if name == "INTERVAL":
            return 5
        if name == "MONITORS":
            return ["cpu", "memory"]
        raise AssertionError(f"Unexpected parameter request: {name}")

    def fake_get_algorithm(algorithm, al_name=None, system=None, **kwargs):
        if algorithm != "MON_PRAM":
            raise AssertionError(f"Unexpected algorithm request: {algorithm}")
        values = {"cpu": 0.4, "memory": 0.6}
        return lambda al_name=al_name, system=system: FakeMonitorWorker(system, al_name, values[al_name], worker_calls)

    monkeypatch.setattr(monitor_module.Context, "get_parameter", staticmethod(fake_get_parameter))
    monkeypatch.setattr(monitor_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    context = RuntimeContext({
        "local_node": "edge-node",
        "cloud_node": "scheduler-node",
        "endpoints": {"scheduler": {"fqdn": "10.0.0.8", "port": 9001}},
    })
    monkeypatch.setattr(monitor_module.RuntimeContext, "get_default", staticmethod(lambda: context))
    monkeypatch.setattr(monitor_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(monitor_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    monkeypatch.setattr(monitor_module.LOGGER, "info", lambda message: None)
    monkeypatch.setattr(
        monitor_module,
        "http_request",
        lambda url, method=None, **kwargs: requests.append((url, method, kwargs)),
    )

    monitor = monitor_module.Monitor()
    monitor.monitor_resource()
    monitor.wait_for_monitor()
    monitor.send_resource_state_to_scheduler()

    assert monitor.scheduler_address == "http://10.0.0.8:9001/resource"
    assert monitor.resource_info == {"cpu": 0.4, "memory": 0.6}
    assert worker_calls == [
        ("start", "cpu"),
        ("start", "memory"),
        ("join", "cpu"),
        ("join", "memory"),
    ]
    assert sleeps == [3.0]

    payload = json.loads(requests[0][2]["data"]["data"])
    assert requests[0][0] == "http://10.0.0.8:9001/resource"
    assert requests[0][1] == "POST"
    assert payload == {"device": "edge-node", "resource": {"cpu": 0.4, "memory": 0.6}}


@pytest.mark.unit
def test_monitor_server_runs_monitor_send_wait_in_order(monkeypatch):
    calls = []

    class FakeMonitor:
        def monitor_resource(self):
            calls.append("monitor")

        def send_resource_state_to_scheduler(self):
            calls.append("send")

        def wait_for_monitor(self):
            calls.append("wait")
            raise RuntimeError("stop")

    monkeypatch.setattr(monitor_server_module, "Monitor", FakeMonitor)
    server = monitor_server_module.MonitorServer()

    with pytest.raises(RuntimeError, match="stop"):
        server.run()

    assert calls == ["monitor", "send", "wait"]


@pytest.mark.unit
def test_monitor_reads_exact_processor_routes_once_per_interval(monkeypatch):
    monkeypatch.setattr(
        monitor_module.Context,
        "get_parameter",
        staticmethod(lambda name, direct=False: 10 if name == "INTERVAL" else []),
    )
    context = RuntimeContext({
        "local_node": "edge-node",
        "cloud_node": "cloud-node",
        "endpoints": {"scheduler": {"fqdn": "scheduler.dayu.svc.cluster.local", "port": 9000}},
    })
    monkeypatch.setattr(monitor_module.RuntimeContext, "get_default", staticmethod(lambda: context))
    monkeypatch.setattr(monitor_module.time, "time", lambda: 100.0)

    requests = []
    directory = {
        "revision": 4,
        "routes": [{
            "component": "processor",
            "logical_service": "detector",
            "target_node": "edge-node",
            "runtime_id": "processor-detector-edge-node-4",
            "runtime_revision": 4,
            "dns_name": "processor-detector.dayu.svc.cluster.local",
            "port": 9000,
            "runtime_service_uid": "runtime-uid",
            "service_uid": "service-uid",
            "pod_uid": "pod-uid",
        }],
    }
    monkeypatch.setattr(
        monitor_module,
        "http_request",
        lambda url, method=None, **kwargs: requests.append((url, method)) or directory,
    )

    monitor = monitor_module.Monitor()
    first = monitor.runtime_routes(
        component="processor", target_node="edge-node", logical_service="detector"
    )
    second = monitor.runtime_routes(component="processor", target_node="edge-node")

    assert [route.runtime_id for route in first] == ["processor-detector-edge-node-4"]
    assert [route.runtime_id for route in second] == ["processor-detector-edge-node-4"]
    assert requests == [
        ("http://scheduler.dayu.svc.cluster.local.:9000/runtime-directory", "GET")
    ]


@pytest.mark.unit
def test_monitor_negative_caches_failed_directory_reads(monkeypatch):
    monkeypatch.setattr(
        monitor_module.Context,
        "get_parameter",
        staticmethod(lambda name, direct=False: 10 if name == "INTERVAL" else []),
    )
    context = RuntimeContext({
        "local_node": "edge-node",
        "endpoints": {"scheduler": {"fqdn": "scheduler.dayu.svc.cluster.local", "port": 9000}},
    })
    monkeypatch.setattr(monitor_module.RuntimeContext, "get_default", staticmethod(lambda: context))
    monkeypatch.setattr(monitor_module.time, "time", lambda: 100.0)
    requests = []
    monkeypatch.setattr(
        monitor_module,
        "http_request",
        lambda url, method=None, **kwargs: requests.append(url) or None,
    )

    monitor = monitor_module.Monitor()
    assert monitor.runtime_routes(component="processor") == []
    assert monitor.runtime_routes(component="processor") == []
    assert requests == ["http://scheduler.dayu.svc.cluster.local.:9000/runtime-directory"]
