import importlib
import json

import pytest


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
    monkeypatch.setattr(monitor_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "scheduler-node"))
    monkeypatch.setattr(monitor_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(monitor_module.NodeInfo, "get_local_device", staticmethod(lambda: "edge-node"))
    monkeypatch.setattr(
        monitor_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: 9001),
    )
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
