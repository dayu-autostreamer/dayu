import asyncio
import csv
import importlib
import json
import queue as py_queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


cache_module = importlib.import_module("core.lib.common.cache")
config_module = importlib.import_module("core.lib.common.config")
counter_module = importlib.import_module("core.lib.common.counter")
health_module = importlib.import_module("core.lib.common.health")
instance_module = importlib.import_module("core.lib.common.instance")
queue_module = importlib.import_module("core.lib.common.queue")
record_module = importlib.import_module("core.lib.common.record")
resource_module = importlib.import_module("core.lib.common.resource")
service_config_module = importlib.import_module("core.lib.common.service")
utils_module = importlib.import_module("core.lib.common.utils")
network_client_module = importlib.import_module("core.lib.network.client")
node_module = importlib.import_module("core.lib.network.node")


ConfigBoundInstanceCache = cache_module.ConfigBoundInstanceCache
ConfigLoader = config_module.ConfigLoader
Counter = counter_module.Counter
GlobalInstanceManager = instance_module.GlobalInstanceManager
Queue = queue_module.Queue
Recorder = record_module.Recorder
ResourceLockManager = resource_module.ResourceLockManager
ServiceConfig = service_config_module.ServiceConfig
HealthChecker = health_module.HealthChecker
NodeInfo = node_module.NodeInfo
http_request = network_client_module.http_request


@pytest.fixture(autouse=True)
def reset_global_registries():
    GlobalInstanceManager.release_all_instances()
    NodeInfo._NodeInfo__node_info_hostname = None
    NodeInfo._NodeInfo__node_info_ip = None
    NodeInfo._NodeInfo__node_info_role = None
    yield
    GlobalInstanceManager.release_all_instances()
    NodeInfo._NodeInfo__node_info_hostname = None
    NodeInfo._NodeInfo__node_info_ip = None
    NodeInfo._NodeInfo__node_info_role = None


@pytest.mark.unit
def test_config_loader_supports_json_yaml_and_extensionless_files(tmp_path):
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"alpha": 1, "items": [1, 2]}), encoding="utf-8")

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("alpha: 2\nitems:\n  - x\n", encoding="utf-8")

    extensionless_path = tmp_path / "config"
    extensionless_path.write_text("beta: true\ncount: 3\n", encoding="utf-8")

    invalid_path = tmp_path / "broken.txt"
    invalid_path.write_text("alpha: [1, 2", encoding="utf-8")

    assert ConfigLoader.get_supported_formats() == ["JSON", "YAML"]
    assert ConfigLoader.load(str(json_path)) == {"alpha": 1, "items": [1, 2]}
    assert ConfigLoader.load(str(yaml_path)) == {"alpha": 2, "items": ["x"]}
    assert ConfigLoader.load(str(extensionless_path)) == {"beta": True, "count": 3}

    with pytest.raises(ValueError, match="Failed to parse config file"):
        ConfigLoader.load(str(invalid_path))

    with pytest.raises(ValueError, match="not found"):
        ConfigLoader.load(str(tmp_path / "missing.yaml"))


@pytest.mark.unit
def test_queue_drops_oldest_items_and_exposes_non_destructive_snapshots():
    queue = Queue(maxsize=2)

    assert queue.empty() is True
    queue.put(1)
    queue.put(2)
    assert queue.get_all_without_drop() == [1, 2]
    assert queue.full() is True

    queue.put(3)
    assert queue.get_all_without_drop() == [2, 3]
    assert queue.get() == 2

    queue.put_all([4, 5])
    assert queue.size() == 2
    assert queue.get_all() == [4, 5]
    assert queue.empty() is True

    queue.put("tail")
    queue.clear()
    assert queue.empty() is True


@pytest.mark.unit
def test_queue_and_instance_manager_cover_exception_swallowing_and_registry_snapshots(monkeypatch):
    queue = Queue(maxsize=1)

    class FakeInternalQueue:
        @staticmethod
        def full():
            return True

        @staticmethod
        def get_nowait():
            raise py_queue.Empty

        @staticmethod
        def put_nowait(item):
            raise py_queue.Full

    monkeypatch.setattr(queue, "_Queue__queue", FakeInternalQueue())
    queue.put("payload")

    class Alpha:
        pass

    class Beta:
        pass

    alpha = GlobalInstanceManager.get_instance(Alpha, "alpha")
    beta = GlobalInstanceManager.get_instance(Beta, "beta")
    snapshot = GlobalInstanceManager.get_all_instances()

    assert snapshot[Alpha]["alpha"] is alpha
    assert snapshot[Beta]["beta"] is beta

    snapshot[Alpha].clear()
    assert GlobalInstanceManager.has_instance(Alpha, "alpha") is True
    assert GlobalInstanceManager.get_all_instances(dict) == {}


@pytest.mark.unit
def test_resource_lock_manager_enforces_single_holder_per_resource():
    manager = ResourceLockManager()

    assert asyncio.run(manager.acquire_lock("camera-0", "edge-a")) == "edge-a"
    assert asyncio.run(manager.acquire_lock("camera-0", "edge-b")) == "edge-a"
    assert asyncio.run(manager.get_current_holder("camera-0")) == "edge-a"
    assert asyncio.run(manager.release_lock("camera-0", "edge-b")) is False
    assert asyncio.run(manager.release_lock("camera-0", "edge-a")) is True
    assert asyncio.run(manager.get_current_holder("camera-0")) is None


@pytest.mark.unit
def test_resource_lock_manager_creates_async_lock_only_inside_running_event_loop(monkeypatch):
    created_loops = []
    real_lock = asyncio.Lock

    def tracking_lock():
        created_loops.append(id(asyncio.get_running_loop()))
        return real_lock()

    monkeypatch.setattr(resource_module.asyncio, "Lock", tracking_lock)

    manager = ResourceLockManager()

    assert asyncio.run(manager.acquire_lock("camera-1", "edge-a")) == "edge-a"
    assert asyncio.run(manager.get_current_holder("camera-1")) == "edge-a"
    assert len(created_loops) == 2
    assert created_loops[0] != created_loops[1]


@pytest.mark.unit
def test_resource_lock_manager_reuses_single_async_lock_within_same_event_loop(monkeypatch):
    created_locks = []
    real_lock = asyncio.Lock

    def tracking_lock():
        created_locks.append("created")
        return real_lock()

    monkeypatch.setattr(resource_module.asyncio, "Lock", tracking_lock)
    manager = ResourceLockManager()

    async def run_sequence():
        assert await manager.acquire_lock("camera-2", "edge-a") == "edge-a"
        assert await manager.acquire_lock("camera-2", "edge-b") == "edge-a"
        assert await manager.release_lock("camera-2", "edge-a") is True

    asyncio.run(run_sequence())
    assert created_locks == ["created"]


@pytest.mark.unit
def test_global_instance_manager_scopes_instances_by_class_and_id():
    class Demo:
        init_calls = 0

        def __init__(self, value):
            Demo.init_calls += 1
            self.value = value

    first = GlobalInstanceManager.get_instance(Demo, "alpha", 1)
    second = GlobalInstanceManager.get_instance(Demo, "alpha", 99)
    third = GlobalInstanceManager.get_instance(Demo, "beta", 3)

    assert first is second
    assert first.value == 1
    assert third is not first
    assert Demo.init_calls == 2
    assert GlobalInstanceManager.has_instance(Demo, "alpha") is True
    assert set(GlobalInstanceManager.get_all_instances(Demo)) == {"alpha", "beta"}

    GlobalInstanceManager.release_instance(Demo, "alpha")
    assert GlobalInstanceManager.has_instance(Demo, "alpha") is False

    GlobalInstanceManager.release_all_instances(Demo)
    assert GlobalInstanceManager.get_all_instances(Demo) == {}


@pytest.mark.unit
def test_counter_and_instance_manager_ignore_missing_namespaces_and_instances():
    Counter.reset_count("missing")
    assert Counter.get_all_counts() == {}
    assert Counter.get_count("alpha") == 0
    assert Counter.get_count("alpha") == 1
    Counter.reset_count("alpha")
    Counter.reset_count("alpha")
    assert Counter.get_all_counts() == {}

    class Demo:
        pass

    GlobalInstanceManager.release_instance(Demo, "missing")
    GlobalInstanceManager.release_all_instances(Demo)
    assert GlobalInstanceManager.has_instance(Demo, "missing") is False


@pytest.mark.unit
def test_service_config_and_common_utils_cover_patterns_and_nested_merges():
    assert ServiceConfig.map_pod_name_to_service("processor-face-detection-edgenode-edgeworker-0") == "face-detection"
    assert ServiceConfig.map_pod_name_to_service("processor-car-detection-edgenode-abc123") == "car-detection"
    assert ServiceConfig.map_pod_name_to_service("processor-gender-worker") == "gender"
    assert ServiceConfig.map_pod_name_to_service("backend-0") is None

    assert utils_module.reverse_key_value_in_dict({"a": 1, "b": 2}) == {1: "a", 2: "b"}

    converted = utils_module.convert_ndarray_to_list(
        {
            "frame": np.array([[1, 2], [3, 4]]),
            "nested": (np.array([5, 6]), {"k": np.array([7])}),
        }
    )
    assert converted == {"frame": [[1, 2], [3, 4]], "nested": ([5, 6], {"k": [7]})}

    merged = utils_module.deep_merge(
        {
            "meta": {"fps": 25},
            "items": [{"name": "alpha", "value": 1}, {"plain": 1}],
        },
        {
            "meta": {"resolution": "720p"},
            "items": [{"name": "alpha", "extra": 2}, {"plain": 9}, "tail"],
            "added": True,
        },
    )
    assert merged == {
        "meta": {"fps": 25, "resolution": "720p"},
        "items": [
            {"name": "alpha", "value": 1, "extra": 2},
            {"plain": 9},
            "tail",
        ],
        "added": True,
    }

    @utils_module.singleton
    class DemoSingleton:
        def __init__(self, value):
            self.value = value

    first = DemoSingleton(1)
    second = DemoSingleton(2)
    assert first is second
    assert second.value == 1

    merged_named_append = utils_module.deep_merge(
        [{"name": "alpha", "value": 1}],
        [{"name": "beta", "value": 2}],
    )
    assert merged_named_append == [{"name": "alpha", "value": 1}, {"name": "beta", "value": 2}]

    assert utils_module.deep_merge("left", {"right": True}) == {"right": True}
    assert cache_module._canonical_json({"b": 1, "a": 2}) == cache_module._canonical_json({"a": 2, "b": 1})


@pytest.mark.unit
def test_config_bound_instance_cache_reconfigures_rebuilds_and_cleans_up():
    created = []
    closed = []

    def factory(cfg):
        instance = SimpleNamespace(name=cfg["name"], version=cfg.get("version", 1))
        created.append((instance.name, instance.version))
        return instance

    def reconfigure(instance, cfg):
        if cfg.get("inplace"):
            instance.version = cfg["version"]
            return True
        return False

    def closer(instance):
        closed.append((instance.name, instance.version))

    cache = ConfigBoundInstanceCache(
        factory=factory,
        reconfigure=reconfigure,
        closer=closer,
    )

    first_alpha, first_beta = cache.sync_and_get(
        [{"name": "alpha", "version": 1}, {"name": "beta", "version": 1}]
    )
    second_alpha, second_gamma = cache.sync_and_get(
        [
            {"name": "alpha", "version": 2, "inplace": True},
            {"name": "gamma", "version": 1},
        ]
    )

    assert second_alpha is first_alpha
    assert second_alpha.version == 2
    assert second_gamma is not first_beta
    assert ("beta", 1) in closed

    rebuilt_alpha, _ = cache.sync_and_get(
        [
            {"name": "alpha", "version": 3},
            {"name": "gamma", "version": 1},
        ]
    )
    assert rebuilt_alpha is not first_alpha
    assert ("alpha", 2) in closed
    assert cache.get_existing("name:gamma").name == "gamma"

    cache.remove("name:gamma")
    assert cache.get_existing("name:gamma") is None

    cache.clear_all()
    assert cache.get_existing("name:alpha") is None
    assert len(created) == 4


@pytest.mark.unit
def test_config_bound_instance_cache_rebuilds_when_reconfigure_raises(monkeypatch):
    timestamps = iter([100.0, 101.0, 102.0])
    monkeypatch.setattr(cache_module.time, "time", lambda: next(timestamps))

    created = []
    closed = []

    def factory(cfg):
        instance = SimpleNamespace(name=cfg["name"], version=cfg["version"])
        created.append((instance.name, instance.version))
        return instance

    def reconfigure(instance, cfg):
        raise RuntimeError("boom")

    cache = ConfigBoundInstanceCache(
        factory=factory,
        reconfigure=reconfigure,
        closer=lambda instance: closed.append((instance.name, instance.version)),
    )

    [first] = cache.sync_and_get([{"name": "alpha", "version": 1}])
    [second] = cache.sync_and_get([{"name": "alpha", "version": 2}])

    assert second is not first
    assert created == [("alpha", 1), ("alpha", 2)]
    assert closed == [("alpha", 1)]


@pytest.mark.unit
def test_config_bound_instance_cache_prunes_idle_entries_and_enforces_capacity(monkeypatch):
    timestamps = iter(range(1, 20))
    monkeypatch.setattr(cache_module.time, "time", lambda: next(timestamps))

    closed = []

    def factory(cfg):
        return SimpleNamespace(name=cfg["name"])

    cache = ConfigBoundInstanceCache(
        factory=factory,
        closer=lambda instance: closed.append(instance.name),
        capacity=2,
    )

    cache.sync_and_get([{"name": "one"}], namespace="ns1")
    cache.sync_and_get([{"name": "two"}], namespace="ns2")
    assert cache.get_existing("name:one", namespace="ns1").name == "one"

    cache.sync_and_get([{"name": "three"}], namespace="ns3")
    assert cache.get_existing("name:one", namespace="ns1").name == "one"
    assert cache.get_existing("name:two", namespace="ns2") is None
    assert "two" in closed

    removed = cache.prune_idle(idle_seconds=0.5)
    assert removed == 2
    assert sorted(closed) == ["one", "three", "two"]


@pytest.mark.unit
def test_cache_key_and_hash_helpers_cover_identity_and_behavior_fields():
    assert cache_module.default_stable_key_fn({"id": "camera-1", "name": "ignored"}) == "id:camera-1"
    assert cache_module.default_stable_key_fn({"name": "detector"}) == "name:detector"
    assert cache_module.default_stable_key_fn({"type": "visualizer", "variables": ["fps"]}) == 'visualizer|vars:["fps"]'
    assert cache_module.default_stable_key_fn({"hook_name": "simple"}) == "simple|vars:{}"
    assert cache_module.default_stable_key_fn({}) == "unknown|vars:{}"

    first_hash = cache_module.default_config_hash_fn({"id": "a", "name": "alpha", "fps": 25})
    second_hash = cache_module.default_config_hash_fn({"id": "b", "name": "beta", "fps": 25})
    third_hash = cache_module.default_config_hash_fn({"id": "b", "name": "beta", "fps": 30})
    custom_hash = cache_module.default_config_hash_fn({"slot": "a", "fps": 25}, ignored_keys=("slot",))

    assert first_hash == second_hash
    assert first_hash != third_hash
    assert custom_hash == cache_module.default_config_hash_fn({"slot": "b", "fps": 25}, ignored_keys=("slot",))


@pytest.mark.unit
def test_config_bound_instance_cache_uses_default_namespace_and_swallows_cleanup_errors(monkeypatch):
    timestamps = iter([10.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr(cache_module.time, "time", lambda: next(timestamps))

    closed = []

    def closer(instance):
        closed.append(instance.name)
        if instance.name == "alpha":
            raise RuntimeError("close failed")

    cache = ConfigBoundInstanceCache(
        factory=lambda cfg: SimpleNamespace(name=cfg["name"]),
        closer=closer,
        default_namespace="visualizers",
    )

    [alpha] = cache.sync_and_get([{"name": "alpha"}])
    assert alpha.name == "alpha"
    assert cache.get_existing("name:alpha").name == "alpha"
    assert cache.get_existing("name:missing") is None

    cache.remove("name:missing")
    cache.remove("name:missing", namespace="ghost")
    cache.clear_namespace(namespace="missing")
    cache.clear_namespace()

    assert closed == ["alpha"]
    assert cache.get_existing("name:alpha") is None


@pytest.mark.unit
def test_recorder_supports_csv_jsonl_and_append_validation(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    with Recorder(str(csv_path), fmt="csv", add_timestamp=False) as recorder:
        recorder.log(step=1, loss=0.5)
        recorder.log(step=2, loss=0.25, extra="ignored")

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [
        {"loss": "0.5", "step": "1"},
        {"loss": "0.25", "step": "2"},
    ]

    jsonl_path = tmp_path / "metrics.jsonl"
    recorder = Recorder(str(jsonl_path), fmt="jsonl", add_timestamp=False)
    recorder.log_dict({"step": 1, "reward": 0.2})
    recorder.log(step=2, reward=0.4)
    recorder.close()

    assert [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()] == [
        {"step": 1, "reward": 0.2},
        {"step": 2, "reward": 0.4},
    ]

    bad_recorder = Recorder(str(tmp_path / "bad.jsonl"), fmt="jsonl")
    with pytest.raises(TypeError, match="expects a dict"):
        bad_recorder.log_dict(["not", "a", "dict"])
    bad_recorder.close()

    with pytest.raises(ValueError, match="must provide fieldnames"):
        Recorder(str(csv_path), fmt="csv", overwrite=False)


@pytest.mark.unit
def test_recorder_supports_explicit_fieldnames_timestamps_and_repeated_flushes(monkeypatch, tmp_path):
    csv_path = tmp_path / "timestamped.csv"
    monkeypatch.setattr(record_module.time, "time", lambda: 123.456)

    recorder = Recorder(
        str(csv_path),
        fmt="csv",
        fieldnames=["step", "wall_time"],
        add_timestamp=True,
        flush_every=0,
    )
    recorder.log(step=1, ignored="value")
    recorder.flush()
    recorder.close()
    recorder.flush()
    recorder.close()

    append_recorder = Recorder(
        str(csv_path),
        fmt="csv",
        fieldnames=["step", "wall_time"],
        overwrite=False,
        add_timestamp=False,
    )
    append_recorder.log(step=2, wall_time=456.0, ignored="value")
    append_recorder.close()

    with csv_path.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == [
        {"step": "1", "wall_time": "123.456"},
        {"step": "2", "wall_time": "456.0"},
    ]

    with pytest.raises(AssertionError, match="Unsupported format"):
        Recorder(str(tmp_path / "broken.log"), fmt="tsv")


@pytest.mark.unit
def test_recorder_skips_flush_until_threshold_and_swallows_del_close_errors(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "delayed.jsonl"
    recorder = Recorder(str(jsonl_path), fmt="jsonl", add_timestamp=False, flush_every=2)

    flush_calls = []

    class FakeFile:
        def __init__(self):
            self.closed = False
            self.buffer = []

        def write(self, content):
            self.buffer.append(content)

        def flush(self):
            flush_calls.append("flush")

        def close(self):
            self.closed = True

    fake_file = FakeFile()
    recorder._f.close()
    recorder._f = fake_file

    recorder.log(step=1)
    assert flush_calls == []
    recorder.log(step=2)
    assert flush_calls == ["flush"]

    monkeypatch.setattr(recorder, "close", lambda: (_ for _ in ()).throw(RuntimeError("close failed")))
    recorder.__del__()


@pytest.mark.unit
def test_http_request_handles_success_redirects_and_failures(monkeypatch):
    class FakeResponse:
        def __init__(self, status_code, payload=None, content=b"text", url="http://redirect"):
            self.status_code = status_code
            self._payload = payload
            self.content = content
            self.url = url

        def json(self):
            return self._payload

    response = FakeResponse(200, payload={"ok": True})
    monkeypatch.setattr(network_client_module.requests, "request", lambda **kwargs: response)
    assert http_request("http://service") == {"ok": True}
    assert http_request("http://service", no_decode=True) is response

    monkeypatch.setattr(
        network_client_module.requests,
        "request",
        lambda **kwargs: FakeResponse(200, payload=None, content=b"payload"),
    )
    assert http_request("http://service", binary=False) == "payload"

    monkeypatch.setattr(
        network_client_module.requests,
        "request",
        lambda **kwargs: FakeResponse(302),
    )
    assert http_request("http://service") is None

    monkeypatch.setattr(
        network_client_module.requests,
        "request",
        lambda **kwargs: FakeResponse(404),
    )
    assert http_request("http://service") is None

    monkeypatch.setattr(
        network_client_module.requests,
        "request",
        lambda **kwargs: (_ for _ in ()).throw(network_client_module.requests.exceptions.Timeout("slow")),
    )
    assert http_request("http://service") is None

    monkeypatch.setattr(
        network_client_module.requests,
        "request",
        lambda **kwargs: (_ for _ in ()).throw(network_client_module.requests.exceptions.ConnectionError("down")),
    )
    assert http_request("http://service") is None


@pytest.mark.unit
def test_health_checker_and_node_info_cover_cluster_lookup_and_caching(monkeypatch):
    api_calls = {"list_node": 0, "list_namespaced_pod": 0}

    def node(name, ip, role_label):
        labels = {role_label: ""}
        return SimpleNamespace(
            metadata=SimpleNamespace(name=name, labels=labels),
            status=SimpleNamespace(addresses=[SimpleNamespace(type="InternalIP", address=ip)]),
        )

    nodes = [
        node("cloud-node", "10.0.0.1", "node-role.kubernetes.io/master"),
        node("edge-node", "10.0.0.2", "node-role.kubernetes.io/edge"),
    ]
    pods = [
        SimpleNamespace(spec=SimpleNamespace(node_name="edge-node")),
        SimpleNamespace(spec=SimpleNamespace(node_name="cloud-node")),
        SimpleNamespace(spec=SimpleNamespace(node_name="edge-node")),
    ]

    class FakeCoreV1Api:
        def list_node(self):
            api_calls["list_node"] += 1
            return SimpleNamespace(items=nodes)

        def list_namespaced_pod(self, namespace, label_selector):
            api_calls["list_namespaced_pod"] += 1
            return SimpleNamespace(items=pods)

    monkeypatch.setattr(node_module.config, "load_incluster_config", lambda: None)
    monkeypatch.setattr(node_module.client, "CoreV1Api", FakeCoreV1Api)
    monkeypatch.setattr(
        node_module.Context,
        "get_parameter",
        staticmethod(lambda key: {"NAMESPACE": "dayu", "NODE_NAME": "edge-node"}.get(key)),
    )
    monkeypatch.setenv("NODE_NAME", "edge-node")

    assert NodeInfo.get_node_info() == {"cloud-node": "10.0.0.1", "edge-node": "10.0.0.2"}
    assert NodeInfo.get_node_info_reverse() == {"10.0.0.1": "cloud-node", "10.0.0.2": "edge-node"}
    assert NodeInfo.get_node_info_role() == {"cloud-node": "cloud", "edge-node": "edge"}
    assert api_calls["list_node"] == 1

    assert NodeInfo.hostname2ip("edge-node") == "10.0.0.2"
    assert NodeInfo.ip2hostname("10.0.0.1") == "cloud-node"
    assert NodeInfo.url2hostname("http://10.0.0.2:9000/health") == "edge-node"
    assert NodeInfo.get_node_role("cloud-node") == "cloud"
    assert NodeInfo.get_cloud_node() == "cloud-node"
    assert NodeInfo.get_all_edge_nodes() == ["edge-node"]
    assert sorted(NodeInfo.get_edge_nodes()) == ["edge-node"]
    assert api_calls["list_namespaced_pod"] == 1
    assert NodeInfo.get_local_device() == "edge-node"

    monkeypatch.setattr(health_module.NodeInfo, "get_edge_nodes", staticmethod(lambda: ["edge-node"]))
    monkeypatch.setattr(health_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(health_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: hostname))
    monkeypatch.setattr(health_module.PortInfo, "get_component_port", staticmethod(lambda component: 9002))
    monkeypatch.setattr(
        health_module,
        "http_request",
        lambda url, method=None: {"status": "ok"} if "cloud-node" in url or "edge-node" in url else {},
    )
    assert HealthChecker.check_processors_health() is True

    monkeypatch.setattr(health_module, "http_request", lambda url, method=None: {"status": "ok"} if "edge-node" in url else {})
    assert HealthChecker.check_processors_health() is False
