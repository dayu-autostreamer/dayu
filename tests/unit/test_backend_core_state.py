import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.lib.common import Queue


@pytest.fixture
def backend_core_instance(mounted_runtime, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pod_name",
        staticmethod(lambda *args, **kwargs: False),
    )
    return backend_core_module.BackendCore()


@pytest.mark.unit
def test_component_yaml_state_machine_updates_adds_and_clears_docs(backend_core_instance, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    backend_core_instance.save_yaml_path = str(tmp_path / "resources.yaml")
    original_docs = [
        {"apiVersion": "v1", "kind": "Joint", "metadata": {"name": "scheduler"}, "spec": {"logLevel": {"level": "INFO"}}},
        {"apiVersion": "v1", "kind": "Joint", "metadata": {"name": "processor-a"}, "spec": {"image": "old"}},
        {"apiVersion": "v1", "kind": "Joint", "metadata": {"name": "processor-b"}, "spec": {"image": "keep"}},
    ]
    update_docs = [
        {"apiVersion": "v1", "kind": "Joint", "metadata": {"name": "scheduler"}, "spec": {"logLevel": {"level": "DEBUG"}}},
        {"apiVersion": "v1", "kind": "Joint", "metadata": {"name": "processor-a"}, "spec": {"image": "new"}},
        {"apiVersion": "v1", "kind": "Joint", "metadata": {"name": "processor-c"}, "spec": {"image": "add"}},
    ]

    total_docs, docs_to_add, docs_to_update, docs_to_delete = backend_core_instance.check_and_update_docs_list(
        copy.deepcopy(original_docs),
        copy.deepcopy(update_docs),
    )

    assert [doc["metadata"]["name"] for doc in docs_to_add] == ["processor-c"]
    assert [doc["metadata"]["name"] for doc in docs_to_update] == ["processor-a"]
    assert [doc["metadata"]["name"] for doc in docs_to_delete] == ["processor-b"]
    assert sorted(doc["metadata"]["name"] for doc in total_docs) == ["processor-a", "processor-c", "scheduler"]

    backend_core_instance.save_component_yaml(original_docs)
    assert backend_core_instance.read_component_yaml()[1]["metadata"]["name"] == "processor-a"

    backend_core_instance.update_component_yaml(update_docs)
    updated_names = sorted(doc["metadata"]["name"] for doc in backend_core_instance.read_component_yaml())
    assert updated_names == ["processor-a", "processor-c", "scheduler"]

    backend_core_instance.clear_yaml_docs()
    assert backend_core_instance.read_component_yaml() is None


@pytest.mark.unit
def test_backend_core_log_snapshot_compaction_and_record_count(backend_core_instance, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    backend_core_instance.system_log_store_path = str(tmp_path / "system.jsonl")
    backend_core_instance.system_log_retention_records = 2
    backend_core_instance.system_log_compact_interval = 1

    backend_core_instance._append_system_log_snapshot({"timestamp": "10:00:00", "data": [1]})
    backend_core_instance._append_system_log_snapshot({"timestamp": "10:00:01", "data": [2]})
    backend_core_instance._append_system_log_snapshot({"timestamp": "10:00:02", "data": [3]})
    backend_core_instance._append_system_log_snapshot({"timestamp": "10:00:03", "data": [4]})
    backend_core_instance.system_log_record_count = 4

    assert backend_core_instance._count_jsonl_records(backend_core_instance.system_log_store_path) == 4

    backend_core_instance._maybe_compact_system_log_store_locked()
    lines = Path(backend_core_instance.system_log_store_path).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["timestamp"] == "10:00:02"
    assert json.loads(lines[1])["timestamp"] == "10:00:03"
    assert backend_core_instance.system_log_record_count == 2


@pytest.mark.unit
def test_backend_core_url_helpers_stream_fetch_and_parse_task_results(backend_core_instance, monkeypatch, tmp_path):
    backend_core_module = importlib.import_module("backend_core")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(backend_core_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloudx1"))
    monkeypatch.setattr(backend_core_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(
        backend_core_module.PortInfo,
        "get_component_port",
        staticmethod(lambda component: 9000 if component == "scheduler" else 9002),
    )

    backend_core_instance.get_resource_url()
    backend_core_instance.get_result_url()
    backend_core_instance.get_log_url()

    assert backend_core_instance.resource_url == "http://10.0.0.8:9000/resource"
    assert backend_core_instance.result_url == "http://10.0.0.8:9002/result"
    assert backend_core_instance.result_file_url == "http://10.0.0.8:9002/file"
    assert backend_core_instance.log_fetch_url == "http://10.0.0.8:9002/export_result_log"

    class FakeResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def iter_content(self, chunk_size=8192):
            for chunk in self.chunks:
                yield chunk

    requests = []

    def fake_http_request(url, method=None, **kwargs):
        requests.append((url, method, kwargs))
        if url.endswith("/file"):
            return FakeResponse([b"chunk-", b"data"])
        if url.endswith("/export_result_log"):
            return FakeResponse([b"gzip-data"])
        return None

    monkeypatch.setattr(backend_core_module, "http_request", fake_http_request)

    downloaded = backend_core_instance.get_file_result("artifact.bin")
    assert Path(downloaded).read_bytes() == b"chunk-data"
    assert backend_core_instance.open_result_log_export_stream() is not None

    task = importlib.import_module("core.lib.content").Task(
        source_id=3,
        task_id=5,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=importlib.import_module("core.lib.content").Task.extract_dag_from_dag_deployment(
            {"detector": {"service": {"service_name": "detector", "execute_device": "edge-node"}, "next_nodes": []}}
        ),
        flow_index="detector",
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="artifact.bin",
    )
    monkeypatch.setattr(task, "get_delay_info", lambda: "delay-info")
    monkeypatch.setattr(backend_core_module.Task, "deserialize", classmethod(lambda cls, data: task))
    backend_core_instance.task_results = {3: Queue()}
    backend_core_instance.source_open = True

    backend_core_instance.parse_task_result([task.serialize(), "", None])
    vis_calls = []
    removed = []
    monkeypatch.setattr(backend_core_instance, "prepare_result_visualization_data", lambda cur_task, is_last=False: vis_calls.append((cur_task.get_task_id(), is_last)) or [{"frame": 1}])
    monkeypatch.setattr(backend_core_instance, "get_file_result", lambda file_name: file_name)
    monkeypatch.setattr(backend_core_module.FileOps, "remove_file", lambda file_path: removed.append(file_path))

    results = backend_core_instance.fetch_visualization_data(3)

    assert vis_calls == [(5, True)]
    assert removed == ["artifact.bin"]
    assert results == [{"task_id": 5, "data": [{"frame": 1}]}]


@pytest.mark.unit
def test_backend_core_install_state_and_log_name_behaviors(backend_core_instance, monkeypatch):
    backend_core_module = importlib.import_module("backend_core")
    monkeypatch.setattr(
        backend_core_module.KubeHelper,
        "check_pods_without_string_exists",
        staticmethod(lambda namespace, exclude_str_list=None: True),
    )

    backend_core_instance.template_helper = SimpleNamespace(load_base_info=lambda: {"log-file-name": "dayu.log"})
    assert backend_core_instance.get_log_file_name() == "dayu"
    assert backend_core_instance.check_install_state() is True
    assert backend_core_instance.install_state is True

    backend_core_instance.template_helper = SimpleNamespace(load_base_info=lambda: {"log-file-name": ""})
    assert backend_core_instance.get_log_file_name() is None
