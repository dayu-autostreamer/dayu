import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.lib.common import Queue
from core.lib.content import Task
from runtime_model import RuntimeDirectory, RuntimeEndpoint, RuntimeSlot, RuntimeUnit


def _directory():
    def unit(component, port):
        slot = RuntimeSlot(component, "cloud-a", "cloud")
        runtime_id = slot.runtime_name(4, "install-a")
        return RuntimeUnit(
            slot,
            runtime_id,
            4,
            "a" * 64,
            RuntimeEndpoint(
                f"{runtime_id}.dayu.svc.cluster.local",
                port,
                runtime_service_uid=f"{component}-runtime-uid",
                service_uid=f"{component}-service-uid",
                pod_uid=f"{component}-pod-uid",
            ),
        )

    return RuntimeDirectory("install-a", 4, (unit("scheduler", 9001), unit("distributor", 9003)))


def _backend(mounted_runtime):
    from backend_core import BackendCore

    return BackendCore()


def test_backend_core_log_snapshot_compaction_and_record_count(
        mounted_runtime, monkeypatch, tmp_path,
):
    backend = _backend(mounted_runtime)
    monkeypatch.chdir(tmp_path)
    backend.system_log_store_path = str(tmp_path / "system.jsonl")
    backend.system_log_retention_records = 2
    backend.system_log_compact_interval = 1

    for second in range(4):
        backend._append_system_log_snapshot({"timestamp": f"10:00:0{second}", "data": [second]})
    backend.system_log_record_count = 4

    assert backend._count_jsonl_records(backend.system_log_store_path) == 4
    backend._maybe_compact_system_log_store_locked()

    lines = Path(backend.system_log_store_path).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["timestamp"] for line in lines] == ["10:00:02", "10:00:03"]
    assert backend.system_log_record_count == 2


def test_backend_core_urls_are_bound_only_from_active_runtime_directory(
        mounted_runtime, monkeypatch, tmp_path,
):
    import backend_core as backend_core_module

    backend = _backend(mounted_runtime)
    backend.runtime_orchestrator = SimpleNamespace(active_directory=lambda: _directory())

    backend.get_resource_url()
    backend.get_result_url()
    backend.get_log_url()

    assert backend.resource_url.endswith(":9001/resource")
    assert backend.result_url.endswith(":9003/result")
    assert backend.result_file_url.endswith(":9003/file")
    assert backend.log_fetch_url.endswith(":9003/export_result_log")

    class FakeResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def iter_content(self, chunk_size=8192):
            yield from self.chunks

    def fake_http_request(url, method=None, **kwargs):
        if url.endswith("/file"):
            return FakeResponse([b"chunk-", b"data"])
        if url.endswith("/export_result_log"):
            return FakeResponse([b"gzip-data"])
        return None

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(backend_core_module, "http_request", fake_http_request)
    assert Path(backend.get_file_result("artifact.bin")).read_bytes() == b"chunk-data"
    assert backend.open_result_log_export_stream() is not None


def test_task_result_queue_remains_data_plane_only(mounted_runtime, monkeypatch, tmp_path):
    import backend_core as backend_core_module

    backend = _backend(mounted_runtime)
    monkeypatch.chdir(tmp_path)
    dag = Task.extract_dag_from_dag_deployment({
        "detector": {
            "service": {"service_name": "detector", "execute_device": "edge-a"},
            "next_nodes": [],
        }
    })
    task = Task(
        source_id=3,
        task_id=5,
        source_device="edge-a",
        all_edge_devices=["edge-a"],
        dag=dag,
        flow_index="detector",
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path="artifact.bin",
    )
    task.set_flow_index("_end")
    backend.task_results = {3: Queue()}
    backend.source_open = True
    monkeypatch.setattr(task, "get_delay_info", lambda: {})
    monkeypatch.setattr(
        backend_core_module.Task,
        "deserialize",
        classmethod(lambda cls, value: task),
    )

    backend.parse_task_result([task.serialize(), "", None])
    monkeypatch.setattr(backend, "get_file_result", lambda file_name: file_name)
    monkeypatch.setattr(
        backend,
        "prepare_result_visualization_data",
        lambda current, is_last=False: [{"frame": current.get_task_id()}],
    )
    monkeypatch.setattr(backend_core_module.FileOps, "remove_file", lambda path: None)

    assert backend.fetch_visualization_data(3) == [{
        "task_id": 5,
        "data": [{"frame": 5}],
    }]


def test_query_open_has_no_source_count_dependent_startup_delay(mounted_runtime):
    backend = _backend(mounted_runtime)
    backend.source_configs = [{
        "source_label": "source-a",
        "source_list": [{"id": source_id} for source_id in range(10)],
    }]
    backend.result_url = "http://distributor/result"
    backend._query_admission_enabled = True

    started = time.monotonic()
    assert backend.open_query("source-a") == (True, "Datasource open successfully")
    assert time.monotonic() - started < 0.5
    assert sorted(backend.task_results) == list(range(10))

    assert backend.close_query() == (True, "Datasource close successfully")
    assert backend.source_open is False
    assert backend.is_get_result is False
    assert backend.task_results == {}


def test_query_requires_committed_runtime_admission(mounted_runtime):
    backend = _backend(mounted_runtime)
    backend.source_configs = [{"source_label": "source-a", "source_list": [{"id": 0}]}]
    backend.result_url = "http://distributor/result"

    assert backend.open_query("source-a") == (
        False,
        "Runtime is not ready for datasource queries",
    )


def test_cancelled_result_collector_has_bounded_io_and_cannot_publish_late_results(
        mounted_runtime, monkeypatch,
):
    import backend_core as backend_core_module

    backend = _backend(mounted_runtime)
    backend.source_configs = [{"source_label": "source-a", "source_list": [{"id": 0}]}]
    backend.result_url = "http://distributor/result"
    backend._query_admission_enabled = True
    request_started = threading.Event()
    release_request = threading.Event()
    request = {}

    def fake_http_request(url, method=None, timeout=None, json=None, **kwargs):
        request.update({"url": url, "timeout": timeout, "json": json})
        request_started.set()
        assert release_request.wait(1)
        return {"time_ticket": 1, "result": ["late-result"]}

    monkeypatch.setattr(backend_core_module, "http_request", fake_http_request)
    monkeypatch.setattr(
        backend_core_module.Task,
        "deserialize",
        classmethod(lambda cls, value: pytest.fail("cancelled result must not be parsed")),
    )

    assert backend.open_query("source-a") == (True, "Datasource open successfully")
    with backend.query_lock:
        thread = backend._result_thread
        queue = backend.task_results[0]
    assert request_started.wait(2)

    assert backend.close_query() == (True, "Datasource close successfully")
    release_request.set()
    thread.join(timeout=1)

    assert thread.is_alive() is False
    assert queue.empty() is True
    assert request == {
        "url": "http://distributor/result",
        "timeout": 5.0,
        "json": {"time_ticket": 0, "size": 20},
    }


def test_install_state_is_derived_from_runtime_session(mounted_runtime):
    backend = _backend(mounted_runtime)
    state = {"session": SimpleNamespace(phase="active"), "directory": _directory()}
    backend.runtime_orchestrator = SimpleNamespace(
        current_session=lambda: state["session"],
        active_directory=lambda: state["directory"],
    )

    assert backend.check_install_state() is True
    assert backend.check_pods_running_state() is True

    state["session"] = SimpleNamespace(phase="failed")
    state["directory"] = None
    assert backend.check_install_state() is True
    assert backend.check_pods_running_state() is False

    backend.template_helper = SimpleNamespace(load_base_info=lambda: {"log-file-name": "dayu.log"})
    assert backend.get_log_file_name() == "dayu"
