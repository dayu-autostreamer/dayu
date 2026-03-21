import asyncio
import gzip
import importlib
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks

from core.lib.content import Task


distributor_module = importlib.import_module("core.distributor.distributor")
distributor_server_module = importlib.import_module("core.distributor.distributor_server")


def build_task(source_id=1, task_id=1, file_path="payload.bin"):
    dag_deployment = {
        "detector": {
            "service": {"service_name": "detector", "execute_device": "edge-node"},
            "next_nodes": [],
        }
    }
    return Task(
        source_id=source_id,
        task_id=task_id,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=Task.extract_dag_from_dag_deployment(dag_deployment),
        flow_index="detector",
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path=file_path,
    )


class FakeMoment:
    def __init__(self, ts):
        self.ts = ts

    def timestamp(self):
        return self.ts


class FakeRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


@pytest.fixture
def distributor_instance(monkeypatch, tmp_path):
    db_path = tmp_path / "records.db"
    timestamps = iter([10.0, 20.0, 30.0, 40.0, 50.0])

    class FakeDateTime:
        @classmethod
        def now(cls):
            return FakeMoment(next(timestamps))

    def fake_get_parameter(name, default=None, direct=False):
        values = {
            "RESULT_LOG_EXPORT_BATCH_SIZE": 2,
            "RESULT_LOG_RETENTION_RECORDS": 2,
            "RESULT_LOG_RETENTION_PRUNE_INTERVAL": 1,
        }
        return values.get(name, default)

    monkeypatch.setattr(distributor_module, "FileNameConstant", SimpleNamespace(
        DISTRIBUTOR_RECORD=SimpleNamespace(value=str(db_path))
    ))
    monkeypatch.setattr(distributor_module, "datetime", FakeDateTime)
    monkeypatch.setattr(distributor_module.Context, "get_parameter", staticmethod(fake_get_parameter))
    monkeypatch.setattr(distributor_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-node"))
    monkeypatch.setattr(distributor_module.NodeInfo, "hostname2ip", staticmethod(lambda hostname: "10.0.0.8"))
    monkeypatch.setattr(distributor_module.PortInfo, "get_component_port", staticmethod(lambda component: 9001))
    monkeypatch.setattr(distributor_module.Distributor, "record_total_end_ts", staticmethod(lambda task: None))

    requests = []
    monkeypatch.setattr(
        distributor_module,
        "http_request",
        lambda url, method=None, **kwargs: requests.append((url, method, kwargs)),
    )

    distributor = distributor_module.Distributor()
    return SimpleNamespace(instance=distributor, db_path=db_path, requests=requests)


@pytest.mark.unit
def test_distributor_persists_queries_prunes_and_handles_duplicate_task(distributor_instance, monkeypatch):
    distributor = distributor_instance.instance
    warnings = []
    monkeypatch.setattr(distributor_module.LOGGER, "warning", lambda message: warnings.append(message))

    task1 = build_task(task_id=1)
    task2 = build_task(task_id=2)
    task3 = build_task(task_id=3)

    distributor.save_task_record(task1)
    distributor.save_task_record(task2)
    distributor.save_task_record(task3)
    distributor.save_task_record(task3)

    all_results = distributor.query_all_result()
    loaded = [Task.deserialize(payload) for payload in all_results["result"]]

    assert all_results["size"] == 2
    assert [task.get_task_id() for task in loaded] == [2, 3]
    assert distributor.query_result(time_ticket=0, size=2)["size"] == 2
    assert distributor.query_result(time_ticket=25, size=0)["size"] == 1
    assert distributor.query_results_by_time(15, 35)["size"] == 2
    assert distributor.query_results_by_time(15, 35, source_id=1)["size"] == 2
    assert any("already exists" in message for message in warnings)


@pytest.mark.unit
def test_distributor_exports_valid_json_skips_malformed_rows_and_can_clear_database(distributor_instance):
    distributor = distributor_instance.instance
    task = build_task(task_id=9)
    distributor.save_task_record(task)

    with sqlite3.connect(distributor.record_path) as conn:
        conn.execute(
            "INSERT INTO records (source_id, task_id, ctime, json) VALUES (?, ?, ?, ?)",
            (99, 1, 99.0, "{bad json"),
        )
        conn.commit()

    export_path = distributor.create_result_log_export_file()
    exported = json.loads(gzip.open(export_path, "rt", encoding="utf-8").read())

    assert exported == [task.to_dict()]

    distributor.clear_database()
    assert distributor.query_all_result() == {"result": [], "size": 0}
    assert distributor.is_database_empty() is False


@pytest.mark.unit
def test_distributor_records_transmit_time_and_forwards_scenario(distributor_instance, monkeypatch):
    distributor = distributor_instance.instance
    task = build_task(task_id=6)
    durations = []

    monkeypatch.setattr(
        distributor_module.TimeEstimator,
        "record_dag_ts",
        staticmethod(lambda current_task, is_end, sub_tag="transmit": durations.append((is_end, sub_tag)) or 0.25),
    )

    distributor.record_transmit_ts(task)
    distributor.distribute_data(task)

    assert durations == [(True, "transmit")]
    assert task.get_service("detector").get_transmit_time() == 0.25
    assert distributor_instance.requests == [
        (
            "http://10.0.0.8:9001/scenario",
            "POST",
            {"data": {"data": task.serialize()}},
        )
    ]


@pytest.mark.unit
def test_distributor_handles_empty_queries_scheduler_failures_and_export_cleanup(distributor_instance, monkeypatch, tmp_path):
    distributor = distributor_instance.instance
    warnings = []
    exceptions = []
    removed_paths = []

    monkeypatch.setattr(distributor_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(distributor_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))
    monkeypatch.setattr(
        distributor_module,
        "http_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("scheduler unavailable")),
    )
    monkeypatch.setattr(
        distributor_module.FileOps,
        "remove_file",
        lambda path: removed_paths.append(path),
    )

    assert distributor.query_result(time_ticket=0, size=10) == {"result": [], "time_ticket": 0, "size": 0}
    assert distributor.query_results_by_time(0, 1) == {"result": [], "size": 0}
    assert distributor.query_all_result() == {"result": [], "size": 0}

    task = build_task(task_id=11)
    distributor.distribute_data(task)

    assert any("Send scenario to scheduler failed" in message for message in warnings)
    assert exceptions == ["scheduler unavailable"]

    snapshot_path = tmp_path / "snapshot.db"
    export_path = tmp_path / "export.json.gz"
    monkeypatch.setattr(distributor, "_create_result_log_snapshot", lambda: str(snapshot_path))

    def fail_write(snapshot, export):
        raise RuntimeError("export failed")

    monkeypatch.setattr(distributor, "_write_result_log_export", fail_write)
    monkeypatch.setattr(
        distributor_module.tempfile,
        "NamedTemporaryFile",
        lambda **kwargs: SimpleNamespace(name=str(export_path), close=lambda: None),
    )

    with pytest.raises(RuntimeError, match="export failed"):
        distributor.create_result_log_export_file()

    assert removed_paths == [str(export_path), str(snapshot_path)]


@pytest.mark.unit
def test_distributor_snapshot_and_export_helpers_cover_cleanup_and_separator_rules(distributor_instance, monkeypatch, tmp_path):
    distributor = distributor_instance.instance
    removed_paths = []
    warnings = []
    monkeypatch.setattr(distributor_module.FileOps, "remove_file", lambda path: removed_paths.append(path))
    monkeypatch.setattr(distributor_module.LOGGER, "warning", lambda message: warnings.append(message))
    original_connect = distributor._connect

    with pytest.raises(Exception):
        monkeypatch.setattr(distributor, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("backup failed")))
        distributor._create_result_log_snapshot()
    assert removed_paths, "snapshot cleanup should remove the temporary db on backup failure"

    export_path = tmp_path / "export.json.gz"
    monkeypatch.setattr(
        distributor,
        "_iter_snapshot_records",
        lambda snapshot_path: iter(['{"task": 1}', "{bad json", '{"task": 2}']),
    )
    distributor._write_result_log_export("snapshot", str(export_path))
    exported = gzip.open(export_path, "rt", encoding="utf-8").read()
    assert exported.startswith("[\n")
    assert exported.endswith("]\n")
    assert '"task": 1' in exported and '"task": 2' in exported
    assert "    },\n    {" in exported
    assert warnings == ["[Distributor] Skip malformed result log record during export."]

    monkeypatch.setattr(distributor, "_connect", original_connect)


@pytest.mark.unit
def test_distributor_server_covers_background_queries_download_and_export(monkeypatch, tmp_path):
    calls = []
    export_path = tmp_path / "results.json.gz"
    export_path.write_text("payload", encoding="utf-8")

    class FakeDistributor:
        def record_transmit_ts(self, task):
            calls.append(("record", task.get_task_id()))

        def distribute_data(self, task):
            calls.append(("distribute", task.get_task_id()))

        def query_result(self, time_ticket, size):
            return {"time_ticket": time_ticket, "size": size, "result": ["ok"]}

        def query_results_by_time(self, start_time, end_time, source_id=None):
            return {"start": start_time, "end": end_time, "source_id": source_id}

        def query_all_result(self):
            return {"result": ["all"], "size": 1}

        def create_result_log_export_file(self):
            return str(export_path)

        def clear_database(self):
            calls.append(("clear", None))

        def is_database_empty(self):
            return False

    saved = []
    monkeypatch.setattr(distributor_server_module, "Distributor", FakeDistributor)
    monkeypatch.setattr(
        distributor_server_module.FileOps,
        "save_task_file",
        lambda task, payload: saved.append((task.get_file_path(), payload)),
    )

    server = distributor_server_module.DistributorServer()
    task = build_task(task_id=8, file_path="dist.bin")
    server.distribute_data_background(task.serialize(), b"payload")

    assert saved == [("dist.bin", b"payload")]
    assert calls[:2] == [("record", 8), ("distribute", 8)]

    result = asyncio.run(server.query_result(FakeRequest({"time_ticket": 3.5, "size": 7})))
    by_time = asyncio.run(
        server.query_results_by_time(FakeRequest({"start_time": 1.0, "end_time": 2.0, "source_id": 5}))
    )
    all_result = asyncio.run(server.query_all_result())
    empty_state = asyncio.run(server.is_database_empty())

    assert result == {"time_ticket": 3.5, "size": 7, "result": ["ok"]}
    assert by_time == {"start": 1.0, "end": 2.0, "source_id": 5}
    assert all_result == {"result": ["all"], "size": 1}
    assert empty_state is False

    missing = asyncio.run(server.download_file(FakeRequest({"file": str(tmp_path / "missing.bin")}), BackgroundTasks()))
    assert missing == b""

    download_tasks = BackgroundTasks()
    download_response = asyncio.run(server.download_file(FakeRequest({"file": str(export_path)}), download_tasks))
    export_tasks = BackgroundTasks()
    export_response = asyncio.run(server.export_result_log(export_tasks))
    asyncio.run(server.clear_database())

    assert download_response.path == str(export_path)
    assert export_response.path == str(export_path)
    assert export_response.filename.startswith("DAYU_RESULT_LOG_")
    assert len(download_tasks.tasks) == 1
    assert len(export_tasks.tasks) == 1
    assert calls[-1] == ("clear", None)
