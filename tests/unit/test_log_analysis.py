import gzip
import json

import pytest

from tools import log_analysis


def _service_record(name, device="", transmit_time=0.0, execute_time=0.0, real_execute_time=0.0):
    return {
        "service_name": name,
        "execute_device": device,
        "execute_data": {
            "transmit_time": transmit_time,
            "execute_time": execute_time,
            "real_execute_time": real_execute_time,
        },
        "content": None,
        "scenario": {},
        "tmp_data": {},
    }


def _task_record(task_id, source_device, services, root_uuid=None):
    dag = {
        log_analysis.START_NODE: {
            "service": _service_record(log_analysis.START_NODE),
            "prev_nodes": [],
            "next_nodes": [services[0][0]],
        },
        log_analysis.END_NODE: {
            "service": _service_record(log_analysis.END_NODE),
            "prev_nodes": [services[-1][0]],
            "next_nodes": [],
        },
    }

    for index, (service_name, device, transmit_time, execute_time, real_execute_time) in enumerate(services):
        dag[service_name] = {
            "service": _service_record(service_name, device, transmit_time, execute_time, real_execute_time),
            "prev_nodes": [log_analysis.START_NODE if index == 0 else services[index - 1][0]],
            "next_nodes": [log_analysis.END_NODE if index == len(services) - 1 else services[index + 1][0]],
        }

    return {
        "source_id": task_id,
        "task_id": task_id,
        "source_device": source_device,
        "all_edge_devices": sorted({device for _, device, *_ in services if device}),
        "dag": dag,
        "deployment": None,
        "cur_flow_index": services[-1][0],
        "past_flow_index": services[-2][0] if len(services) > 1 else log_analysis.START_NODE,
        "meta_data": {"frame": task_id},
        "raw_meta_data": {},
        "tmp_data": {},
        "hash_data": [],
        "file_path": f"/tmp/task-{task_id}.json",
        "task_uuid": f"task-{task_id}",
        "parent_uuid": "",
        "root_uuid": root_uuid or f"root-{task_id}",
    }


@pytest.mark.unit
def test_load_tasks_reads_exported_log_file(tmp_path):
    log_file = tmp_path / "sample-log.json"
    log_file.write_text(
        json.dumps(
            [
                _task_record(
                    1,
                    "camera-a",
                    [
                        ("detector", "edge-a", 0.5, 1.5, 1.0),
                        ("tracker", "edge-b", 0.2, 0.8, 0.7),
                    ],
                )
            ]
        ),
        encoding="utf-8",
    )

    tasks = log_analysis.load_tasks(log_file)

    assert len(tasks) == 1
    assert tasks[0]["source_device"] == "camera-a"
    assert tasks[0]["dag"]["detector"]["service"]["execute_device"] == "edge-a"


@pytest.mark.unit
def test_load_tasks_reads_gzipped_jsonl_export(tmp_path):
    log_file = tmp_path / "sample-log.jsonl.gz"
    with gzip.open(log_file, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_task_record(1, "camera-a", [("detector", "edge-a", 0.5, 1.5, 1.0)])))
        fh.write("\n")
        fh.write(json.dumps(_task_record(2, "camera-b", [("tracker", "edge-b", 0.2, 0.8, 0.7)])))
        fh.write("\n")

    tasks = log_analysis.load_tasks(log_file)

    assert [task["task_id"] for task in tasks] == [1, 2]
    assert tasks[1]["dag"]["tracker"]["service"]["execute_device"] == "edge-b"


@pytest.mark.unit
def test_summarize_tasks_rolls_up_devices_and_service_timings():
    tasks = [
        _task_record(
            1,
            "camera-a",
            [
                ("detector", "edge-a", 0.5, 1.5, 1.0),
                ("tracker", "edge-b", 0.2, 0.8, 0.7),
            ],
            root_uuid="root-1",
        ),
        _task_record(
            2,
            "camera-b",
            [("detector", "edge-a", 0.7, 1.7, 1.1)],
            root_uuid="root-2",
        ),
    ]

    summary = log_analysis.summarize_tasks(tasks)

    assert summary["task_count"] == 2
    assert summary["root_task_count"] == 2
    assert summary["source_devices"] == {"camera-a": 1, "camera-b": 1}
    assert summary["edge_devices"] == {"edge-a": 2, "edge-b": 1}
    assert summary["average_task_latency"] == pytest.approx(2.7)
    assert summary["services"]["detector"] == {
        "occurrences": 2,
        "avg_execute_time": 1.6,
        "avg_real_execute_time": 1.05,
        "avg_transmit_time": 0.6,
        "execute_devices": {"edge-a": 2},
    }
    assert summary["services"]["tracker"] == {
        "occurrences": 1,
        "avg_execute_time": 0.8,
        "avg_real_execute_time": 0.7,
        "avg_transmit_time": 0.2,
        "execute_devices": {"edge-b": 1},
    }


@pytest.mark.unit
def test_main_supports_json_output(tmp_path, capsys):
    log_file = tmp_path / "sample-log.json"
    log_file.write_text(
        json.dumps(
            [
                _task_record(
                    3,
                    "camera-c",
                    [("detector", "edge-c", 0.3, 1.2, 0.9)],
                )
            ]
        ),
        encoding="utf-8",
    )

    exit_code = log_analysis.main(["--log", str(log_file), "--output-format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["task_count"] == 1
    assert payload["services"]["detector"]["execute_devices"] == {"edge-c": 1}


@pytest.mark.unit
def test_main_returns_error_for_missing_log_file(capsys):
    exit_code = log_analysis.main(["--log", "/tmp/does-not-exist.json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "does not exist" in captured.err
