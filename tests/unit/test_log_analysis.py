import gzip
import json
from pathlib import Path

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

    summary = log_analysis.summarize_tasks(tasks, slo_target_seconds=2.5)

    assert summary["task_count"] == 2
    assert summary["root_task_count"] == 2
    assert summary["source_devices"] == {"camera-a": 1, "camera-b": 1}
    assert summary["edge_devices"] == {"edge-a": 2, "edge-b": 1}
    assert summary["average_task_latency"] == pytest.approx(2.7)
    assert summary["task_latency_seconds"] == {
        "count": 2,
        "avg": 2.7,
        "min": 2.4,
        "max": 3.0,
        "p50": 2.7,
        "p90": 2.94,
        "p95": 2.97,
        "p99": 2.994,
        "slo_target_seconds": 2.5,
        "slo_hit_count": 1,
        "slo_miss_count": 1,
        "slo_compliance_rate": 0.5,
    }
    assert summary["services"]["detector"] == {
        "occurrences": 2,
        "avg_execute_time": 1.6,
        "avg_real_execute_time": 1.05,
        "avg_transmit_time": 0.6,
        "avg_stage_latency": 2.2,
        "execute_devices": {"edge-a": 2},
        "execute_time_seconds": {
            "count": 2,
            "avg": 1.6,
            "min": 1.5,
            "max": 1.7,
            "p50": 1.6,
            "p90": 1.68,
            "p95": 1.69,
            "p99": 1.698,
        },
        "real_execute_time_seconds": {
            "count": 2,
            "avg": 1.05,
            "min": 1.0,
            "max": 1.1,
            "p50": 1.05,
            "p90": 1.09,
            "p95": 1.095,
            "p99": 1.099,
        },
        "transmit_time_seconds": {
            "count": 2,
            "avg": 0.6,
            "min": 0.5,
            "max": 0.7,
            "p50": 0.6,
            "p90": 0.68,
            "p95": 0.69,
            "p99": 0.698,
        },
        "stage_latency_seconds": {
            "count": 2,
            "avg": 2.2,
            "min": 2.0,
            "max": 2.4,
            "p50": 2.2,
            "p90": 2.36,
            "p95": 2.38,
            "p99": 2.396,
        },
    }
    assert summary["services"]["tracker"] == {
        "occurrences": 1,
        "avg_execute_time": 0.8,
        "avg_real_execute_time": 0.7,
        "avg_transmit_time": 0.2,
        "avg_stage_latency": 1.0,
        "execute_devices": {"edge-b": 1},
        "execute_time_seconds": {
            "count": 1,
            "avg": 0.8,
            "min": 0.8,
            "max": 0.8,
            "p50": 0.8,
            "p90": 0.8,
            "p95": 0.8,
            "p99": 0.8,
        },
        "real_execute_time_seconds": {
            "count": 1,
            "avg": 0.7,
            "min": 0.7,
            "max": 0.7,
            "p50": 0.7,
            "p90": 0.7,
            "p95": 0.7,
            "p99": 0.7,
        },
        "transmit_time_seconds": {
            "count": 1,
            "avg": 0.2,
            "min": 0.2,
            "max": 0.2,
            "p50": 0.2,
            "p90": 0.2,
            "p95": 0.2,
            "p99": 0.2,
        },
        "stage_latency_seconds": {
            "count": 1,
            "avg": 1.0,
            "min": 1.0,
            "max": 1.0,
            "p50": 1.0,
            "p90": 1.0,
            "p95": 1.0,
            "p99": 1.0,
        },
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
    assert payload["task_latency_seconds"]["p95"] == 1.5


@pytest.mark.unit
def test_generate_report_supports_full_json_with_detailed_tasks(tmp_path):
    log_file = tmp_path / "sample-log.json"
    log_file.write_text(
        json.dumps(
            [
                _task_record(
                    4,
                    "camera-d",
                    [("detector", "edge-d", 0.4, 1.1, 0.9)],
                )
            ]
        ),
        encoding="utf-8",
    )

    payload = json.loads(log_analysis.generate_report(log_file, output_format="full-json", slo_target_seconds=2.0))

    assert payload["summary"]["task_count"] == 1
    assert payload["tasks"][0]["task_id"] == 4
    assert payload["tasks"][0]["analysis"]["task_latency_seconds"] == 1.5
    assert payload["tasks"][0]["analysis"]["services"][0]["stage_latency"] == 1.5


@pytest.mark.unit
def test_main_supports_output_file_for_full_json(tmp_path, capsys):
    log_file = tmp_path / "sample-log.json"
    output_file = tmp_path / "reports" / "full-report.json"
    log_file.write_text(
        json.dumps(
            [
                _task_record(
                    5,
                    "camera-e",
                    [("detector", "edge-e", 0.3, 0.9, 0.6)],
                )
            ]
        ),
        encoding="utf-8",
    )

    exit_code = log_analysis.main(
        [
            "--log",
            str(log_file),
            "--output-format",
            "full-json",
            "--output-file",
            str(output_file),
            "--slo-seconds",
            "2.0",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(output_file.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert Path(output_file).exists()
    assert "Report written to" in captured.out
    assert payload["summary"]["task_latency_seconds"]["slo_compliance_rate"] == 1.0


@pytest.mark.unit
def test_main_returns_error_for_missing_log_file(capsys):
    exit_code = log_analysis.main(["--log", "/tmp/does-not-exist.json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "does not exist" in captured.err
