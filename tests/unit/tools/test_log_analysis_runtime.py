import json
import runpy
import sys

import pytest

from tools import log_analysis


def make_task(*, dag=None, source_device="camera-a", edge_devices=None, root_uuid="root-1", task_uuid="task-1"):
    return {
        "source_device": source_device,
        "all_edge_devices": edge_devices if edge_devices is not None else ["edge-a"],
        "root_uuid": root_uuid,
        "task_uuid": task_uuid,
        "dag": dag
        or {
            log_analysis.START_NODE: {"service": {"service_name": log_analysis.START_NODE}},
            "detector": {
                "service": {
                    "service_name": "detector",
                    "execute_device": "",
                    "execute_data": {"execute_time": 1.0, "real_execute_time": 0.6, "transmit_time": 0.4},
                }
            },
            log_analysis.END_NODE: {"service": {"service_name": log_analysis.END_NODE}},
        },
    }


@pytest.mark.unit
def test_log_analysis_load_tasks_rejects_invalid_paths_and_payload_shapes(tmp_path):
    with pytest.raises(ValueError, match="is not a file"):
        log_analysis.load_tasks(tmp_path)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        log_analysis.load_tasks(invalid_json)

    wrong_shape = tmp_path / "wrong-shape.json"
    wrong_shape.write_text("[]", encoding="utf-8")
    original_json_load = log_analysis.json.load
    log_analysis.json.load = lambda fh: {"task": 1}
    with pytest.raises(ValueError, match="does not contain a task list"):
        log_analysis.load_tasks(wrong_shape)
    log_analysis.json.load = original_json_load

    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps([{"ok": True}, "bad-record"]), encoding="utf-8")
    with pytest.raises(ValueError, match="contains malformed task records"):
        log_analysis.load_tasks(malformed)


@pytest.mark.unit
def test_log_analysis_helpers_cover_whitespace_invalid_dag_and_empty_summaries(tmp_path):
    spaced_log = tmp_path / "spaced.json"
    spaced_log.write_text(" \n\t[]", encoding="utf-8")
    assert log_analysis.load_tasks(spaced_log) == []

    with pytest.raises(ValueError, match="missing a valid 'dag'"):
        log_analysis._iter_services({"dag": ["invalid"]})

    assert log_analysis.summarize_tasks([]) == {
        "task_count": 0,
        "root_task_count": 0,
        "average_task_latency": 0.0,
        "source_devices": {},
        "edge_devices": {},
        "services": {},
    }

    summary = log_analysis.summarize_tasks([make_task(source_device=None, edge_devices=[None, "", "edge-z"])])
    assert summary["source_devices"] == {"None": 1}
    assert summary["edge_devices"] == {"edge-z": 1}
    assert summary["services"]["detector"]["execute_devices"] == {"unknown": 1}


@pytest.mark.unit
def test_log_analysis_report_and_main_cover_text_output_and_error_paths(tmp_path, capsys):
    log_file = tmp_path / "report.json"
    log_file.write_text(json.dumps([make_task()]), encoding="utf-8")

    summary = {
        "task_count": 0,
        "root_task_count": 0,
        "average_task_latency": 0.0,
        "source_devices": {},
        "edge_devices": {},
        "services": {},
    }
    rendered = log_analysis.render_text_summary(log_file, summary)
    assert "Source devices:" in rendered
    assert "  - none" in rendered
    assert "Service summary:" in rendered

    report = log_analysis.generate_report(log_file, output_format="text")
    assert "Dayu Log Analysis Tool" in report
    assert "detector" in report

    exit_code = log_analysis.main(["--log", str(log_file)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Tasks analyzed: 1" in captured.out

    bad_task_file = tmp_path / "bad-task.json"
    bad_task_file.write_text(json.dumps([{"dag": ["invalid"]}]), encoding="utf-8")
    exit_code = log_analysis.main(["--log", str(bad_task_file)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "missing a valid 'dag'" in captured.err


@pytest.mark.unit
def test_log_analysis_module_entrypoint_exits_through_main(tmp_path, monkeypatch):
    log_file = tmp_path / "entrypoint.json"
    log_file.write_text(json.dumps([make_task()]), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["log_analysis.py", "--log", str(log_file), "--output-format", "json"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("tools.log_analysis", run_name="__main__")

    assert exc_info.value.code == 0
