import importlib
import sys

import numpy as np
import pytest


accuracy_module = importlib.import_module("core.lib.estimation.accuracy_estimation")
flops_module = importlib.import_module("core.lib.estimation.model_flops_estimation")
overhead_module = importlib.import_module("core.lib.estimation.overhead_estimation")
time_module = importlib.import_module("core.lib.estimation.time_estimation")


AccEstimator = accuracy_module.AccEstimator
FlopsEstimator = flops_module.FlopsEstimator
OverheadEstimator = overhead_module.OverheadEstimator
TimeEstimator = time_module.TimeEstimator
Timer = time_module.Timer


class DummyTimingTask:
    def __init__(self):
        self.tmp_data = {}
        self.flow_index = 2
        self.events = []

    def get_source_id(self):
        return 1

    def get_task_id(self):
        return 9

    def get_root_uuid(self):
        return "uuid-9"

    def get_tmp_data(self):
        return self.tmp_data

    def get_flow_index(self):
        return self.flow_index

    def record_time_ticket_in_service(self, type_tag, is_end, time_ticket):
        self.events.append(("record", type_tag, is_end, time_ticket))

    def erase_time_ticket_in_service(self, type_tag, is_end):
        self.events.append(("erase", type_tag, is_end))


@pytest.mark.unit
def test_time_estimator_and_timer_cover_recording_erasing_and_logging(monkeypatch):
    timestamps = iter([10.0, 10.25, 20.0, 20.5, 30.0, 31.0, 40.0, 50.0, 60.0, 70.0])
    logs = []

    monkeypatch.setattr(time_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(time_module.LOGGER, "info", lambda message: logs.append(message))

    with Timer("unit-test-stage") as timer:
        pass

    assert timer.get_elapsed_time() == 0.25
    assert logs[-1].startswith("[unit-test-stage] Execution time:")

    with Timer() as plain_timer:
        pass
    assert plain_timer.get_elapsed_time() == 0.5
    assert logs[-1].startswith("Execution time:")

    data = {}
    duration, start_ts = TimeEstimator.record_ts(data, "task-start")
    assert duration == 0
    assert start_ts == 30.0

    duration, end_ts = TimeEstimator.record_ts(data, "task-start", is_end=True)
    assert duration == 1.0
    assert end_ts == 31.0
    assert data == {}

    task = DummyTimingTask()
    assert TimeEstimator.record_task_ts(task, "dispatch") == 0
    assert TimeEstimator.record_dag_ts(task, is_end=False, sub_tag="transmit") == 0
    assert task.events[-1][0:3] == ("record", "transmit", False)

    TimeEstimator.erase_dag_ts(task, is_end=False, sub_tag="transmit")
    assert task.events[-1] == ("erase", "transmit", False)

    TimeEstimator.erase_ts(task.get_tmp_data(), "missing-tag", non_exist_ok=True)
    with pytest.raises(KeyError, match="not existed"):
        TimeEstimator.erase_ts(task.get_tmp_data(), "missing-tag")

    with pytest.raises(AssertionError, match="has existed"):
        TimeEstimator.record_ts({"dup": 1.0}, "dup")

    with pytest.raises(AssertionError, match="does not exists"):
        TimeEstimator.record_ts({}, "missing-end", is_end=True)


@pytest.mark.unit
def test_accuracy_estimator_covers_frame_mapping_ground_truth_and_map_calculation(tmp_path):
    ground_truth = tmp_path / "ground_truth.txt"
    ground_truth.write_text(
        "0 0 0 10 10\n"
        "1 5 5 15 15\n"
        "2\n",
        encoding="utf-8",
    )

    estimator = AccEstimator(str(ground_truth))

    assert estimator.find_gt_frames_index(0.5, ["0", "1"]) == [[0, 1], [1, 2]]
    assert estimator.find_gt_frames_index(0.75, ["0", "1", "2", "3"]) == [[0], [1], [2], [3, 4]]
    assert estimator.find_gt_frames_index(1.0, ["2"]) == [[2]]

    assert estimator.get_frame_ground_truth(1, (0.5, 2.0)) == [{"bbox": [2.5, 10.0, 7.5, 30.0], "class": 1}]
    assert estimator.get_frame_ground_truth(10, (1.0, 1.0)) == []
    assert estimator.search_frame_index("7") == 7

    predictions = [([[0, 0, 10, 10]], [0.9])]
    assert estimator.calculate_accuracy(["0"], predictions, (1.0, 1.0), 1.0) == 1.0
    assert estimator.calculate_accuracy([], predictions, (1.0, 1.0), 1.0) == 1
    assert estimator.calculate_accuracy(["0"], [], (1.0, 1.0), 1.0) == 0

    assert AccEstimator.calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert AccEstimator.compute_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5])) == 0.75
    assert AccEstimator.calculate_map([], []) == 1
    assert AccEstimator.calculate_map([], [{"bbox": [0, 0, 1, 1], "class": 1}]) == 0

    low_ranked_match = AccEstimator.calculate_map(
        predictions=[
            {"bbox": [50, 50, 60, 60], "prob": 0.9, "class": 1},
            {"bbox": [0, 0, 10, 10], "prob": 0.8, "class": 1},
        ],
        ground_truths=[{"bbox": [0, 0, 10, 10], "class": 1}],
    )
    assert low_ranked_match == pytest.approx(0.5)

    mismatched_ground_truth = tmp_path / "mismatched_ground_truth.txt"
    mismatched_ground_truth.write_text("5 0 0 1 1\n", encoding="utf-8")
    mismatch_estimator = AccEstimator(str(mismatched_ground_truth))
    with pytest.raises(AssertionError, match="frame index 0 is not equal"):
        mismatch_estimator.get_frame_ground_truth(0, (1.0, 1.0))


@pytest.mark.unit
def test_overhead_estimator_initializes_logs_tracks_context_manager_and_parses_average(tmp_path, monkeypatch):
    class FakeTimer:
        def __init__(self, label=""):
            self.label = label
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = 100.0
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = 100.5

        def get_elapsed_time(self):
            return 0.5

    monkeypatch.setattr(overhead_module, "Timer", FakeTimer)
    monkeypatch.setattr(
        overhead_module.Context,
        "get_file_path",
        staticmethod(lambda relative_path: str(tmp_path / relative_path)),
    )

    estimator = OverheadEstimator("scheduler", "metrics", agent_id=7)
    with estimator:
        pass

    assert estimator.get_latest_overhead() == 0.5
    overhead_file = tmp_path / "metrics" / "scheduler_Overhead.txt"
    file_lines = overhead_file.read_text(encoding="utf-8").splitlines()
    assert file_lines[0] == "# Overhead Log for scheduler"
    assert file_lines[-1].startswith("7,")

    with overhead_file.open("a", encoding="utf-8") as fh:
        fh.write("0.75\n")

    assert estimator.get_average_overhead() == 0.625

    estimator.clear()
    cleared_lines = overhead_file.read_text(encoding="utf-8").splitlines()
    assert cleared_lines[-1] == "agent_id,timestamp,start_time,end_time,duration_seconds"
    assert estimator._format_dt(overhead_module.datetime(2024, 1, 1, 12, 0, 0)) == "2024-01-01 12:00:00.000000"


@pytest.mark.unit
def test_flops_estimator_prefers_model_info_and_falls_back_to_ptflops(monkeypatch):
    class InfoModel:
        def info(self):
            return 1, 2, 3, 4.5

    assert FlopsEstimator(InfoModel(), (3, 32, 32)).compute_flops() == 4.5e9

    fake_ptflops = type(
        "FakePtflops",
        (),
        {
            "get_model_complexity_info": staticmethod(
                lambda model, input_shape, print_per_layer_stat, as_strings, verbose: (12.0, None)
            )
        },
    )
    monkeypatch.setitem(sys.modules, "ptflops", fake_ptflops)

    class PlainModel:
        pass

    assert FlopsEstimator(PlainModel(), (3, 32, 32)).compute_flops() == 24.0

    class BrokenInfoModel:
        def info(self):
            raise RuntimeError("unsupported")

    assert FlopsEstimator(BrokenInfoModel(), (3, 32, 32)).compute_flops() == 24.0


@pytest.mark.unit
def test_overhead_estimator_handles_missing_logs_empty_files_and_lock_fallback(monkeypatch, tmp_path):
    class FakeTimer:
        def __init__(self, label=""):
            self.label = label
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

        def get_elapsed_time(self):
            return 1.25

    monkeypatch.setattr(overhead_module, "Timer", FakeTimer)
    monkeypatch.setattr(
        overhead_module.Context,
        "get_file_path",
        staticmethod(lambda relative_path: str(tmp_path / relative_path)),
    )
    monkeypatch.setattr(overhead_module.os, "fsync", lambda fd: None)

    estimator = OverheadEstimator("processor", "metrics")
    overhead_file = tmp_path / "metrics" / "processor_Overhead.txt"

    overhead_file.unlink()
    assert estimator.get_average_overhead() == 0.0

    overhead_file.write_text("", encoding="utf-8")
    estimator._ensure_file_initialized()
    assert overhead_file.read_text(encoding="utf-8").startswith("# Overhead Log for processor")

    monkeypatch.setattr(overhead_module, "fcntl", None)
    with overhead_file.open("r", encoding="utf-8") as handle:
        with estimator._lock_file_shared(handle):
            pass
    with overhead_file.open("a", encoding="utf-8") as handle:
        with estimator._lock_file(handle):
            handle.write("")

    with overhead_file.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("invalid-line\n")
        handle.write("legacy-text\n")
        handle.write("1,bad,bad,bad,not-a-number\n")

    estimator.write_overhead(1.25)

    file_lines = overhead_file.read_text(encoding="utf-8").splitlines()
    assert file_lines[-1].startswith("0,")
    assert ",,," in file_lines[-1]
    assert estimator.get_average_overhead() == 1.25
