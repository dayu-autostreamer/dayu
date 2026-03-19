import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from core.lib.common import Counter, TaskConstant
from core.lib.content import Task


after_schedule_module = importlib.import_module("core.lib.algorithms.after_schedule_operation")
after_schedule_simple_module = importlib.import_module("core.lib.algorithms.after_schedule_operation.simple_operation")
after_schedule_casva_module = importlib.import_module("core.lib.algorithms.after_schedule_operation.casva_operation")
before_schedule_module = importlib.import_module("core.lib.algorithms.before_schedule_operation")
before_submit_module = importlib.import_module("core.lib.algorithms.before_submit_task_operation")
before_submit_chameleon_module = importlib.import_module(
    "core.lib.algorithms.before_submit_task_operation.chameleon_operation"
)
data_getter_filter_module = importlib.import_module("core.lib.algorithms.data_getter_filter")
data_getter_filter_casva_module = importlib.import_module(
    "core.lib.algorithms.data_getter_filter.casva_getter_filter"
)
frame_filter_dynamic_module = importlib.import_module("core.lib.algorithms.frame_filter.dynamic_filter")
frame_filter_motion_module = importlib.import_module("core.lib.algorithms.frame_filter.motion_filter")
frame_filter_simple_module = importlib.import_module("core.lib.algorithms.frame_filter.simple_filter")
frame_process_adaptive_module = importlib.import_module("core.lib.algorithms.frame_process.adaptive_process")
frame_process_simple_module = importlib.import_module("core.lib.algorithms.frame_process.simple_process")
scenario_extraction_module = importlib.import_module("core.lib.algorithms.scenario_extraction")
task_queue_module = importlib.import_module("core.lib.algorithms.task_queue")

BaseASOperation = importlib.import_module(
    "core.lib.algorithms.after_schedule_operation.base_operation"
).BaseASOperation
BaseBSOperation = importlib.import_module(
    "core.lib.algorithms.before_schedule_operation.base_operation"
).BaseBSOperation
BaseBSTOperation = importlib.import_module(
    "core.lib.algorithms.before_submit_task_operation.base_operation"
).BaseBSTOperation
BaseCompress = importlib.import_module(
    "core.lib.algorithms.frame_compress.base_compress"
).BaseCompress


def load_frame_compress_module(module_name):
    package_dir = (
        Path(__file__).resolve().parents[4]
        / "dependency"
        / "core"
        / "lib"
        / "algorithms"
        / "frame_compress"
    )
    package_name = f"_test_frame_compress_{uuid4().hex}"
    package_module = ModuleType(package_name)
    package_module.__path__ = [str(package_dir)]
    sys.modules[package_name] = package_module

    original_register = importlib.import_module("core.lib.common.class_factory").ClassFactory.register
    importlib.import_module("core.lib.common.class_factory").ClassFactory.register = staticmethod(
        lambda *args, **kwargs: (lambda cls: cls)
    )
    try:
        base_spec = importlib.util.spec_from_file_location(
            f"{package_name}.base_compress",
            package_dir / "base_compress.py",
        )
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules[f"{package_name}.base_compress"] = base_module
        base_spec.loader.exec_module(base_module)

        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{module_name}",
            package_dir / f"{module_name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{module_name}"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        importlib.import_module("core.lib.common.class_factory").ClassFactory.register = original_register


simple_compress_module = load_frame_compress_module("simple_compress")
casva_compress_module = load_frame_compress_module("casva_compress")
BaseFilter = importlib.import_module(
    "core.lib.algorithms.frame_filter.base_filter"
).BaseFilter
BaseProcess = importlib.import_module(
    "core.lib.algorithms.frame_process.base_process"
).BaseProcess
BaseDataGetterFilter = importlib.import_module(
    "core.lib.algorithms.data_getter_filter.base_getter_filter"
).BaseDataGetterFilter
BaseExtraction = importlib.import_module(
    "core.lib.algorithms.scenario_extraction.base_extraction"
).BaseExtraction
BaseQueue = importlib.import_module("core.lib.algorithms.task_queue.base_queue").BaseQueue


def service_entry(name, *, execute_device="", next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": execute_device,
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_task():
    dag = Task.extract_dag_from_dict(
        {
            "detector": service_entry("detector", execute_device="edge-a", next_nodes=["classifier"]),
            "classifier": service_entry("classifier", execute_device="cloud-a"),
        }
    )
    task = Task(
        source_id=1,
        task_id=2,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        metadata={"fps": 10, "buffer_size": 2, "resolution": "720p", "encoding": "mp4v"},
        raw_metadata={"fps": 20, "buffer_size": 2, "resolution": "1080p", "encoding": "mp4v"},
        file_path="payload.mp4",
        hash_data=["hash-0"],
    )
    task.get_service("detector").set_content_data([([[1, 1, 4, 4]], [0.9])])
    task.get_service("detector").set_scenario_data({"obj_num": [2, 4]})
    task.get_service("classifier").set_content_data([(["car"], [0.9])])
    task.get_service("detector").set_execute_time(0.4)
    task.get_service("classifier").set_execute_time(0.7)
    task.set_tmp_data({"file_size": 1.0, "file_dynamics": 0.2})
    task.set_flow_index("detector")
    return task


@pytest.mark.unit
def test_algorithm_base_contracts_raise_or_default_as_expected():
    with pytest.raises(NotImplementedError):
        BaseASOperation()(SimpleNamespace(), None)
    with pytest.raises(NotImplementedError):
        BaseBSOperation()(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        BaseBSTOperation()(SimpleNamespace(), SimpleNamespace())
    with pytest.raises(NotImplementedError):
        BaseDataGetterFilter()(SimpleNamespace())
    with pytest.raises(NotImplementedError):
        BaseFilter()(SimpleNamespace(), object())
    with pytest.raises(NotImplementedError):
        BaseProcess()(SimpleNamespace(), object(), "1080p", "720p")
    with pytest.raises(NotImplementedError):
        BaseCompress()(SimpleNamespace(), [np.zeros((2, 2, 3), dtype=np.uint8)], "file.mp4")
    with pytest.raises(NotImplementedError):
        BaseQueue().get()
    with pytest.raises(NotImplementedError):
        BaseQueue().size()
    with pytest.raises(NotImplementedError):
        BaseQueue().put(object())
    with pytest.raises(NotImplementedError):
        BaseQueue().empty()
    assert BaseExtraction()([], None) is None


@pytest.mark.unit
def test_before_schedule_operations_export_source_metadata_and_reset_filter():
    task = build_task()
    getter_filter = SimpleNamespace(skip_count=3, reset_filter=lambda: setattr(getter_filter, "skip_count", 0))
    system = SimpleNamespace(
        source_id=7,
        raw_meta_data={"fps": 20, "buffer_size": 2},
        local_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        task_dag=task.get_dag(),
        getter_filter=getter_filter,
        temp_encoded_frame="frame-b64",
        temp_hash_code="hash-1",
    )

    simple_payload = before_schedule_module.SimpleBSOperation()(system)
    casva_payload = before_schedule_module.CASVABSOperation()(system)
    chameleon_payload = before_schedule_module.ChameleonBSOperation()(system)

    assert simple_payload["source_id"] == 7
    assert simple_payload["meta_data"] == {"fps": 20, "buffer_size": 2}
    assert simple_payload["source_device"] == "edge-a"
    assert simple_payload["all_edge_devices"] == ["edge-a", "edge-b"]
    assert set(simple_payload["dag"]) == {TaskConstant.START.value, "detector", "classifier", TaskConstant.END.value}

    assert casva_payload["skip_count"] == 3
    assert getter_filter.skip_count == 0
    assert chameleon_payload["frame"] == "frame-b64"
    assert chameleon_payload["hash_code"] == "hash-1"


@pytest.mark.unit
def test_after_schedule_operations_keep_local_execution_or_apply_scheduler_plan(monkeypatch):
    task = build_task()
    task_dag = task.get_dag()
    system = SimpleNamespace(
        task_dag=task_dag,
        local_device="edge-a",
        meta_data={"fps": 10, "buffer_size": 2},
        service_deployment={},
    )
    monkeypatch.setattr(after_schedule_simple_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))
    monkeypatch.setattr(after_schedule_casva_module.NodeInfo, "get_cloud_node", staticmethod(lambda: "cloud-a"))

    after_schedule_module.SimpleASOperation()(system, None)
    assert all(
        node.service.get_execute_device() == "edge-a"
        for name, node in system.task_dag.nodes.items()
        if name not in (TaskConstant.START.value, TaskConstant.END.value)
    )

    dag_deployment = Task.extract_dag_deployment_from_dag(build_task().get_dag())
    dag_deployment["detector"]["service"]["execute_device"] = "edge-b"
    scheduler_response = {
        "plan": {"fps": 5, "dag": dag_deployment},
        "deployment": {"detector": ["edge-b"]},
    }

    after_schedule_module.SimpleASOperation()(system, scheduler_response)
    assert system.meta_data["fps"] == 5
    assert system.service_deployment == {"detector": ["edge-b"]}
    assert system.task_dag.get_node("detector").service.get_execute_device() == "edge-b"
    assert system.task_dag.get_end_node().service.get_execute_device() == "cloud-a"

    casva_system = SimpleNamespace(
        task_dag=build_task().get_dag(),
        local_device="edge-a",
        meta_data={"fps": 10},
        service_deployment={},
    )
    after_schedule_module.CASVAASOperation()(casva_system, {"plan": {"dag": dag_deployment}, "deployment": {}})
    assert casva_system.meta_data["qp"] == 23


@pytest.mark.unit
def test_before_submit_task_operations_track_file_metadata_and_last_frame_state(monkeypatch, tmp_path):
    task = build_task()
    current_task = build_task()
    compressed_file = tmp_path / "buffer.mp4"
    compressed_file.write_bytes(b"payload" * 4)
    new_task = SimpleNamespace(
        get_file_path=lambda: str(compressed_file),
        get_hash_data=lambda: ["hash-new"],
    )
    system = SimpleNamespace(current_task=current_task)

    before_submit_module.SimpleBSTOperation()(system, new_task)

    before_submit_module.CEVASBSTOperation()(system, new_task)
    assert current_task.get_tmp_data()["file_size"] == pytest.approx(compressed_file.stat().st_size / 1024)

    class DummyCap:
        def __init__(self, frames):
            self.frames = iter(frames)

        def read(self):
            return next(self.frames)

    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda file_name: DummyCap([(True, np.zeros((4, 4, 3), dtype=np.uint8))]))
    monkeypatch.setattr(before_submit_chameleon_module.EncodeOps, "encode_image", staticmethod(lambda frame: "encoded-frame"))
    chameleon_system = SimpleNamespace(temp_encoded_frame="", temp_hash_code="")
    before_submit_module.ChameleonBSTOperation()(chameleon_system, new_task)
    assert chameleon_system.temp_encoded_frame == "encoded-frame"
    assert chameleon_system.temp_hash_code == "hash-new"

    monkeypatch.setattr(cv2, "VideoCapture", lambda file_name: DummyCap([(False, None)]))
    chameleon_system = SimpleNamespace(temp_encoded_frame="keep", temp_hash_code="keep")
    before_submit_module.ChameleonBSTOperation()(chameleon_system, new_task)
    assert chameleon_system.temp_encoded_frame == ""

    casva_system = SimpleNamespace(current_task=task)
    casva_operation = before_submit_module.CASVABSTOperation()
    casva_operation(casva_system, new_task)
    assert task.get_tmp_data()["file_size"] > 0
    assert task.get_tmp_data()["file_dynamics"] == 0

    casva_system.past_metadata = {"resolution": "480p", "fps": 5}
    casva_system.past_file_size = 0.5
    task.set_metadata({"resolution": "720p", "fps": 10, "buffer_size": 2, "encoding": "mp4v"})
    casva_operation(casva_system, new_task)
    assert "file_dynamics" in task.get_tmp_data()
    assert casva_operation.get_fps_adjust_mode(20, 20) == ("same", 0, 0)
    assert casva_operation.get_fps_adjust_mode(20, 15) == ("skip", 4, 0)
    assert casva_operation.get_fps_adjust_mode(20, 4) == ("remain", 0, 5)
    assert casva_operation.filter_frame(20, 10) is True


@pytest.mark.unit
def test_getter_filters_scenario_extractors_and_task_queues_cover_runtime_contracts(monkeypatch):
    time_values = iter([10.0, 20.0, 22.0])
    monkeypatch.setattr(data_getter_filter_casva_module.time, "time", lambda: next(time_values))

    casva_filter = data_getter_filter_module.CASVADataGetterFilter(data_coming_interval=5)
    assert casva_filter(SimpleNamespace()) is True
    assert casva_filter(SimpleNamespace()) is False
    assert casva_filter.skip_count == 1
    casva_filter.reset_filter()
    assert casva_filter.skip_count == 0
    assert data_getter_filter_module.SimpleDataGetterFilter()(SimpleNamespace()) is True

    task = build_task()
    results = [([[0, 0, 10, 10], [10, 10, 20, 20]], [0.9, 0.8]), ([], [])]
    assert scenario_extraction_module.ObjectNumberExtraction()(results, task) == [2, 0]
    obj_size = scenario_extraction_module.ObjectSizeExtraction()(results, task)
    assert obj_size[0] > 0
    assert obj_size[1] == 0
    assert scenario_extraction_module.ObjectVelocityExtraction()(results, task) is None

    simple_queue = task_queue_module.SimpleQueue()
    assert simple_queue.get() is None
    simple_queue.put("alpha")
    simple_queue.put("beta")
    assert simple_queue.size() == 2
    assert simple_queue.get() == "alpha"
    assert simple_queue.empty() is False

    limit_queue = task_queue_module.LimitQueue(max_size=2)
    limit_queue.put("one")
    limit_queue.put("two")
    limit_queue.put("three")
    assert limit_queue.size() >= 1
    assert limit_queue.get() is not None


@pytest.mark.unit
def test_frame_filters_cover_simple_dynamic_and_motion_strategies(monkeypatch):
    Counter.reset_all_counts()
    simple_filter = frame_filter_simple_module.SimpleFilter()
    system = SimpleNamespace(raw_meta_data={"fps": 20}, meta_data={"fps": 10})
    assert simple_filter(system, object()) is True
    assert simple_filter(system, object()) is False
    assert simple_filter(system, object()) is True

    time_values = iter([100.0, 100.0, 100.2, 101.0])
    random_values = iter([1.0, 4.0, 6.0])
    monkeypatch.setattr(frame_filter_dynamic_module.time, "time", lambda: next(time_values))
    monkeypatch.setattr(frame_filter_dynamic_module.random, "uniform", lambda a, b: next(random_values))
    dynamic_filter = frame_filter_dynamic_module.DynamicFilter(min_fps=1, max_fps=5, min_duration=4, max_duration=4)
    assert dynamic_filter(SimpleNamespace(), np.zeros((2, 2, 3), dtype=np.uint8)) is True
    assert dynamic_filter._calculate_target_fps(100.2) > 0

    dummy_bg = SimpleNamespace(apply=lambda frame: np.ones((4, 4), dtype=np.uint8))
    monkeypatch.setattr(frame_filter_motion_module.cv2, "createBackgroundSubtractorMOG2", lambda **kwargs: dummy_bg)
    monkeypatch.setattr(frame_filter_motion_module.cv2, "morphologyEx", lambda mask, op, kernel: mask)
    motion_filter = frame_filter_motion_module.MotionFilter(min_fps=2, max_fps=8, smoothing_factor=0.0)
    motion_system = SimpleNamespace(raw_meta_data={"fps": 20}, meta_data={"fps": 10})
    monkeypatch.setattr(motion_filter, "_calculate_motion", lambda frame: 0.06)
    assert motion_filter._calculate_target_fps(0.0) == 2
    assert motion_filter._calculate_target_fps(1.0) == motion_filter.max_fps
    assert motion_filter(motion_system, np.zeros((4, 4, 3), dtype=np.uint8)) in {True, False}
    assert motion_filter.get_fps_adjust_mode(20, 20) == ("same", 0, 0)


@pytest.mark.unit
def test_frame_process_and_compress_algorithms_transform_frames_and_cleanup(monkeypatch, tmp_path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    process_system = SimpleNamespace(meta_data={"resolution": "480p", "buffer_size": 1})

    import cv2

    monkeypatch.setattr(cv2, "resize", lambda image, resolution: np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
    assert frame_process_simple_module.SimpleProcess()(process_system, frame, "720p", "720p").shape == frame.shape
    assert frame_process_simple_module.SimpleProcess()(process_system, frame, "720p", "480p").shape[0:2] == (480, 640)

    dummy_bg = SimpleNamespace(apply=lambda image: np.ones((8, 8), dtype=np.uint8))
    monkeypatch.setattr(frame_process_adaptive_module.cv2, "createBackgroundSubtractorMOG2", lambda **kwargs: dummy_bg)
    monkeypatch.setattr(frame_process_adaptive_module.cv2, "resize", lambda image, resolution: np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8))
    monkeypatch.setattr(frame_process_adaptive_module.cv2, "morphologyEx", lambda mask, op, kernel: mask)
    monkeypatch.setattr(frame_process_adaptive_module.cv2, "findContours", lambda mask, mode, method: ([np.array([[[1, 1]], [[1, 3]], [[3, 3]], [[3, 1]]])], None))
    monkeypatch.setattr(frame_process_adaptive_module.cv2, "contourArea", lambda contour: 1600)
    monkeypatch.setattr(frame_process_adaptive_module.cv2, "boundingRect", lambda contour: (1, 1, 2, 2))
    adaptive = frame_process_adaptive_module.AdaptiveProcess()
    generated = []
    monkeypatch.setattr(adaptive, "generate_roi_file", lambda system: generated.append(system.source_id))
    adaptive_system = SimpleNamespace(source_id=3, task_id=5, meta_data={"resolution": "480p", "buffer_size": 1})
    processed_frame, valid_rois = adaptive(adaptive_system, frame)
    assert processed_frame.shape[0:2] == (480, 640)
    assert valid_rois == [(1, 1, 3, 3)]
    assert generated == [3]
    assert adaptive.generate_roi_message([(1, 2, 5, 6)]).startswith("1 -10 1 2 4 4")

    writer_events = []

    class DummyWriter:
        def __init__(self, path, fourcc, fps, shape):
            writer_events.append(("open", path, fourcc, fps, shape))
            self.path = path

        def write(self, image):
            writer_events.append(("write", self.path, image.shape))

        def release(self):
            writer_events.append(("release", self.path))

    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *args: "mp4v")
    monkeypatch.setattr(cv2, "VideoWriter", DummyWriter)

    frame_buffer = [np.zeros((4, 6, 3), dtype=np.uint8), np.zeros((4, 6, 3), dtype=np.uint8)]
    compressor_system = SimpleNamespace(meta_data={"encoding": "mp4v", "qp": 28})
    output_path = tmp_path / "clip.mp4"

    simple_compress_module.SimpleCompress()(compressor_system, frame_buffer, str(output_path))
    assert any(event[0] == "write" for event in writer_events)

    removed = []
    commands = []
    monkeypatch.setattr(casva_compress_module.FileOps, "remove_file", lambda file_path: removed.append(file_path))
    monkeypatch.setattr(casva_compress_module.os, "system", lambda cmd: commands.append(cmd) or 0)
    casva_compress_module.CasvaCompress()(compressor_system, frame_buffer, str(output_path))
    assert commands and "ffmpeg -i" in commands[0]
    assert removed == [f"tmp_{output_path}"]
