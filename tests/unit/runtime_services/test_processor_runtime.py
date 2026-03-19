import importlib

import numpy as np
import pytest

from core.lib.content import Task


processor_base_module = importlib.import_module("core.processor.processor")
detector_module = importlib.import_module("core.processor.detector_processor")
detector_tracker_module = importlib.import_module("core.processor.detector_tracker_processor")
classifier_module = importlib.import_module("core.processor.classifier_processor")
roi_classifier_module = importlib.import_module("core.processor.roi_classifier_processor")


def build_task(service_names, flow_index, file_path="payload.bin"):
    dag_deployment = {}
    for index, service_name in enumerate(service_names):
        dag_deployment[service_name] = {
            "service": {"service_name": service_name, "execute_device": "edge-node"},
            "next_nodes": service_names[index + 1:index + 2],
        }
    return Task(
        source_id=1,
        task_id=2,
        source_device="edge-node",
        all_edge_devices=["edge-node"],
        dag=Task.extract_dag_from_dag_deployment(dag_deployment),
        flow_index=flow_index,
        metadata={"buffer_size": 1},
        raw_metadata={"buffer_size": 1},
        file_path=file_path,
    )


class DummyTimer:
    def __init__(self, label=""):
        self.label = label

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class FakeVideoCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def get(self, key):
        if not self.frames:
            return 0
        height, width = self.frames[0].shape[:2]
        if key == detector_module.cv2.CAP_PROP_FRAME_WIDTH:
            return width
        if key == detector_module.cv2.CAP_PROP_FRAME_HEIGHT:
            return height
        return 0


@pytest.fixture
def patch_processor_scenarios(monkeypatch):
    def fake_get_parameter(name, direct=False):
        if name == "SCENARIOS_EXTRACTORS":
            return ["objects"]
        raise AssertionError(f"Unexpected parameter request: {name}")

    def fake_get_algorithm(algorithm, al_name=None, **kwargs):
        if algorithm == "PRO_SCENARIO":
            return lambda result, task: len(result)
        raise AssertionError(f"Unexpected algorithm request: {algorithm}")

    monkeypatch.setattr(processor_base_module.Context, "get_parameter", staticmethod(fake_get_parameter))
    monkeypatch.setattr(processor_base_module.Context, "get_algorithm", staticmethod(fake_get_algorithm))
    for module in (detector_module, detector_tracker_module, classifier_module, roi_classifier_module):
        monkeypatch.setattr(module, "Timer", DummyTimer)


@pytest.mark.unit
def test_detector_processor_reads_frames_runs_detector_and_saves_scenario(patch_processor_scenarios, monkeypatch):
    class FakeDetector:
        flops = 321.0

        def __init__(self):
            self.calls = []

        def __call__(self, images):
            self.calls.append(images)
            return [
                [np.array([[0, 0, 2, 2]]), np.array([0.9]), np.array([1]), np.array([11])],
                [np.array([[1, 1, 3, 3]]), np.array([0.8]), np.array([2]), np.array([12])],
            ]

    detector = FakeDetector()
    frames = [np.ones((4, 6, 3), dtype=np.uint8), np.zeros((4, 6, 3), dtype=np.uint8)]

    monkeypatch.setattr(detector_module.Context, "get_instance", staticmethod(lambda name: detector))
    monkeypatch.setattr(detector_module.Context, "get_temporary_file_path", staticmethod(lambda path: path))
    monkeypatch.setattr(detector_module.cv2, "VideoCapture", lambda path: FakeVideoCapture(frames))

    processor = detector_module.DetectorProcessor()
    task = build_task(["detector"], "detector")

    result_task = processor(task)

    assert result_task is task
    assert len(detector.calls) == 1
    assert len(detector.calls[0]) == 2
    assert processor.frame_size == (6, 4)
    assert task.get_current_content() == [
        [[[0, 0, 2, 2]], [0.9], [1], [11]],
        [[[1, 1, 3, 3]], [0.8], [2], [12]],
    ]
    assert task.get_scenario_data("detector") == {"objects": 2}
    assert processor.flops == 321.0


@pytest.mark.unit
def test_detector_processor_handles_empty_inputs_and_missing_detector(patch_processor_scenarios, monkeypatch):
    warnings = []

    class FakeDetector:
        flops = 1.0

        def __call__(self, images):
            return images

    monkeypatch.setattr(detector_module.Context, "get_instance", staticmethod(lambda name: FakeDetector()))
    monkeypatch.setattr(detector_module.Context, "get_temporary_file_path", staticmethod(lambda path: path))
    monkeypatch.setattr(detector_module.cv2, "VideoCapture", lambda path: FakeVideoCapture([]))
    monkeypatch.setattr(detector_module.LOGGER, "warning", lambda message: warnings.append(message))

    processor = detector_module.DetectorProcessor()
    task = build_task(["detector"], "detector")

    assert processor(task) is None
    assert any("Image list length is 0" in message for message in warnings)

    processor.detector = None
    with pytest.raises(AssertionError, match="No detector defined"):
        processor.infer([np.ones((2, 2, 3), dtype=np.uint8)])


@pytest.mark.unit
def test_detector_tracker_processor_runs_detection_then_tracking_and_updates_task(patch_processor_scenarios, monkeypatch):
    class FakeDetector:
        flops = 222.0

        def __call__(self, images):
            assert len(images) == 1
            return [[np.array([[0, 0, 2, 2]]), np.array([0.9]), np.array([1]), np.array([7])]]

    class FakeTracker:
        def __init__(self):
            self.calls = []

        def __call__(self, tracking_list, first_frame, detection_output):
            self.calls.append((tracking_list, first_frame, detection_output))
            return [[np.array([[1, 1, 3, 3]]), np.array([0.8]), np.array([1]), np.array([7])]]

    tracker = FakeTracker()
    frames = [np.ones((4, 6, 3), dtype=np.uint8), np.full((4, 6, 3), 3, dtype=np.uint8)]

    def fake_get_instance(name):
        if name == "Detector":
            return FakeDetector()
        if name == "Tracker":
            return tracker
        raise AssertionError(f"Unexpected instance request: {name}")

    monkeypatch.setattr(detector_tracker_module.Context, "get_instance", staticmethod(fake_get_instance))
    monkeypatch.setattr(detector_tracker_module.Context, "get_temporary_file_path", staticmethod(lambda path: path))
    monkeypatch.setattr(detector_tracker_module.cv2, "VideoCapture", lambda path: FakeVideoCapture(frames))

    processor = detector_tracker_module.DetectorTrackerProcessor()
    task = build_task(["detector-tracker"], "detector-tracker")

    result_task = processor(task)

    assert result_task is task
    assert len(tracker.calls) == 1
    assert len(tracker.calls[0][0]) == 1
    assert tracker.calls[0][2][3].tolist() == [7]
    assert task.get_current_content() == [[[[1, 1, 3, 3]], [0.8], [1], [7]]]
    assert task.get_scenario_data("detector-tracker") == {"objects": 1}
    assert processor.flops == 222.0


@pytest.mark.unit
def test_classifier_processor_uses_previous_content_and_crops_faces(patch_processor_scenarios, monkeypatch):
    class FakeClassifier:
        flops = 111.0

        def __init__(self):
            self.faces = None

        def __call__(self, faces):
            self.faces = faces
            return ["adult", "child"]

    classifier = FakeClassifier()
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    monkeypatch.setattr(classifier_module.Context, "get_instance", staticmethod(lambda name: classifier))
    monkeypatch.setattr(classifier_module.Context, "get_temporary_file_path", staticmethod(lambda path: path))
    monkeypatch.setattr(classifier_module.cv2, "VideoCapture", lambda path: FakeVideoCapture([frame]))

    processor = classifier_module.ClassifierProcessor()
    task = build_task(["detector", "classifier"], "classifier")
    task.get_service("detector").set_content_data(
        [
            (
                [(-1, -1, 2, 3), (4, 1, 10, 6)],
                [0.9, 0.8],
                [1, 1],
                [101, 102],
            )
        ]
    )

    result_task = processor(task)

    assert result_task is task
    assert len(classifier.faces) == 2
    assert classifier.faces[0].shape == (3, 2, 3)
    assert classifier.faces[1].shape == (3, 2, 3)
    assert task.get_current_content() == [[["adult", "child"]]]
    assert task.get_scenario_data("classifier") == {"objects": 1}
    assert processor.flops == 111.0


@pytest.mark.unit
def test_classifier_processor_returns_task_when_previous_content_missing(patch_processor_scenarios, monkeypatch):
    warnings = []

    class FakeClassifier:
        flops = 100.0

        def __call__(self, faces):
            return faces

    monkeypatch.setattr(classifier_module.Context, "get_instance", staticmethod(lambda name: FakeClassifier()))
    monkeypatch.setattr(classifier_module.Context, "get_temporary_file_path", staticmethod(lambda path: path))
    monkeypatch.setattr(classifier_module.cv2, "VideoCapture", lambda path: FakeVideoCapture([]))
    monkeypatch.setattr(classifier_module.LOGGER, "warning", lambda message: warnings.append(message))

    processor = classifier_module.ClassifierProcessor()
    task = build_task(["detector", "classifier"], "classifier")

    assert processor(task) is task
    assert task.get_current_content() is None
    assert any("is none" in message for message in warnings)


@pytest.mark.unit
def test_roi_classifier_processor_resets_cache_uses_roi_ids_and_handles_missing_frames(
    patch_processor_scenarios, monkeypatch
):
    class FakeRoiClassifier:
        flops = 222.0

        def __init__(self):
            self.reset_count = 0
            self.calls = []

        def reset_cache(self):
            self.reset_count += 1

        def __call__(self, rois, roi_ids):
            self.calls.append((rois, roi_ids))
            return [{"roi_id": roi_id, "pixels": roi.shape[0] * roi.shape[1]} for roi, roi_id in zip(rois, roi_ids)]

    classifier = FakeRoiClassifier()
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    monkeypatch.setattr(roi_classifier_module.Context, "get_instance", staticmethod(lambda name: classifier))
    monkeypatch.setattr(roi_classifier_module.Context, "get_temporary_file_path", staticmethod(lambda path: path))
    monkeypatch.setattr(roi_classifier_module.cv2, "VideoCapture", lambda path: FakeVideoCapture([frame]))

    processor = roi_classifier_module.RoiClassifierProcessor()
    task = build_task(["detector", "roi-classifier"], "roi-classifier")
    task.get_service("detector").set_content_data(
        [
            (
                [(-1, -1, 2, 3), (3, 1, 8, 5)],
                [0.9, 0.8],
                [1, 1],
                [10, 20],
            ),
            (
                [(0, 0, 1, 1)],
                [0.7],
                [1],
                [30],
            ),
        ]
    )

    result_task = processor(task)

    assert result_task is task
    assert classifier.reset_count == 1
    assert len(classifier.calls) == 1
    assert classifier.calls[0][1] == [10, 20]
    assert task.get_current_content() == [
        [[{"roi_id": 10, "pixels": 6}, {"roi_id": 20, "pixels": 9}]],
        [[]],
    ]
    assert task.get_scenario_data("roi-classifier") == {"objects": 2}
    assert processor.flops == 222.0
