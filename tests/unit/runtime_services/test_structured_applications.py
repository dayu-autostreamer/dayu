import importlib

import numpy as np
import pytest

from core.lib.common import Context
from core.applications.pedestrian_intent_recognition.pedestrian_intent_recognition import (
    PedestrianIntentRecognition,
)
from core.applications.pedestrian_pose_estimation.pedestrian_pose_estimation import (
    PedestrianPoseEstimation,
)
from core.applications.road_context_segmentation.road_context_segmentation import RoadContextSegmentation
from core.applications.traffic_detection.traffic_detection import TrafficDetection
from core.applications.risk_graph_generation.risk_graph_generation import RiskGraphGeneration
from core.applications.traffic_signal_recognition.traffic_signal_recognition import TrafficSignalRecognition
from core.applications.traffic_signal_recognition.traffic_signal_recognition_without_tensorrt import (
    TrafficSignalRecognitionWithoutTensorRT,
)
from core.applications.vehicle_attribute_recognition.vehicle_attribute_recognition import VehicleAttributeRecognition
from core.applications.vehicle_tracking.vehicle_tracking import (
    VehicleTracking,
)
from core.applications.vehicle_tracking.vehicle_tracking_without_tensorrt import (
    VehicleTrackingWithoutTensorRT,
)
from core.applications.vehicle_trajectory_prediction.vehicle_trajectory_prediction import VehicleTrajectoryPrediction


STRUCTURED_PROCESSOR_CLASSES = [
    TrafficDetection,
    RoadContextSegmentation,
    TrafficSignalRecognition,
    VehicleTracking,
    VehicleAttributeRecognition,
    VehicleTrajectoryPrediction,
    PedestrianPoseEstimation,
    PedestrianIntentRecognition,
    RiskGraphGeneration,
]

EXPECTED_OUTPUT_KEYS = [
    {"bbox"},
    {"segmentation"},
    {"text"},
    {"track"},
    {"attribute"},
    {"trajectory"},
    {"pose"},
    {"text"},
    {"graph"},
]

def content(service_name, outputs, frame_count=1):
    return {
        "service": service_name,
        "outputs": outputs,
        "profile": {"frame_count": frame_count},
    }


@pytest.mark.unit
def test_structured_processor_exports_are_named_explicitly():
    modules = [
        "traffic_detection",
        "road_context_segmentation",
        "traffic_signal_recognition",
        "vehicle_tracking",
        "vehicle_attribute_recognition",
        "vehicle_trajectory_prediction",
        "pedestrian_pose_estimation",
        "pedestrian_intent_recognition",
        "risk_graph_generation",
    ]

    for module_name, expected_class in zip(modules, STRUCTURED_PROCESSOR_CLASSES):
        module = importlib.import_module(f"core.applications.{module_name}")
        assert module.__all__ == ["Structured_Processor"]
        assert module.Structured_Processor is expected_class
        assert not hasattr(module, "Application")


@pytest.mark.unit
def test_structured_processors_are_independent_and_schema_free():
    assert all(cls.__bases__ == (object,) for cls in STRUCTURED_PROCESSOR_CLASSES)

    base_payload = {
        "task": {
            "file_path": "",
            "hash_data": ["frame-0"],
        },
        "frames": [np.zeros((360, 640, 3), dtype=np.uint8)],
        "inputs": {},
    }

    object_result = TrafficDetection()(base_payload)
    object_content = content("traffic-detection", object_result)
    road_result = RoadContextSegmentation()(base_payload)
    road_content = content("road-context-segmentation", road_result)
    signal_result = TrafficSignalRecognition()({
        **base_payload,
        "inputs": {"traffic-detection": object_content},
    })
    signal_content = content("traffic-signal-recognition", signal_result)
    tracking_result = VehicleTracking()({
        **base_payload,
        "inputs": {"traffic-detection": object_content},
    })
    tracking_content = content("vehicle-tracking", tracking_result)
    attribute_result = VehicleAttributeRecognition()({
        **base_payload,
        "inputs": {"traffic-detection": object_content},
    })
    attribute_content = content("vehicle-attribute-recognition", attribute_result)
    trajectory_result = VehicleTrajectoryPrediction()({
        **base_payload,
        "inputs": {
            "road-context-segmentation": road_content,
            "vehicle-tracking": tracking_content,
            "vehicle-attribute-recognition": attribute_content,
        },
    })
    trajectory_content = content("vehicle-trajectory-prediction", trajectory_result)
    pose_result = PedestrianPoseEstimation()({
        **base_payload,
        "inputs": {"traffic-detection": object_content},
    })
    pose_content = content("pedestrian-pose-estimation", pose_result)
    intent_result = PedestrianIntentRecognition()({
        **base_payload,
        "inputs": {
            "road-context-segmentation": road_content,
            "pedestrian-pose-estimation": pose_content,
        },
    })
    intent_content = content("pedestrian-intent-recognition", intent_result)
    risk_result = RiskGraphGeneration()({
        **base_payload,
        "inputs": {
            "traffic-signal-recognition": signal_content,
            "vehicle-trajectory-prediction": trajectory_content,
            "pedestrian-intent-recognition": intent_content,
        },
    })

    for result, output_keys in zip([
        object_result,
        road_result,
        signal_result,
        tracking_result,
        attribute_result,
        trajectory_result,
        pose_result,
        intent_result,
        risk_result,
    ], EXPECTED_OUTPUT_KEYS):
        assert "schema" not in result
        assert "service" not in result
        assert "profile" not in result
        assert set(result) == output_keys
        for records in result.values():
            assert isinstance(records, list)
            assert all("frame_index" in record and isinstance(record.get("items"), list) for record in records)


@pytest.mark.unit
def test_vehicle_tracking_uses_reid_embeddings_to_keep_track_identity(monkeypatch):
    processor = VehicleTrackingWithoutTensorRT(match_score_threshold=0.30, high_score_threshold=0.30)

    def fake_embedding(_payload, detection):
        object_id = detection.get("object_id", "")
        if "alpha" in object_id:
            return [1.0, 0.0, 0.0]
        if "bravo" in object_id:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    monkeypatch.setattr(processor, "_embedding_for_detection", fake_embedding)

    payload = {
        "task": {"hash_data": ["frame-0", "frame-1", "frame-2"]},
        "frames": [np.zeros((160, 260, 3), dtype=np.uint8) for _ in range(3)],
        "inputs": {
            "traffic-detection": content("traffic-detection", {
                "bbox": [
                    {"frame_index": 0, "items": [
                        {"object_id": "alpha-0", "label": "car", "category": "car",
                         "bbox": [10, 40, 50, 80], "score": 0.9},
                        {"object_id": "bravo-0", "label": "car", "category": "car",
                         "bbox": [210, 40, 250, 80], "score": 0.9},
                    ]},
                    {"frame_index": 1, "items": [
                        {"object_id": "alpha-1", "label": "car", "category": "car",
                         "bbox": [95, 40, 135, 80], "score": 0.9},
                        {"object_id": "bravo-1", "label": "car", "category": "car",
                         "bbox": [125, 40, 165, 80], "score": 0.9},
                    ]},
                    {"frame_index": 2, "items": [
                        {"object_id": "alpha-2", "label": "car", "category": "car",
                         "bbox": [180, 40, 220, 80], "score": 0.9},
                        {"object_id": "bravo-2", "label": "car", "category": "car",
                         "bbox": [50, 40, 90, 80], "score": 0.9},
                    ]},
                ]
            }, frame_count=3),
        },
    }

    result = processor(payload)
    tracks = result["track"][0]["items"]
    assert len(tracks) == 2

    by_source = {track["source_object_id"]: track for track in tracks}
    assert by_source["alpha-0"]["frames"] == [0, 1, 2]
    assert by_source["alpha-0"]["bboxes"] == [[10, 40, 50, 80], [95, 40, 135, 80], [180, 40, 220, 80]]
    assert by_source["bravo-0"]["frames"] == [0, 1, 2]
    assert by_source["bravo-0"]["bboxes"] == [[210, 40, 250, 80], [125, 40, 165, 80], [50, 40, 90, 80]]


@pytest.mark.unit
def test_structured_processors_resolve_weight_aliases(monkeypatch, tmp_path):
    weight_file = tmp_path / "service_weight.pt"
    weight_file.write_bytes(b"weight")

    def fake_get_file_path(file_path):
        return str(tmp_path / file_path)

    monkeypatch.setattr(Context, "get_file_path", staticmethod(fake_get_file_path))

    for structured_processor_cls in STRUCTURED_PROCESSOR_CLASSES:
        structured_processor = structured_processor_cls(non_trt_weights="service_weight.pt",
                                                        trt_weights="service_weight.engine",
                                                        trt_plugin_library="libplugins.so")
        assert structured_processor.non_trt_weights == str(weight_file)
        assert structured_processor.trt_weights.endswith("service_weight.engine")
        assert structured_processor.trt_plugin_library.endswith("libplugins.so")
        assert structured_processor.model.weights == str(weight_file)
        assert structured_processor.model.model["exists"] is True
        assert "loaded" in structured_processor.model.model


@pytest.mark.unit
def test_structured_processors_tensor_rt_paths_are_explicitly_unimplemented(monkeypatch, tmp_path):
    def fake_get_file_path(file_path):
        return str(tmp_path / file_path)

    def fake_get_parameter(param, default=None, direct=True):
        if param == "USE_TENSORRT":
            return True
        if param == "JETPACK":
            return 6
        return default

    monkeypatch.setattr(Context, "get_file_path", staticmethod(fake_get_file_path))
    monkeypatch.setattr(Context, "get_parameter", classmethod(lambda cls, param, default=None, direct=True: fake_get_parameter(param, default, direct)))

    for structured_processor_cls in STRUCTURED_PROCESSOR_CLASSES:
        with pytest.raises(NotImplementedError):
            structured_processor_cls(non_trt_weights="service_weight.pt",
                                     trt_weights="service_weight.engine",
                                     trt_plugin_library="libplugins.so")


@pytest.mark.unit
def test_traffic_signal_recognition_requires_upstream_traffic_light_bbox():
    processor = TrafficSignalRecognitionWithoutTensorRT()
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    result = processor({
        "frames": [frame],
        "inputs": {
            "traffic-detection": content("traffic-detection", {
                "bbox": [{"frame_index": 0, "items": [
                    {"label": "car", "category": "car", "bbox": [10, 10, 30, 30], "score": 0.9},
                    {"label": "traffic_sign", "category": "traffic_sign", "bbox": [40, 10, 60, 30], "score": 0.8},
                ]}]
            })
        },
    })

    assert result == {"text": []}


@pytest.mark.unit
def test_traffic_signal_recognition_runs_model_only_on_traffic_light_crops():
    class Boxes:
        xyxy = [[1, 1, 6, 8]]
        conf = [0.93]
        cls = [1]

    class Result:
        names = {0: "red", 1: "green"}
        boxes = Boxes()

    class Detector:
        def __init__(self):
            self.sources = []

        def predict(self, source, **kwargs):
            self.sources.append(source)
            return [Result()]

    detector = Detector()
    processor = TrafficSignalRecognitionWithoutTensorRT()
    processor.model = {"loaded": True, "model": detector}
    frame = np.zeros((100, 120, 3), dtype=np.uint8)

    result = processor({
        "frames": [frame],
        "inputs": {
            "traffic-detection": content("traffic-detection", {
                "bbox": [{"frame_index": 0, "items": [
                    {"object_id": "car-0", "label": "car", "category": "car", "bbox": [0, 0, 20, 20], "score": 0.9},
                    {"object_id": "light-0", "label": "traffic_light", "category": "traffic_light",
                     "bbox": [10, 20, 30, 60], "score": 0.88},
                    {"object_id": "sign-0", "label": "traffic_sign", "category": "traffic_sign",
                     "bbox": [40, 20, 60, 60], "score": 0.77},
                ]}]
            })
        },
    })

    assert len(detector.sources) == 1
    assert detector.sources[0].shape == (40, 20, 3)
    assert result == {"text": [{
        "frame_index": 0,
        "items": [{
            "frame_id": 0,
            "signal_id": "traffic-signal-0-0",
            "source_object_id": "light-0",
            "bbox": [10, 20, 30, 60],
            "label": "traffic_light",
            "type": "traffic_light",
            "text": "green",
            "state": "green",
            "score": 0.93,
            "source_score": 0.88,
            "model_label": "green",
            "frame_index": 0,
        }],
    }]}
