import numpy as np
import pytest

from core.lib.common import Context
from core.applications.pedestrian_cyclist_intent_recognition.pedestrian_cyclist_intent_recognition import (
    PedestrianCyclistIntentRecognition,
)
from core.applications.pedestrian_cyclist_pose_estimation.pedestrian_cyclist_pose_estimation import (
    PedestrianCyclistPoseEstimation,
)
from core.applications.road_context_segmentation.road_context_segmentation import RoadContextSegmentation
from core.applications.traffic_object_detection.traffic_object_detection import TrafficObjectDetection
from core.applications.traffic_risk_graph_inference.traffic_risk_graph_inference import TrafficRiskGraphInference
from core.applications.traffic_signal_recognition.traffic_signal_recognition import TrafficSignalRecognition
from core.applications.vehicle_attribute_recognition.vehicle_attribute_recognition import VehicleAttributeRecognition
from core.applications.vehicle_reidentification_tracking.vehicle_reidentification_tracking import (
    VehicleReidentificationTracking,
)
from core.applications.vehicle_trajectory_prediction.vehicle_trajectory_prediction import VehicleTrajectoryPrediction


APPLICATION_CLASSES = [
    TrafficObjectDetection,
    RoadContextSegmentation,
    TrafficSignalRecognition,
    VehicleReidentificationTracking,
    VehicleAttributeRecognition,
    VehicleTrajectoryPrediction,
    PedestrianCyclistPoseEstimation,
    PedestrianCyclistIntentRecognition,
    TrafficRiskGraphInference,
]


@pytest.mark.unit
def test_structured_applications_are_independent_and_schema_free():
    assert all(cls.__bases__ == (object,) for cls in APPLICATION_CLASSES)

    base_payload = {
        "task": {
            "file_path": "",
            "hash_data": ["frame-0"],
        },
        "frames": [np.zeros((360, 640, 3), dtype=np.uint8)],
        "inputs": {},
    }

    object_result = TrafficObjectDetection()(base_payload)
    road_result = RoadContextSegmentation()(base_payload)
    signal_result = TrafficSignalRecognition()({
        **base_payload,
        "inputs": {"traffic-object-detection": object_result},
    })
    tracking_result = VehicleReidentificationTracking()({
        **base_payload,
        "inputs": {"traffic-object-detection": object_result},
    })
    attribute_result = VehicleAttributeRecognition()({
        **base_payload,
        "inputs": {"traffic-object-detection": object_result},
    })
    trajectory_result = VehicleTrajectoryPrediction()({
        **base_payload,
        "inputs": {
            "road-context-segmentation": road_result,
            "vehicle-reidentification-tracking": tracking_result,
            "vehicle-attribute-recognition": attribute_result,
        },
    })
    pose_result = PedestrianCyclistPoseEstimation()({
        **base_payload,
        "inputs": {"traffic-object-detection": object_result},
    })
    intent_result = PedestrianCyclistIntentRecognition()({
        **base_payload,
        "inputs": {
            "road-context-segmentation": road_result,
            "pedestrian-cyclist-pose-estimation": pose_result,
        },
    })
    risk_result = TrafficRiskGraphInference()({
        **base_payload,
        "inputs": {
            "traffic-signal-recognition": signal_result,
            "vehicle-trajectory-prediction": trajectory_result,
            "pedestrian-cyclist-intent-recognition": intent_result,
        },
    })

    for result in [
        object_result,
        road_result,
        signal_result,
        tracking_result,
        attribute_result,
        trajectory_result,
        pose_result,
        intent_result,
        risk_result,
    ]:
        assert "schema" not in result
        assert result["service"]
        assert isinstance(result["outputs"], dict)
        assert result["profile"]["model_name"]
        assert "model_variant" not in result["profile"]
        assert "synthetic_complexity" not in result["profile"]
        assert "model_loaded" in result["profile"]
        assert "inference_backend" in result["profile"]
        assert "model_error" in result["profile"]


@pytest.mark.unit
def test_structured_applications_resolve_weight_aliases(monkeypatch, tmp_path):
    weight_file = tmp_path / "service_weight.pt"
    weight_file.write_bytes(b"weight")

    def fake_get_file_path(file_path):
        return str(tmp_path / file_path)

    monkeypatch.setattr(Context, "get_file_path", staticmethod(fake_get_file_path))

    for application_cls in APPLICATION_CLASSES:
        application = application_cls(non_trt_weights="service_weight.pt",
                                      trt_weights="service_weight.engine",
                                      trt_plugin_library="libplugins.so")
        assert application.non_trt_weights == str(weight_file)
        assert application.trt_weights.endswith("service_weight.engine")
        assert application.trt_plugin_library.endswith("libplugins.so")
        assert application.model.weights == str(weight_file)
        assert application.model.model["exists"] is True
        assert "loaded" in application.model.model


@pytest.mark.unit
def test_structured_applications_tensor_rt_paths_are_explicitly_unimplemented(monkeypatch, tmp_path):
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

    for application_cls in APPLICATION_CLASSES:
        with pytest.raises(NotImplementedError):
            application_cls(non_trt_weights="service_weight.pt",
                            trt_weights="service_weight.engine",
                            trt_plugin_library="libplugins.so")
