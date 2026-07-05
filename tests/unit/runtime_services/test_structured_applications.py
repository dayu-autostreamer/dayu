import numpy as np
import pytest

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


@pytest.mark.unit
def test_structured_applications_are_independent_and_schema_free():
    application_classes = [
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
    assert all(cls.__bases__ == (object,) for cls in application_classes)

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
