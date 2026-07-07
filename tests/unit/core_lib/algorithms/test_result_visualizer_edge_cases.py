import ast
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from core.lib.common import ClassFactory, ClassType, YamlOps
from core.lib.content import Task


bbox_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.bbox_frame_visualizer")
event_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.event_frame_visualizer")
image_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.image_visualizer")
multiple_roi_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.multiple_roi_frame_visualizer")
pose_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.pose_frame_visualizer")
roi_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_frame_visualizer")
roi_label_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_label_frame_visualizer")
segmentation_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.segmentation_frame_visualizer")
text_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.text_frame_visualizer")
track_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.track_frame_visualizer")
trajectory_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.trajectory_frame_visualizer")


def content_profile(frame_count=1):
    return {
        "frame_count": frame_count,
    }


def service_entry(name, *, execute_device="", next_nodes=None, prev_nodes=None):
    return {
        "service": {
            "service_name": name,
            "execute_device": execute_device,
        },
        "next_nodes": next_nodes or [],
        "prev_nodes": prev_nodes or [],
    }


def build_visualization_task():
    dag = Task.extract_dag_from_dict(
        {
            "detector": service_entry("detector", execute_device="edge-a", next_nodes=["classifier"]),
            "classifier": service_entry("classifier", execute_device="cloud-a"),
        }
    )
    task = Task(
        source_id=1,
        task_id=9,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        metadata={"buffer_size": 2, "resolution": "720p"},
        raw_metadata={"buffer_size": 2, "resolution": "1080p"},
        file_path="sample.mp4",
    )
    task.get_service("detector").set_content_data({
        "service": "detector",
        "outputs": {"bbox": [{"frame_index": 0, "items": [{"bbox": [1, 1, 5, 5], "score": 0.9, "label": "car", "object_id": 1}]}]},
        "profile": content_profile(),
    })
    task.get_service("classifier").set_content_data({
        "service": "classifier",
        "outputs": {"text": [{"frame_index": 0, "items": [{"text": "car", "source_object_id": 1, "bbox": [1, 1, 5, 5]}]}]},
        "profile": content_profile(),
    })
    return task


def build_service_overlay_task():
    service_names = [
        "traffic-object-detection",
        "road-context-segmentation",
        "traffic-signal-recognition",
        "vehicle-reidentification-tracking",
        "vehicle-attribute-recognition",
        "vehicle-trajectory-prediction",
        "pedestrian-cyclist-pose-estimation",
        "pedestrian-cyclist-intent-recognition",
        "traffic-risk-graph-inference",
    ]
    dag_dict = {}
    for index, service_name in enumerate(service_names):
        next_nodes = [service_names[index + 1]] if index + 1 < len(service_names) else []
        dag_dict[service_name] = service_entry(service_name, execute_device="edge-a", next_nodes=next_nodes)
    dag = Task.extract_dag_from_dict(dag_dict)
    task = Task(
        source_id=1,
        task_id=99,
        source_device="edge-a",
        all_edge_devices=["edge-a", "edge-b"],
        dag=dag,
        file_path="sample.mp4",
    )

    task.get_service("traffic-object-detection").set_content_data({
        "service": "traffic-object-detection",
        "outputs": {"bbox": [{"frame_index": 0, "items": [
            {"bbox": [10, 20, 42, 60], "label": "car", "score": 0.91, "object_id": "car-1"},
            {"bbox": [70, 18, 78, 68], "label": "pedestrian", "score": 0.88, "object_id": "person-1"},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("road-context-segmentation").set_content_data({
        "service": "road-context-segmentation",
        "outputs": {"segmentation": [{"frame_index": 0, "items": [
            {"type": "lane_polyline", "points": [[20, 78], [46, 44], [56, 24]]},
            {"type": "drivable_area", "polygon": [[0, 79], [36, 34], [84, 34], [119, 79]]},
            {"type": "crosswalk_region", "polygon": [[20, 52], [92, 52], [100, 62], [16, 62]]},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("traffic-signal-recognition").set_content_data({
        "service": "traffic-signal-recognition",
        "outputs": {"text": [{"frame_index": 0, "items": [
            {"bbox": [52, 8, 62, 24], "state": "red", "text": "red", "score": 0.83},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("vehicle-reidentification-tracking").set_content_data({
        "service": "vehicle-reidentification-tracking",
        "outputs": {"track": [{"frame_index": None, "items": [
            {"track_id": "vehicle-1", "bboxes": [[10, 22, 40, 60], [16, 22, 46, 60]], "frames": [0, 1],
             "direction": "eastbound", "speed_px_per_s": 6.2, "source_object_id": "car-1"},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("vehicle-attribute-recognition").set_content_data({
        "service": "vehicle-attribute-recognition",
        "outputs": {"attribute": [{"frame_index": None, "items": [
            {"bbox": [16, 22, 46, 60], "source_object_id": "car-1", "confidence": 0.87,
             "attributes": {"type": "car", "color": "blue", "orientation": "side"}},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("vehicle-trajectory-prediction").set_content_data({
        "service": "vehicle-trajectory-prediction",
        "outputs": {"trajectory": [{"frame_index": None, "items": [
            {"track_id": "vehicle-1", "vehicle_type": "car", "abnormal_stop_prob": 0.08,
             "future_trajectories": [{"prob": 0.72, "points": [[50, 42, 0.5], [58, 42, 1.0], [66, 42, 1.5]]}]},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("pedestrian-cyclist-pose-estimation").set_content_data({
        "service": "pedestrian-cyclist-pose-estimation",
        "outputs": {"pose": [{"frame_index": 0, "items": [
            {"person_id": "person-1", "bbox": [70, 18, 78, 68], "orientation": "toward-road",
             "keypoints": [[74, 24, 0.9], [72, 36, 0.8], [76, 36, 0.8], [72, 62, 0.8], [76, 62, 0.8]]},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("pedestrian-cyclist-intent-recognition").set_content_data({
        "service": "pedestrian-cyclist-intent-recognition",
        "outputs": {"text": [{"frame_index": None, "items": [
            {"person_id": "person-1", "intent": "likely_to_cross", "text": "likely_to_cross", "confidence": 0.82},
        ]}]},
        "profile": content_profile(),
    })
    task.get_service("traffic-risk-graph-inference").set_content_data({
        "service": "traffic-risk-graph-inference",
        "outputs": {"graph": [{"frame_index": None, "items": [
            {"nodes": [{"id": "vehicle-1"}, {"id": "person-1"}],
             "edges": [{"source": "vehicle-1", "target": "person-1"}],
             "events": [{"type": "near_miss", "risk_score": 0.88}],
             "summary": {"entity_count": 2, "relation_count": 1, "signal_count": 1},
             "risk_level": "high", "risk_confidence": 0.88},
        ]}]},
        "profile": content_profile(),
    })
    return task


@pytest.mark.unit
def test_image_visualizer_validates_coordinate_values_and_first_frame_reads(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(NotImplementedError):
        image_visualizer_module.ImageVisualizer(variables=["image"])(build_visualization_task())
    with pytest.raises(ValueError, match="convertible to integers"):
        image_visualizer_module.ImageVisualizer.draw_bboxes(frame.copy(), [["a", 1, 2, 3]])
    with pytest.raises(ValueError, match="out of frame bounds"):
        image_visualizer_module.ImageVisualizer.draw_bboxes(frame.copy(), [[0, 0, 9, 9]])
    with pytest.raises(ValueError, match="numpy array"):
        image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels("not-a-frame", [[0, 0, 1, 1]], ["car"])
    with pytest.raises(ValueError, match="4-element tuples/lists"):
        image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels(frame.copy(), "bad-boxes", ["car"])
    with pytest.raises(ValueError, match="must be numeric"):
        image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels(frame.copy(), [["a", 1, 2, 3]], ["car"])
    with pytest.raises(ValueError, match="Invalid coordinates"):
        image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels(frame.copy(), [[0, 0, 9, 9]], ["car"])

    import cv2

    class DummyCap:
        def __init__(self, opened, success):
            self.opened = opened
            self.success = success

        def isOpened(self):
            return self.opened

        def read(self):
            return self.success, None

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: DummyCap(True, False))
    with pytest.raises(ValueError, match="Failed to read the first frame"):
        image_visualizer_module.ImageVisualizer.get_first_frame_from_video("demo.mp4")

    text_positions = []
    monkeypatch.setattr(cv2, "rectangle", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "getTextSize", lambda text, font, scale, thickness: ((3, 2), 0))
    monkeypatch.setattr(
        cv2,
        "putText",
        lambda image, text, org, font, scale, color, thickness, line_type: text_positions.append(org),
    )
    result = image_visualizer_module.ImageVisualizer.draw_bboxes_and_labels(
        np.zeros((12, 12, 3), dtype=np.uint8),
        [[1, 10, 5, 11]],
        ["car"],
    )
    assert result.shape == (12, 12, 3)
    assert text_positions == [(1, 5)]


@pytest.mark.unit
def test_roi_frame_visualizer_falls_back_to_first_content_when_named_service_is_missing(monkeypatch):
    task = build_visualization_task()
    drawn_boxes = []

    monkeypatch.setattr(roi_frame_visualizer_module.EncodeOps, "encode_image", staticmethod(lambda image: "roi-encoded"))
    monkeypatch.setattr(
        roi_frame_visualizer_module.ROIFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        roi_frame_visualizer_module.ROIFrameVisualizer,
        "draw_bboxes",
        staticmethod(lambda image, boxes: drawn_boxes.append(list(boxes)) or image),
    )

    visualizer = roi_frame_visualizer_module.ROIFrameVisualizer(variables=["roi"], roi_service="missing-service")
    assert visualizer(task) == {"roi": "roi-encoded"}
    assert drawn_boxes == [[[1, 1, 5, 5]]]


@pytest.mark.unit
def test_roi_visualizers_use_default_image_when_rendering_raises(monkeypatch):
    task = build_visualization_task()
    encoded = []
    warnings = []
    exceptions = []

    def fake_encode(image):
        encoded.append(int(image.sum()))
        return f"encoded-{len(encoded)}"

    monkeypatch.setattr(roi_frame_visualizer_module.EncodeOps, "encode_image", staticmethod(fake_encode))
    monkeypatch.setattr(roi_label_visualizer_module.EncodeOps, "encode_image", staticmethod(fake_encode))
    monkeypatch.setattr(
        roi_frame_visualizer_module.ROIFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        roi_frame_visualizer_module.ROIFrameVisualizer,
        "draw_bboxes",
        staticmethod(lambda image, boxes: (_ for _ in ()).throw(ValueError("bad boxes"))),
    )
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "draw_bboxes_and_labels",
        staticmethod(lambda image, boxes, labels: (_ for _ in ()).throw(ValueError("bad labels"))),
    )
    monkeypatch.setattr(roi_frame_visualizer_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(roi_frame_visualizer_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))
    monkeypatch.setattr(roi_label_visualizer_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(roi_label_visualizer_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))

    import cv2

    monkeypatch.setattr(cv2, "imread", lambda path: np.ones((4, 4, 3), dtype=np.uint8))

    roi_visualizer = roi_frame_visualizer_module.ROIFrameVisualizer(variables=["roi"])
    label_visualizer = roi_label_visualizer_module.ROILabelFrameVisualizer(
        variables=["labeled"],
        roi_service="missing-roi",
        label_service="missing-label",
    )

    assert roi_visualizer(task) == {"roi": "encoded-1"}
    assert label_visualizer(task) == {"labeled": "encoded-2"}
    assert len(warnings) == 2
    assert "bad boxes" in exceptions[0]
    assert "bad labels" in exceptions[1]


@pytest.mark.unit
def test_roi_label_and_multiple_roi_visualizers_fall_back_to_default_task_content(monkeypatch):
    task = build_visualization_task()
    draw_calls = []
    encoded = []

    def fake_encode(image):
        encoded.append(True)
        return "label-encoded" if len(encoded) == 1 else "multi-encoded"

    monkeypatch.setattr(roi_label_visualizer_module.EncodeOps, "encode_image", staticmethod(fake_encode))
    monkeypatch.setattr(multiple_roi_visualizer_module.EncodeOps, "encode_image", staticmethod(fake_encode))
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "draw_bboxes_and_labels",
        staticmethod(lambda image, boxes, labels: draw_calls.append(("label", list(boxes), list(labels))) or image),
    )
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "draw_bboxes",
        staticmethod(lambda image, boxes: draw_calls.append(("multi", list(boxes))) or image),
    )

    label_visualizer = roi_label_visualizer_module.ROILabelFrameVisualizer(
        variables=["labeled"],
        roi_service="missing-roi",
        label_service="missing-label",
    )
    multiple_visualizer = multiple_roi_visualizer_module.ROIFrameVisualizer(
        variables=["multi"],
        roi_services=["missing-roi"],
    )

    assert label_visualizer(task) == {"labeled": "label-encoded"}
    assert multiple_visualizer(task) == {"multi": "multi-encoded"}
    assert draw_calls == [
        ("label", [[1, 1, 5, 5]], ["car"]),
        ("multi", [[1, 1, 5, 5]]),
    ]


@pytest.mark.unit
def test_roi_visualizers_use_default_content_when_services_are_not_configured(monkeypatch):
    task = build_visualization_task()
    draw_calls = []
    encode_calls = []

    def fake_encode(_image):
        encode_calls.append(True)
        return "roi-label" if len(encode_calls) == 1 else "multi"

    monkeypatch.setattr(roi_label_visualizer_module.EncodeOps, "encode_image", staticmethod(fake_encode))
    monkeypatch.setattr(multiple_roi_visualizer_module.EncodeOps, "encode_image", staticmethod(fake_encode))
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        roi_label_visualizer_module.ROILabelFrameVisualizer,
        "draw_bboxes_and_labels",
        staticmethod(lambda image, boxes, labels: draw_calls.append(("label", list(boxes), list(labels))) or image),
    )
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "draw_bboxes",
        staticmethod(lambda image, boxes: draw_calls.append(("multi", list(boxes))) or image),
    )

    label_visualizer = roi_label_visualizer_module.ROILabelFrameVisualizer(variables=["labeled"])
    multiple_visualizer = multiple_roi_visualizer_module.ROIFrameVisualizer(variables=["multi"])

    assert label_visualizer(task) == {"labeled": "roi-label"}
    assert multiple_visualizer(task) == {"multi": "multi"}
    assert draw_calls == [
        ("label", [[1, 1, 5, 5]], ["car"]),
        ("multi", [[1, 1, 5, 5]]),
    ]


@pytest.mark.unit
def test_multiple_roi_visualizer_uses_default_image_when_rendering_fails(monkeypatch):
    task = build_visualization_task()
    warnings = []
    exceptions = []

    monkeypatch.setattr(multiple_roi_visualizer_module.EncodeOps, "encode_image", staticmethod(lambda image: "fallback"))
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "get_first_frame_from_video",
        staticmethod(lambda path: np.zeros((8, 8, 3), dtype=np.uint8)),
    )
    monkeypatch.setattr(
        multiple_roi_visualizer_module.ROIFrameVisualizer,
        "draw_bboxes",
        staticmethod(lambda image, boxes: (_ for _ in ()).throw(ValueError("multi render failed"))),
    )
    monkeypatch.setattr(multiple_roi_visualizer_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(multiple_roi_visualizer_module.LOGGER, "exception", lambda exc: exceptions.append(str(exc)))

    import cv2

    monkeypatch.setattr(cv2, "imread", lambda path: np.ones((4, 4, 3), dtype=np.uint8))

    visualizer = multiple_roi_visualizer_module.ROIFrameVisualizer(variables=["multi"])
    assert visualizer(task) == {"multi": "fallback"}
    assert len(warnings) == 1
    assert exceptions == ["multi render failed"]


@pytest.mark.unit
def test_driving_perception_visualization_config_uses_registered_hooks():
    config = YamlOps.read_yaml("config/visualization_configs/driving_perception_visualization_config.yaml")
    expected_services = {
        "traffic-object-detection",
        "road-context-segmentation",
        "traffic-signal-recognition",
        "vehicle-reidentification-tracking",
        "vehicle-attribute-recognition",
        "vehicle-trajectory-prediction",
        "pedestrian-cyclist-pose-estimation",
        "pedestrian-cyclist-intent-recognition",
        "traffic-risk-graph-inference",
    }
    image_services = set()

    for visualization in config:
        assert isinstance(visualization["variables"], list)
        assert ClassFactory.is_exists(ClassType.RESULT_VISUALIZER, visualization["hook_name"])
        hook_params = visualization.get("hook_params")
        if hook_params:
            hook_params = ast.literal_eval(hook_params)
            assert isinstance(hook_params, dict)
            service = hook_params.get("service")
            if visualization["type"] == "image" and service in expected_services:
                image_services.add(service)

    assert image_services == expected_services


@pytest.mark.unit
def test_service_frame_visualizers_render_structured_outputs(monkeypatch):
    task = build_service_overlay_task()
    encoded_sums = []

    def fake_encode(image):
        encoded_sums.append(int(image.sum()))
        return f"encoded-{len(encoded_sums)}"

    for module in (
        bbox_frame_visualizer_module,
        event_frame_visualizer_module,
        pose_frame_visualizer_module,
        segmentation_frame_visualizer_module,
        text_frame_visualizer_module,
        track_frame_visualizer_module,
        trajectory_frame_visualizer_module,
    ):
        monkeypatch.setattr(module.EncodeOps, "encode_image", staticmethod(fake_encode))

    monkeypatch.setattr(
        image_visualizer_module.ImageVisualizer,
        "get_frame_from_video",
        staticmethod(lambda path, frame_index=0: np.zeros((80, 120, 3), dtype=np.uint8)),
    )

    visualizers = [
        bbox_frame_visualizer_module.BBoxFrameVisualizer(
            variables=["image"],
            service="traffic-object-detection",
            output="bbox",
            label_fields=["label", "score"],
        ),
        segmentation_frame_visualizer_module.SegmentationFrameVisualizer(
            variables=["image"],
            service="road-context-segmentation",
            output="segmentation",
        ),
        bbox_frame_visualizer_module.BBoxFrameVisualizer(
            variables=["image"],
            service="traffic-signal-recognition",
            output="text",
            label_fields=["state", "score"],
        ),
        track_frame_visualizer_module.TrackFrameVisualizer(
            variables=["image"],
            service="vehicle-reidentification-tracking",
            output="track",
        ),
        bbox_frame_visualizer_module.BBoxFrameVisualizer(
            variables=["image"],
            service="vehicle-attribute-recognition",
            output="attribute",
            label_template="{attributes.type}/{attributes.color}/{attributes.orientation} {confidence}",
            label_fields=["attributes.type", "attributes.color", "attributes.orientation", "confidence"],
        ),
        trajectory_frame_visualizer_module.TrajectoryFrameVisualizer(
            variables=["image"],
            service="vehicle-trajectory-prediction",
            output="trajectory",
            track_service="vehicle-reidentification-tracking",
        ),
        pose_frame_visualizer_module.PoseFrameVisualizer(
            variables=["image"],
            service="pedestrian-cyclist-pose-estimation",
            output="pose",
        ),
        text_frame_visualizer_module.TextFrameVisualizer(
            variables=["image"],
            service="pedestrian-cyclist-intent-recognition",
            output="text",
            label_fields=["intent", "confidence"],
            anchor_service="pedestrian-cyclist-pose-estimation",
            anchor_output="pose",
            anchor_key="person_id",
        ),
        event_frame_visualizer_module.EventFrameVisualizer(
            variables=["image"],
            service="traffic-risk-graph-inference",
            output="graph",
        ),
        bbox_frame_visualizer_module.BBoxFrameVisualizer(
            variables=["image"],
            service="traffic-object-detection",
            output="missing",
        ),
    ]

    for index, visualizer in enumerate(visualizers, start=1):
        assert visualizer(task) == {"image": f"encoded-{index}"}

    assert len(encoded_sums) == len(visualizers)
    assert all(encoded_sum > 0 for encoded_sum in encoded_sums)
