import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from core.lib.content import Task


image_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.image_visualizer")
multiple_roi_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.multiple_roi_frame_visualizer")
roi_frame_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_frame_visualizer")
roi_label_visualizer_module = importlib.import_module("core.lib.algorithms.result_visualizer.roi_label_frame_visualizer")


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
    task.get_service("detector").set_content_data([([[1, 1, 5, 5]], [0.9])])
    task.get_service("classifier").set_content_data([(["car"], [0.9])])
    return task


@pytest.mark.unit
def test_image_visualizer_validates_coordinate_values_and_first_frame_reads(monkeypatch):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="convertible to integers"):
        image_visualizer_module.ImageVisualizer.draw_bboxes(frame.copy(), [["a", 1, 2, 3]])
    with pytest.raises(ValueError, match="out of frame bounds"):
        image_visualizer_module.ImageVisualizer.draw_bboxes(frame.copy(), [[0, 0, 9, 9]])
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
