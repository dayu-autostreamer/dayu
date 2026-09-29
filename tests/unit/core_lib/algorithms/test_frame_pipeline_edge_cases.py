import importlib
from types import SimpleNamespace

import numpy as np
import pytest


adaptive_process_module = importlib.import_module("core.lib.algorithms.frame_process.adaptive_process")
motion_filter_module = importlib.import_module("core.lib.algorithms.frame_filter.motion_filter")


@pytest.mark.unit
def test_motion_filter_computes_motion_ratio_and_skip_decisions(monkeypatch):
    dummy_bg = SimpleNamespace(apply=lambda frame: np.ones((4, 4), dtype=np.uint8))
    monkeypatch.setattr(motion_filter_module.cv2, "createBackgroundSubtractorMOG2", lambda **kwargs: dummy_bg)
    monkeypatch.setattr(motion_filter_module.cv2, "morphologyEx", lambda mask, op, kernel: mask)

    motion_filter = motion_filter_module.MotionFilter(min_fps=2, max_fps=8, smoothing_factor=0.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert motion_filter._calculate_motion(frame) == 1.0

    system = SimpleNamespace(raw_meta_data={"fps": 20}, meta_data={"fps": 15})
    monkeypatch.setattr(motion_filter, "_calculate_motion", lambda current_frame: 1.0)

    decisions = [motion_filter(system, frame) for _ in range(4)]
    assert decisions == [True, True, True, False]
    assert motion_filter_module.MotionFilter.get_fps_adjust_mode(20, 15) == ("skip", 4, 0)


@pytest.mark.unit
def test_motion_filter_uses_remain_mode_and_interpolated_target_fps(monkeypatch):
    dummy_bg = SimpleNamespace(apply=lambda frame: np.zeros((4, 4), dtype=np.uint8))
    monkeypatch.setattr(motion_filter_module.cv2, "createBackgroundSubtractorMOG2", lambda **kwargs: dummy_bg)
    monkeypatch.setattr(motion_filter_module.cv2, "morphologyEx", lambda mask, op, kernel: mask)

    motion_filter = motion_filter_module.MotionFilter(
        min_fps=2,
        max_fps=8,
        motion_threshold_min=0.1,
        motion_threshold_max=0.5,
        smoothing_factor=0.0,
    )
    assert motion_filter._calculate_target_fps(0.3) == pytest.approx(5.0)
    assert motion_filter_module.MotionFilter.get_fps_adjust_mode(20, 2) == ("remain", 0, 10)
    assert motion_filter_module.MotionFilter.get_fps_adjust_mode(20, 20) == ("same", 0, 0)

    system = SimpleNamespace(raw_meta_data={"fps": 20}, meta_data={"fps": 20})
    monkeypatch.setattr(motion_filter, "_calculate_motion", lambda current_frame: 0.0)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    decisions = [motion_filter(system, frame) for _ in range(10)]
    assert decisions == [False, False, False, False, False, False, False, False, False, True]


@pytest.mark.unit
def test_adaptive_process_helpers_cover_contour_boundaries_and_roi_message_limits(monkeypatch):
    dummy_bg = SimpleNamespace(apply=lambda frame: np.ones((8, 8), dtype=np.uint8))
    monkeypatch.setattr(adaptive_process_module.cv2, "createBackgroundSubtractorMOG2", lambda **kwargs: dummy_bg)

    adaptive = adaptive_process_module.AdaptiveProcess()

    morph_calls = []
    monkeypatch.setattr(
        adaptive_process_module.cv2,
        "morphologyEx",
        lambda mask, op, kernel: morph_calls.append(op) or mask,
    )
    adaptive.apply_morphological_operations(np.ones((4, 4), dtype=np.uint8))
    assert morph_calls == [adaptive_process_module.cv2.MORPH_OPEN, adaptive_process_module.cv2.MORPH_CLOSE]

    monkeypatch.setattr(
        adaptive_process_module.cv2,
        "findContours",
        lambda mask, mode, method: ([900, 1500, 60000], None),
    )
    monkeypatch.setattr(adaptive_process_module.cv2, "contourArea", lambda contour: contour)
    assert adaptive.find_contours(np.ones((4, 4), dtype=np.uint8)) == [900, 1500, 60000]
    assert adaptive.filter_contours_by_area([900, 1500, 60000], min_area=1000, max_area=50000) == [1500]

    contour_scores = [
        (10.0, (0, 0, 20, 20)),
        (9.0, (0, 0, 80, 80)),
    ]
    assert adaptive.get_valid_rois(contour_scores, frame_width=100, frame_height=100) == [(0, 0, 20, 20)]
    assert adaptive.is_roi_valid(0, 0, 20, 20, 100, 100) is True
    assert adaptive.is_roi_valid(0, 0, 80, 80, 100, 100) is False
    assert adaptive.is_roi_valid(0, 0, 20, 20, 0, 100) is False

    roi_message = adaptive.generate_roi_message([(i, i, i + 1, i + 1) for i in range(10)])
    assert roi_message.startswith("8 ")
    assert roi_message.count("-10") == 8


@pytest.mark.unit
def test_adaptive_process_returns_original_frame_on_errors(monkeypatch):
    dummy_bg = SimpleNamespace(apply=lambda frame: np.ones((8, 8), dtype=np.uint8))
    monkeypatch.setattr(adaptive_process_module.cv2, "createBackgroundSubtractorMOG2", lambda **kwargs: dummy_bg)

    adaptive = adaptive_process_module.AdaptiveProcess()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(
        adaptive_process_module.VideoOps,
        "text2resolution",
        staticmethod(lambda text: (_ for _ in ()).throw(RuntimeError("bad resolution"))),
    )
    warnings = []
    monkeypatch.setattr(
        adaptive_process_module.LOGGER,
        "warning",
        lambda message: warnings.append(message),
    )
    processed, rois = adaptive(
        SimpleNamespace(meta_data={"resolution": "720p", "buffer_size": 1}),
        frame,
        "720p",
        "480p",
    )
    assert processed is frame
    assert rois == []
    assert warnings == ["Adaptive frame processing failed: bad resolution"]

    resized = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        adaptive_process_module.VideoOps,
        "text2resolution",
        staticmethod(lambda text: (4, 4)),
    )
    monkeypatch.setattr(
        adaptive_process_module.cv2,
        "resize",
        lambda image, resolution: resized,
    )
    monkeypatch.setattr(
        adaptive,
        "extract_foreground_mask",
        lambda image: (_ for _ in ()).throw(RuntimeError("bad roi")),
    )
    processed, rois = adaptive(
        SimpleNamespace(meta_data={"resolution": "720p", "buffer_size": 1}),
        frame,
        "720p",
        "480p",
    )
    assert processed is resized
    assert rois == []
    assert warnings[-1] == "Adaptive frame processing failed: bad roi"
