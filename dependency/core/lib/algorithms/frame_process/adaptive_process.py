import abc
import cv2
import numpy as np
from typing import List, Tuple

from core.lib.common import ClassFactory, ClassType, LOGGER
from core.lib.common import VideoOps
from .base_process import BaseProcess

__all__ = ('AdaptiveProcess',)


@ClassFactory.register(ClassType.GEN_PROCESS, alias='adaptive')
class AdaptiveProcess(BaseProcess, abc.ABC):
    def __init__(self):
        super().__init__()
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    def __call__(self, system, frame, source_resolution, target_resolution):
        fallback_frame = frame
        try:
            resolution = VideoOps.text2resolution(target_resolution)
            frame_resize = cv2.resize(frame, resolution)
            fallback_frame = frame_resize
            frame_height, frame_width, _ = frame_resize.shape

            # ROI coordinates must be measured in the same coordinate system
            # as the frame later written by AdaptiveCompress.
            fg_mask = self.extract_foreground_mask(frame_resize)

            fg_mask = self.apply_morphological_operations(fg_mask)

            contours = self.find_contours(fg_mask)
            filtered_contours = self.filter_contours_by_area(contours, min_area=1000, max_area=50000)

            contour_scores = self.calculate_contour_scores(filtered_contours, frame_resize)
            valid_rois = self.get_valid_rois(contour_scores, frame_width, frame_height)

            return (frame_resize, valid_rois)
        except Exception as e:
            LOGGER.warning(f"Adaptive frame processing failed: {e}")
            # Keep every buffered frame at one resolution when ROI extraction
            # fails after resize; AdaptiveCompress writes one fixed-size YUV
            # sequence for the whole buffer.
            return (fallback_frame, [])

    def extract_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        return self.backSub.apply(frame)

    def apply_morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def filter_contours_by_area(self, contours: List[np.ndarray], min_area: int, max_area: int) -> List[np.ndarray]:
        """
        filter by area
        """
        return [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    def calculate_contour_scores(self, contours: List[np.ndarray], frame: np.ndarray) -> List[Tuple[float, Tuple[int, int, int, int]]]:
        return [(self.calculate_score(cnt), self.get_bounding_box(cnt)) for cnt in contours]

    def calculate_score(self, contour: np.ndarray) -> float:
        return cv2.contourArea(contour)

    def get_bounding_box(self, contour: np.ndarray) -> Tuple[int, int, int, int]:
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, x + w, y + h)

    def get_valid_rois(self, contour_scores: List[Tuple[float, Tuple[int, int, int, int]]], frame_width: int, frame_height: int) -> List[Tuple[int, int, int, int]]:
        return [
            bbox for score, bbox in contour_scores
            if self.is_roi_valid(bbox[0], bbox[1], bbox[2], bbox[3], frame_width, frame_height)
        ]

    def is_roi_valid(self, x1: int, y1: int, x2: int, y2: int, frame_width: int, frame_height: int) -> bool:
        if frame_width <= 0 or frame_height <= 0:
            return False
        return (x2 - x1) / frame_width <= 0.3 and (y2 - y1) / frame_height <= 0.3

    def generate_roi_message(self, rois: List[Tuple[int, int, int, int]]) -> str:
        num_regions = min(len(rois), 8)
        message = f"{num_regions} "
        for (x1, y1, x2, y2) in rois[:num_regions]:
            message += f"-10 {x1} {y1} {x2 - x1} {y2 - y1} "
        return message
