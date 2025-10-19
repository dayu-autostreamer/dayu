"""
Car tacking with optical flow
"""

from typing import List

import cv2
import numpy as np


class CarTracking:
    def __init__(self):
        pass

    def __call__(self, tracking_frame_list: List[np.ndarray],
                 prev_detection_frame: np.ndarray,
                 content_result: tuple):
        bbox, prob, class_id, roi_id = content_result
        grey_prev_frame = cv2.cvtColor(prev_detection_frame, cv2.COLOR_BGR2GRAY)
        key_points = self.select_key_points(bbox, grey_prev_frame)

        result = [(bbox.copy(), prob.copy(), class_id.copy(), roi_id.copy())]
        for present_frame in tracking_frame_list:
            grey_present_frame = cv2.cvtColor(present_frame, cv2.COLOR_BGR2GRAY)
            new_points, status, error = cv2.calcOpticalFlowPyrLK(grey_prev_frame, grey_present_frame, key_points, None)

            if len(key_points) > 0 and len(new_points) > 0:
                bbox, prob, class_id, roi_id = self.update_bounding_boxes((bbox, prob, class_id, roi_id),
                                                                          key_points, new_points, status)
                key_points = new_points[status == 1].reshape(-1, 1, 2)

            result.append((bbox.copy(), prob.copy(), class_id.copy(), roi_id.copy()))

            grey_prev_frame = grey_present_frame.copy()

        return result

    @staticmethod
    def select_key_points(bounding_boxes, gray_image, max_corners=10, quality_level=0.01, min_distance=1):
        points = []
        for (x1, y1, x2, y2) in bounding_boxes:
            roi = gray_image[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=quality_level,
                                              minDistance=min_distance)
            if corners is not None:
                corners += np.array([x1, y1], dtype=np.float32)
                points.extend(corners.tolist())

        return np.array(points, dtype=np.float32) if points else np.empty((0, 1, 2), dtype=np.float32)

    @staticmethod
    def update_bounding_boxes(content_result, old_points, new_points, status):
        bounding_boxes, probs, class_ids, roi_ids = content_result
        updated_boxes = []
        updated_probs = []
        updated_classes = []
        updated_roi_ids = []
        point_movements = new_points - old_points
        for box, prob, class_id, rid in zip(bounding_boxes, probs, class_ids, roi_ids):

            x1, y1, x2, y2 = box
            points_in_box = ((old_points[:, 0, 0] >= x1) & (old_points[:, 0, 0] < x2) &
                             (old_points[:, 0, 1] >= y1) & (old_points[:, 0, 1] < y2)).reshape(-1)
            valid_points_in_box = points_in_box & (status.flatten() == 1)
            if not np.any(valid_points_in_box):
                continue
            average_movement = np.mean(point_movements[valid_points_in_box], axis=0).reshape(-1)
            dx, dy = average_movement[0], average_movement[1]
            updated_box = (x1 + int(dx), y1 + int(dy), x2 + int(dx), y2 + int(dy))
            updated_boxes.append(updated_box)
            updated_probs.append(prob)
            updated_classes.append(class_id)
            updated_roi_ids.append(rid)
        return np.asarray(updated_boxes), np.asarray(updated_probs), \
            np.asarray(updated_classes), np.asarray(updated_roi_ids)
