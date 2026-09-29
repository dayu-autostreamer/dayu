import math
import threading
from collections import deque

import cv2
import numpy as np
from core.lib.common import LOGGER, ClassFactory, ClassType, FileOps

from .base_extraction import BaseExtraction

__all__ = ("ObjectVelocityExtraction",)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias="obj_velocity")
class ObjectVelocityExtraction(BaseExtraction):
    def __init__(self):
        super().__init__()
        self.pre_frame = None
        self.pre_bbox = None
        self.cur_frame = None
        self.cur_bbox = None
        self.cap_fps = None
        self.obj_speed = 0

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._compute_loop, daemon=True)
        self._thread.start()

    def __call__(self, result, task):
        if not isinstance(result, dict):
            return 0
        outputs = result.get("outputs")
        if not isinstance(outputs, dict):
            return 0

        data_file_path = FileOps.get_task_file_in_temp(task)
        cap = cv2.VideoCapture(data_file_path)
        image_list = deque(maxlen=2)
        try:
            success, frame = cap.read()
            while success:
                image_list.append(frame)
                success, frame = cap.read()
        finally:
            cap.release()

        if len(image_list) < 2:
            LOGGER.critical("ERROR: image list length is less than 2")
            LOGGER.critical(f"Source: {task.get_source_id()}, Task: {task.get_task_id()}")
            LOGGER.critical(f"file_path: {task.get_file_path()}")
            return 0

        bboxes_list = []
        for record in outputs.get("bbox", []) or []:
            if not isinstance(record, dict):
                continue
            bboxes = [
                item.get("bbox")
                for item in record.get("items") or []
                if isinstance(item, dict) and len(item.get("bbox") or []) == 4
            ]
            bboxes_list.append(bboxes)
        if len(bboxes_list) < 2:
            return 0

        pre_frame = image_list[-2]
        pre_bbox = bboxes_list[-2]

        cur_frame = image_list[-1]
        cur_bbox = bboxes_list[-1]

        cap_fps = task.get_metadata()["fps"]

        cur_obj_speed = self.update_and_cal_obj_speed(
            pre_frame=pre_frame, pre_bbox=pre_bbox, cur_frame=cur_frame, cur_bbox=cur_bbox, cap_fps=cap_fps
        )

        return cur_obj_speed

    def update_and_cal_obj_speed(self, pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):
        self.update_scenario(
            pre_frame=pre_frame, pre_bbox=pre_bbox, cur_frame=cur_frame, cur_bbox=cur_bbox, cap_fps=cap_fps
        )

        return self.get_obj_speed

    def _compute_loop(self):
        while not self._stop_event.wait(0.01):
            with self._lock:
                previous, current = self.pre_frame, self.cur_frame
                previous_boxes, current_boxes = self.pre_bbox, self.cur_bbox
                fps = self.cap_fps
            if previous is None or current is None or not previous_boxes or not current_boxes:
                continue
            try:
                speed = cal_obj_speed_by_my_tracker(previous, previous_boxes, current, current_boxes, fps)
                with self._lock:
                    self.obj_speed = speed
            except cv2.error:
                LOGGER.exception("Could not estimate object velocity from the current frame pair")

    def update_scenario(self, pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):
        with self._lock:
            self.pre_frame = pre_frame
            self.pre_bbox = pre_bbox
            self.cur_frame = cur_frame
            self.cur_bbox = cur_bbox
            self.cap_fps = cap_fps

    @property
    def get_obj_speed(self):
        with self._lock:
            return self.obj_speed

    def stop(self):
        self._stop_event.set()
        self._thread.join()


def cal_obj_speed_by_my_tracker(pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):
    pre_frame_1 = cv2.resize(pre_frame, (1920, 1080))
    pre_bbox_1 = []
    for i in range(len(pre_bbox)):
        x_min = pre_bbox[i][0] / pre_frame.shape[1] * 1920
        y_min = pre_bbox[i][1] / pre_frame.shape[0] * 1080
        x_max = pre_bbox[i][2] / pre_frame.shape[1] * 1920
        y_max = pre_bbox[i][3] / pre_frame.shape[0] * 1080

        pre_bbox_1.append([int(x_min), int(y_min), int(x_max), int(y_max)])

    cur_frame_1 = cv2.resize(cur_frame, (1920, 1080))
    cur_bbox_1 = []
    for i in range(len(cur_bbox)):
        x_min = cur_bbox[i][0] / cur_frame.shape[1] * 1920
        y_min = cur_bbox[i][1] / cur_frame.shape[0] * 1080
        x_max = cur_bbox[i][2] / cur_frame.shape[1] * 1920
        y_max = cur_bbox[i][3] / cur_frame.shape[0] * 1080

        cur_bbox_1.append([int(x_min), int(y_min), int(x_max), int(y_max)])

    speed_list = []

    for i in range(len(pre_bbox_1)):
        pre_frame_bbox = pre_bbox_1[i]
        pre_x_min, pre_y_min, pre_x_max, pre_y_max = pre_frame_bbox

        ok, track_bbox = Tracking().track_bbox(
            pre_frame=pre_frame_1, pre_bbox_single=pre_frame_bbox, cur_frame=cur_frame_1
        )

        if ok:
            track_x_min = int(track_bbox[0])
            track_y_min = int(track_bbox[1])
            track_x_max = int(track_bbox[2])
            track_y_max = int(track_bbox[3])

            temp_iou_list = []
            for temp_box in cur_bbox_1:
                temp_iou = cal_iou([track_x_min, track_y_min, track_x_max, track_y_max], temp_box)
                temp_iou_list.append(temp_iou)

            if len(temp_iou_list) > 0:
                obj_index = np.argmax(np.array(temp_iou_list))

                if temp_iou_list[obj_index] >= 0.1:
                    temp_box = cur_bbox_1[obj_index]
                    temp_center = ((temp_box[0] + temp_box[2]) / 2, (temp_box[1] + temp_box[3]) / 2)
                    pre_center = ((pre_x_min + pre_x_max) / 2, (pre_y_min + pre_y_max) / 2)

                    temp_speed_x = math.fabs((temp_center[0] - pre_center[0])) * cap_fps
                    temp_speed_y = math.fabs((temp_center[1] - pre_center[1])) * cap_fps
                    temp_speed = (temp_speed_x**2 + temp_speed_y**2) ** 0.5

                    speed_list.append(temp_speed)

    if len(speed_list) == 0:
        return 0

    return float(np.max(speed_list))


def cal_iou(predict_bbox, gt_bbox):
    xmin1, ymin1, xmax1, ymax1 = predict_bbox
    xmin2, ymin2, xmax2, ymax2 = gt_bbox
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h
    a2 = s1 + s2 - a1
    iou = a1 / a2 if a2 > 0 else 0.0
    return iou


class Tracking:
    def track_bbox(self, pre_frame, pre_bbox_single, cur_frame):

        bounding_boxes = [pre_bbox_single]

        grey_prev_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)

        key_points = self.select_key_points(bounding_boxes=bounding_boxes, gray_image=grey_prev_frame)

        grey_present_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        if len(key_points) == 0:
            return False, [0]
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(grey_prev_frame, grey_present_frame, key_points, None)

        new_bounding_boxes = None
        if new_points is not None and status is not None and len(new_points) > 0:
            new_bounding_boxes = self.update_bounding_boxes(bounding_boxes, key_points, new_points, status)

        if new_bounding_boxes is not None:
            if len(new_bounding_boxes) > 0:
                return True, new_bounding_boxes[0]
            else:
                return False, [0]

        else:
            return False, [0]

    def select_key_points(self, bounding_boxes, gray_image, max_corners=10, quality_level=0.01, min_distance=1):

        points = []
        for x1, y1, x2, y2 in bounding_boxes:
            height, width = gray_image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(x2)), min(height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = gray_image[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(
                roi, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance
            )
            if corners is not None:
                corners += np.array([x1, y1], dtype=np.float32)
                points.extend(corners.tolist())

        return np.array(points, dtype=np.float32) if points else np.empty((0, 1, 2), dtype=np.float32)

    def update_bounding_boxes(self, bounding_boxes, old_points, new_points, status):

        updated_boxes = []
        point_movements = new_points - old_points

        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            points_in_box = (
                (old_points[:, 0, 0] >= x1)
                & (old_points[:, 0, 0] < x2)
                & (old_points[:, 0, 1] >= y1)
                & (old_points[:, 0, 1] < y2)
            ).reshape(-1)
            valid_points_in_box = points_in_box & (status.flatten() == 1)
            if not np.any(valid_points_in_box):
                continue
            average_movement = np.mean(point_movements[valid_points_in_box], axis=0).reshape(-1)
            dx, dy = average_movement[0], average_movement[1]
            updated_box = (x1 + int(dx), y1 + int(dy), x2 + int(dx), y2 + int(dy))
            updated_boxes.append(updated_box)

        return np.asarray(updated_boxes)
