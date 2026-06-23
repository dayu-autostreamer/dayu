import abc

from .base_extraction import BaseExtraction
from core.lib.common import ClassFactory, ClassType

from core.lib.common import LOGGER
import os
import cv2
import numpy as np
import math
import time
import asyncio
import threading

__all__ = ('ObjectVelocityExtraction',)


@ClassFactory.register(ClassType.PRO_SCENARIO, alias='obj_velocity')
class ObjectVelocityExtraction(BaseExtraction, abc.ABC):
    def __init__(self):
        super().__init__()
        self.pre_frame = None
        self.pre_bbox = None
        self.cur_frame = None
        self.cur_bbox = None
        self.cap_fps = None
        self.obj_speed = 0

        self._loop = asyncio.new_event_loop() 
        self._stop_event = threading.Event() 
        self._thread = threading.Thread(target=self._run_loop, daemon=True)  #负责循环的线程
        self._thread.start()

    def __call__(self, result, task):

        # LOGGER.info(f'ready for get obj_speed, test if TrackerCSRT_create exists')

        data_file_path = task.get_file_path()
        cap = cv2.VideoCapture(data_file_path)
        image_list = []
        success, frame = cap.read()
        while success:
            self.frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            image_list.append(frame)
            success, frame = cap.read()

        if len(image_list) < 2:
            LOGGER.critical('ERROR: image list length is less than 2')
            LOGGER.critical(f'Source: {task.get_source_id()}, Task: {task.get_task_id()}')
            LOGGER.critical(f'file_path: {task.get_file_path()}')
            return 0
    
        bboxes_list = []
        for frame_result in result:
            bboxes = frame_result[0]
            bboxes_list.append(bboxes)

        pre_frame = image_list[-2]
        pre_bbox = bboxes_list[-2]

        cur_frame = image_list[-1]
        cur_bbox = bboxes_list[-1]

        cap_fps =task.get_metadata()['fps']

        # LOGGER.info(f'before len(pre_bbox)={len(pre_bbox)}, len(cur_bbox)={len(cur_bbox)}')

        cur_obj_speed = self.update_and_cal_obj_speed(pre_frame=pre_frame,
                                                      pre_bbox=pre_bbox,
                                                      cur_frame=cur_frame,
                                                      cur_bbox=cur_bbox,
                                                      cap_fps=cap_fps)
        
        return cur_obj_speed

    def update_and_cal_obj_speed(self, pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):

        # LOGGER.info(f'ready for update_and_cal_obj_speed, info')
        self.update_scenario(pre_frame=pre_frame,
                             pre_bbox=pre_bbox,
                             cur_frame=cur_frame,
                             cur_bbox=cur_bbox,
                             cap_fps=cap_fps)
        
        return self.get_obj_speed

    def _run_loop(self):
        # 设置事件循环在后台线程中运行
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._compute_loop())
        finally:
            self._loop.close()

    async def _compute_loop(self):
        
        async def compute_f(pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):
            cal_result=cal_obj_speed_by_my_tracker(pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps)
            return cal_result

        while not self._stop_event.is_set():
            
            tmp_pre_frame = self.pre_frame
            tmp_pre_bbox = self.pre_bbox
            tmp_cur_frame = self.cur_frame
            tmp_cur_bbox = self.cur_bbox
            tmp_cap_fps = self.cap_fps

            if (tmp_pre_frame is not None) and (tmp_cur_frame is not None) and (tmp_pre_bbox is not None) and (tmp_cur_bbox is not None):

                if (len(tmp_pre_bbox)>0) and (len(tmp_cur_bbox)>0):

                    new_obj_speed = await compute_f(tmp_pre_frame, tmp_pre_bbox, tmp_cur_frame, tmp_cur_bbox, tmp_cap_fps)
                    with self._lock:
                        self.obj_speed = new_obj_speed
            
            await asyncio.sleep(0.01)

    def update_scenario(self, pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):

        # LOGGER.info(f'update wait for lock')
        with self._lock:   
            self.pre_frame = pre_frame
            self.pre_bbox = pre_bbox
            self.cur_frame = cur_frame
            self.cur_bbox = cur_bbox
            self.cap_fps = cap_fps

            # LOGGER.info(f'update_scenario cap.fps = {cap_fps}')
    

    @property
    def get_obj_speed(self):
        with self._lock:
            return self.obj_speed
    
    def stop(self):
        self._stop_event.set()
        self._thread.join()
    
    _lock = threading.Lock()





def cal_obj_speed_by_my_tracker(pre_frame, pre_bbox, cur_frame, cur_bbox, cap_fps):
    # 首先将两帧都转换为1080p的图像，且将两帧的bbox也转化为1080p下的坐标
    # LOGGER.info(f'begin cal_obj_speed len_pre_bbox = {len(pre_bbox)},len_cur_bbox = {len(cur_bbox)}')
    pre_frame_1 = cv2.resize(pre_frame, (1920, 1080))
    pre_bbox_1 = []
    for i in range(len(pre_bbox)):
        x_min = pre_bbox[i][0] / pre_frame.shape[1] * 1920
        y_min = pre_bbox[i][1] / pre_frame.shape[0] * 1080
        x_max = pre_bbox[i][2] / pre_frame.shape[1] * 1920
        y_max = pre_bbox[i][3] / pre_frame.shape[0] * 1080
        # LOGGER.info(f'x_min={x_min},x_max={x_max},y_min={y_min},y_max={y_max}')

        pre_bbox_1.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    
    cur_frame_1 = cv2.resize(cur_frame, (1920, 1080))
    cur_bbox_1 = []
    for i in range(len(cur_bbox)):
        x_min = cur_bbox[i][0] / cur_frame.shape[1] * 1920
        y_min = cur_bbox[i][1] / cur_frame.shape[0] * 1080
        x_max = cur_bbox[i][2] / cur_frame.shape[1] * 1920
        y_max = cur_bbox[i][3] / cur_frame.shape[0] * 1080
        # LOGGER.info(f'x_min={x_min},x_max={x_max},y_min={y_min},y_max={y_max}')

        cur_bbox_1.append([int(x_min), int(y_min), int(x_max), int(y_max)])
    
    
    # 利用tracking+匹配的方式计算目标速度
    speed_list = []

    for i in range(len(pre_bbox_1)):
        pre_frame_bbox = pre_bbox_1[i]
        pre_x_min, pre_y_min, pre_x_max, pre_y_max = pre_frame_bbox  # 前一帧中目标的bbox
        
        ok, track_bbox = Tracking().track_bbox(pre_frame = pre_frame_1,
                                               pre_bbox_single = pre_frame_bbox,
                                               cur_frame = cur_frame_1)
        
        if ok:  # 如果追踪成功，则将追踪之后bbox与当前帧真实的所有目标的bbox进行配对（利用iou），从而实现两帧之间目标的匹配
            # LOGGER.info(f'track is OK')
            track_x_min = int(track_bbox[0])
            track_y_min = int(track_bbox[1])
            track_x_max = int(track_bbox[2])
            track_y_max = int(track_bbox[3])
            
            temp_iou_list = []
            for temp_box in cur_bbox_1:
                temp_iou = cal_iou([track_x_min, track_y_min, track_x_max, track_y_max], temp_box)
                temp_iou_list.append(temp_iou)
            
            if len(temp_iou_list)>0:
            
                obj_index = np.argmax(np.array(temp_iou_list))  # 与track之后的bbox重叠程度最大的bbox为该目标在当前帧中的bbox
                
                if temp_iou_list[obj_index] >= 0.1:  # track之后的bbox与真实的bbox之间一定要满足一定程度的重叠，否则认为是误判
                    # 此时说明完成了前一帧中的目标和当前帧中的目标之间的匹配，可以计算速度
                    temp_box = cur_bbox_1[obj_index]
                    temp_center = ((temp_box[0] + temp_box[2]) / 2, (temp_box[1] + temp_box[3]) / 2)
                    pre_center = ((pre_x_min + pre_x_max) / 2, (pre_y_min + pre_y_max) / 2)
                    
                    temp_speed_x = math.fabs((temp_center[0] - pre_center[0]))  * cap_fps
                    temp_speed_y = math.fabs((temp_center[1] - pre_center[1]))  * cap_fps
                    temp_speed = (temp_speed_x ** 2 + temp_speed_y ** 2) ** 0.5
                    
                    speed_list.append(temp_speed)
        
        # LOGGER.info(f'track is NOT OK')
    
    if len(speed_list) == 0:
        # print('算出来速度是0')
        # LOGGER.info(f'no good, len(speed_list) == 0')
        return 0
    # LOGGER.info(f'good, obj_speed = {np.max(speed_list)}')

    return np.max(speed_list)  # 返回速度的最大值，因为最大值决定帧率
  

def cal_iou(predict_bbox, gt_bbox):
    xmin1, ymin1, xmax1, ymax1 = predict_bbox
    xmin2, ymin2, xmax2, ymax2 = gt_bbox
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积

    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2  # iou = a1/ (s1 + s2 - a1)
    return iou



class Tracking:

    def __init__(self):
        pass

    def track_bbox(self, pre_frame, pre_bbox_single, cur_frame):

        bounding_boxes = [pre_bbox_single]

        grey_prev_frame = cv2.cvtColor(pre_frame, cv2.COLOR_BGR2GRAY)
        
        key_points = self.select_key_points(bounding_boxes = bounding_boxes,
                                            gray_image = grey_prev_frame)

        grey_present_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        new_points, status, error = cv2.calcOpticalFlowPyrLK(grey_prev_frame, grey_present_frame, key_points, None)

        new_bounding_boxes = None
        if len(key_points) > 0 and len(new_points) > 0:
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
        for (x1, y1, x2, y2) in bounding_boxes:
            roi = gray_image[y1:y2, x1:x2]
            corners = cv2.goodFeaturesToTrack(roi, maxCorners=max_corners, qualityLevel=quality_level,
                                              minDistance=min_distance)
            if corners is not None:
                corners += np.array([x1, y1], dtype=np.float32)
                points.extend(corners.tolist())

        return np.array(points, dtype=np.float32) if points else np.empty((0, 1, 2), dtype=np.float32)


    def update_bounding_boxes(self, bounding_boxes, old_points, new_points, status):

        updated_boxes = []
        point_movements = new_points - old_points

        for box in bounding_boxes:

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

        return np.asarray(updated_boxes)