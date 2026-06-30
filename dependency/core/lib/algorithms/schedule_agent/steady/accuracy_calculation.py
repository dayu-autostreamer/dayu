from core.lib.content import Task

resolution_wh = {
    "240p": {
        "w": 320,
        "h": 240
    },
    "360p": {
        "w": 640,
        "h": 360
    },
    "480p": {
        "w": 640,
        "h": 480
    },
    "540p": {
        "w": 960,
        "h": 540
    },
    "720p": {
        "w": 1280,
        "h": 900
    },
    "900p": {
        "w": 1440,
        "h": 900
    },
    "1080p": {
        "w": 1920,
        "h": 1080
    }
}


class AccuracyCalculation():

    def __init__(self):

        pass

    @classmethod
    def get_real_acc(self, det_task: Task, gt_task: Task):

        det_per_frame_bbox_list = self.get_bbox_list_from_task(det_task)
        gt_per_frame_bbox_list = self.get_bbox_list_from_task(gt_task)

        det_boxes = det_per_frame_bbox_list[-1]
        gt_boxes = gt_per_frame_bbox_list[0]

        if len(gt_boxes) == 0 and len(det_boxes) == 0:
            return 1
        elif len(gt_boxes) > 0 and len(det_boxes) == 0:
            return 0
        elif len(gt_boxes) == 0 and len(det_boxes) > 0:
            return 0

        det_reso = det_task.get_metadata()['resolution']
        gt_reso = gt_task.get_metadata()['resolution']

        acc_info = self.calculate_accuracy(gt_boxes=gt_boxes,
                                           det_boxes=det_boxes,
                                           gt_width=resolution_wh[gt_reso]['w'],
                                           gt_height=resolution_wh[gt_reso]['h'],
                                           det_width=resolution_wh[det_reso]['w'],
                                           det_height=resolution_wh[det_reso]['h'],
                                           iou_threshold=0.5
                                           )

        return acc_info['Recall']

    @classmethod
    def get_bbox_list_from_task(self, task: Task):

        bbox_list = []
        task_content = task.get_first_content()
        for frame_content in task_content:
            bbox_list.append(frame_content[0])
        return bbox_list

    @classmethod
    def calculate_accuracy(self, gt_boxes, det_boxes, gt_width, gt_height, det_width, det_height, iou_threshold=0.5):
        """
        Evaluate how well detection boxes match the ground-truth boxes.

        Args:
            gt_boxes (list/np.array): Ground-truth boxes in
                [[x_min, y_min, x_max, y_max], ...] format.
            det_boxes (list/np.array): Detection boxes in
                [[x_min, y_min, x_max, y_max], ...] format.
            gt_width (int): Ground-truth frame width.
            gt_height (int): Ground-truth frame height.
            det_width (int): Detection frame width.
            det_height (int): Detection frame height.
            iou_threshold (float): IoU threshold for a true positive.

        Returns:
            dict: TP, FP, FN, precision, and recall statistics.
        """
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives

        gt_matched = [False] * len(gt_boxes)

        for det_box in det_boxes:
            det_box_mapped = self.map_bbox_to_resolution(
                det_box, det_width, det_height, gt_width, gt_height
            )

            max_iou = 0.0
            best_gt_idx = -1

            for i, gt_box in enumerate(gt_boxes):
                iou = self.calculate_iou(gt_box, det_box_mapped)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i

            if max_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1
            else:
                fp += 1

        fn = sum([not matched for matched in gt_matched])

        return {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }

    @classmethod
    def map_bbox_to_resolution(self, box, src_width, src_height, dst_width, dst_height):
        """
        Map a bounding box from the source resolution to the target resolution.

        Args:
            box (list/np.array): Source box [x_min, y_min, x_max, y_max].
            src_width (int): Source frame width.
            src_height (int): Source frame height.
            dst_width (int): Target frame width.
            dst_height (int): Target frame height.

        Returns:
            list: Mapped box [x_min, y_min, x_max, y_max].
        """
        x_min = box[0] * (dst_width / src_width)
        y_min = box[1] * (dst_height / src_height)
        x_max = box[2] * (dst_width / src_width)
        y_max = box[3] * (dst_height / src_height)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    @classmethod
    def calculate_iou(self, box1, box2):
        """
        Calculate the intersection-over-union value for two boxes.

        Args:
            box1 (list/np.array): [x_min, y_min, x_max, y_max].
            box2 (list/np.array): [x_min, y_min, x_max, y_max].

        Returns:
            float: IoU value.
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area
        return iou
