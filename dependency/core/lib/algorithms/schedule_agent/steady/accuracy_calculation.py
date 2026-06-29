

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
    def get_real_acc(self, det_task:Task, gt_task:Task):

        det_per_frame_bbox_list = self.get_bbox_list_from_task(det_task)
        gt_per_frame_bbox_list = self.get_bbox_list_from_task(gt_task)

        # 计算精度的时候，我只需要det_task的最后一个结果，以及gt_task的第一个结果
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


    # 获取task中的bbox
    @classmethod
    def get_bbox_list_from_task(self, task:Task):

        bbox_list = []
        task_content = task.get_first_content()
        for frame_content in task_content:
            bbox_list.append(frame_content[0])
        return bbox_list

    # 基于两个task计算真实精度
    # 首先需要获取对应的分辨率
    @classmethod
    def calculate_accuracy(self, gt_boxes, det_boxes, gt_width, gt_height, det_width, det_height, iou_threshold=0.5):
        """
        评估检测结果与 ground truth 的匹配程度。
        
        参数:
            gt_boxes (list/np.array): Ground truth 边界框列表，格式为 [[x_min, y_min, x_max, y_max], ...]
            det_boxes (list/np.array): 检测结果边界框列表，格式为 [[x_min, y_min, x_max, y_max], ...]
            gt_width (int): Ground truth 分辨率宽度
            gt_height (int): Ground truth 分辨率高度
            det_width (int): 检测结果分辨率宽度
            det_height (int): 检测结果分辨率高度
            iou_threshold (float): IoU 阈值，大于此值认为检测正确
        
        返回:
            dict: 包含 TP、FP、FN 的统计结果
        """
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        # 初始化匹配标记（标记 GT 是否已被匹配）
        gt_matched = [False] * len(gt_boxes)
        
        # 遍历所有检测结果
        for det_box in det_boxes:
            # 将检测结果映射到 GT 的分辨率
            det_box_mapped = self.map_bbox_to_resolution(
                det_box, det_width, det_height, gt_width, gt_height
            )
            
            max_iou = 0.0
            best_gt_idx = -1
            
            # 遍历所有 GT 边界框，找到最佳匹配
            for i, gt_box in enumerate(gt_boxes):
                iou = self.calculate_iou(gt_box, det_box_mapped)
                if iou > max_iou:
                    max_iou = iou
                    best_gt_idx = i
            
            # 判断是否匹配成功
            if max_iou >= iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp += 1
            else:
                fp += 1
        
        # 统计未匹配的 GT（漏检）
        fn = sum([not matched for matched in gt_matched])
        
        # 最终使用recall召回率作最终的精度，也就是看看漏检了多少
        return {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }


    # 该函数将一种分辨率下的box映射为另一种分辨率 
    @classmethod
    def map_bbox_to_resolution(self, box, src_width, src_height, dst_width, dst_height):
        """
        将边界框从源分辨率映射到目标分辨率。
        
        参数:
            box (list/np.array): 源边界框 [x_min, y_min, x_max, y_max]
            src_width (int): 源分辨率宽度
            src_height (int): 源分辨率高度
            dst_width (int): 目标分辨率宽度
            dst_height (int): 目标分辨率高度
        
        返回:
            list: 映射后的边界框 [x_min, y_min, x_max, y_max]
        """
        x_min = box[0] * (dst_width / src_width)
        y_min = box[1] * (dst_height / src_height)
        x_max = box[2] * (dst_width / src_width)
        y_max = box[3] * (dst_height / src_height)
        return [int(x_min), int(y_min), int(x_max), int(y_max)]

    @classmethod
    def calculate_iou(self,box1, box2):
        """
        计算两个边界框的 IoU（Intersection over Union）。
        
        参数:
            box1 (list/np.array): [x_min, y_min, x_max, y_max]
            box2 (list/np.array): [x_min, y_min, x_max, y_max]
        
        返回:
            float: IoU 值
        """
        # 计算交集区域的坐标
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # 检查是否有交集
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # 计算交集区域面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # 计算两个边界框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集区域面积
        union_area = box1_area + box2_area - intersection_area
        
        # 计算 IoU
        iou = intersection_area / union_area
        return iou




    

