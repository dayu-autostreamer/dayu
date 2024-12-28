from annoy import AnnoyIndex
import numpy as np


class AccEstimator:
    def __init__(self, ground_truth_file: str):

        with open(ground_truth_file, 'r') as gt_f:
            self.data_gt = gt_f.readlines()

    def calculate_accuracy(self, frame_hash_codes, predictions, resolution_ratio, fps_ratio):
        acc_list = []
        gt_frames_index_list = self.find_gt_frames_index(fps_ratio, frame_hash_codes)

        # no object in scene
        if not gt_frames_index_list:
            return 1

        # no prediction
        if not predictions:
            return 0

        for prediction, gt_frames_index in zip(predictions, gt_frames_index_list):
            prediction_normal = [{'bbox': box, 'prob': prob, 'class': 1} for box, prob in
                                 zip(prediction[0], prediction[1])]
            for gt_frame_index in gt_frames_index:
                frame_gt = self.get_frame_ground_truth(gt_frame_index, resolution_ratio)
                acc = self.calculate_map(prediction_normal, frame_gt)
                acc_list.append(acc)

        return np.mean(acc_list) if acc_list else 0

    def find_gt_frames_index(self, fps_ratio, frame_hash_codes):
        gt_frames_index_list = []

        if fps_ratio <= 0.5:
            add_frame_num = int(1 / fps_ratio)
            for hash_data in frame_hash_codes:
                frame_index = self.search_frame_index(hash_data)
                gt_frames_index_list.append([frame_index + i for i in range(add_frame_num)])
        elif 0.5 < fps_ratio < 1.0:
            add_frame_interval = int(1 / (1 - fps_ratio))
            for iter_num, hash_data in enumerate(frame_hash_codes):
                frame_index = self.search_frame_index(hash_data)
                gt_frames_index_list.append([frame_index, frame_index + 1]
                                            if (iter_num + 1) % add_frame_interval == 0 else [frame_index])
        else:
            gt_frames_index_list = [[self.search_frame_index(hash_data)] for hash_data in frame_hash_codes]

        return gt_frames_index_list

    def get_frame_ground_truth(self, index, resolution_ratio):
        if index >= len(self.data_gt):
            return []

        info = self.data_gt[index].strip()

        info = info.split(' ')
        assert int(info[0]) == index, f'frame index {index} is not equal to ground truth index {info[0]}'

        info = info[1:]

        bbox_gt = [float(b) for b in info]
        boxes_gt = np.array(bbox_gt, dtype=np.float32).reshape(-1, 4)
        frame_gt = []
        for box in boxes_gt.tolist():
            box[0] *= resolution_ratio[0]
            box[1] *= resolution_ratio[1]
            box[2] *= resolution_ratio[0]
            box[3] *= resolution_ratio[1]
            frame_gt.append({'bbox': box, 'class': 1})
        return frame_gt

    def search_frame_index(self, hash_data):
        # closest_frame_index = self.hash_table.get_nns_by_vector(np.array(hash_data, dtype=int), 1)[0]
        closest_frame_index = hash_data
        return closest_frame_index

    @staticmethod
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    @staticmethod
    def compute_ap(recalls, precisions):
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        for i in range(precisions.size - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
        return ap

    @staticmethod
    def calculate_map(predictions, ground_truths, iou_threshold=0.5):
        """
        predictions: list of dicts with keys {'bbox': [x1, y1, x2, y2], 'prob': confidence, 'class': class_id}
        ground_truths: list of dicts with keys {'bbox': [x1, y1, x2, y2], 'class': class_id}
        """

        # no object in scene
        if not ground_truths:
            return 1

        # no prediction
        if not predictions:
            return 0

        aps = []
        for class_id in set([gt['class'] for gt in ground_truths]):
            # Filter predictions and ground truths by class
            preds = [p for p in predictions if p['class'] == class_id]
            gts = [gt for gt in ground_truths if gt['class'] == class_id]

            # Sort predictions by confidence
            preds.sort(key=lambda x: x['prob'], reverse=True)

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            used_gts = set()

            for i, pred in enumerate(preds):
                max_iou = 0
                max_gt_idx = -1
                for j, gt in enumerate(gts):
                    iou = AccEstimator.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou > max_iou and j not in used_gts:
                        max_iou = iou
                        max_gt_idx = j

                if max_iou >= iou_threshold:
                    tp[i] = 1
                    used_gts.add(max_gt_idx)
                else:
                    fp[i] = 1

            # Calculate precision and recall
            fp_cumsum = np.cumsum(fp)
            tp_cumsum = np.cumsum(tp)
            recalls = tp_cumsum / len(gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            ap = AccEstimator.compute_ap(recalls, precisions)
            aps.append(ap)

        # Mean AP
        mAP = np.mean(aps)
        return mAP
