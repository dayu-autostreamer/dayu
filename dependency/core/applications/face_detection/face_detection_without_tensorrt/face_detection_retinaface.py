import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import sys

from core.lib.common import Context, LOGGER

# Import models module to make it available for torch.load
from . import models
# Register models module in sys.modules so torch.load can find it
sys.modules['models'] = models


# Configuration for MobileNet backbone
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


class PriorBox(object):
    """Prior box generator for RetinaFace"""
    def __init__(self, cfg, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        from math import ceil
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        from itertools import product
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode(loc, priors, variances):
    """Decode locations from predictions"""
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landmarks from predictions"""
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class FaceDetectionRetinaFace:
    """Face detection using RetinaFace model"""
    
    def __init__(self, weights, device=0):
        self.weights = weights
        self.device_id = device
        self.class_id = 'face'
        
        # Detection parameters
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.vis_thres = 0.6
        
        # Use MobileNet configuration
        self.cfg = cfg_mnet
        
        # Setup device
        if torch.cuda.is_available() and device >= 0:
            self.device = torch.device(f'cuda:{device}')
            self.use_cpu = False
        else:
            self.device = torch.device('cpu')
            self.use_cpu = True
            
        torch.set_grad_enabled(False)
        cudnn.benchmark = True
        
        # Load model
        self.model = None
        self._load_model()
        
        LOGGER.info(f"FaceDetectionRetinaFace initialized on device: {self.device}")
        
        # Warmup
        self._warmup()

    def _load_model(self):
        """Load the RetinaFace model"""
        try:
            LOGGER.info(f'Loading RetinaFace model from {self.weights}')
            
            # Load the entire model directly (compatible with older PyTorch versions)
            if self.use_cpu:
                # For older PyTorch versions, don't use weights_only parameter
                try:
                    self.model = torch.load(self.weights, map_location='cpu')
                except TypeError:
                    # Fallback for very old versions
                    self.model = torch.load(self.weights, map_location='cpu')
            else:
                try:
                    self.model = torch.load(self.weights, map_location=self.device)
                except TypeError:
                    # Fallback for very old versions
                    self.model = torch.load(self.weights, map_location=self.device)
                
            self.model.eval()
            self.model = self.model.to(self.device)
            
            LOGGER.info('Successfully loaded RetinaFace model')
            
        except Exception as e:
            LOGGER.error(f'Error loading RetinaFace model: {e}')
            LOGGER.exception(e)
            raise

    def _warmup(self):
        """Warmup the model"""
        LOGGER.info("Warming up...")
        try:
            # Create dummy images for warmup
            for _ in range(5):
                dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                self.infer(dummy_img)
            LOGGER.info("Warmup completed")
        except Exception as e:
            LOGGER.warning(f"Warmup failed: {e}")

    def _preprocess_image(self, raw_image):
        """Preprocess image for inference"""
        img = np.float32(raw_image)
        im_height, im_width, _ = img.shape
        
        # Subtract mean
        img -= (104, 117, 123)
        
        # Transpose to CHW format
        img = img.transpose(2, 0, 1)
        
        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        
        return img, im_height, im_width

    def _post_process(self, loc, conf, landms, im_height, im_width):
        """Post-process model outputs"""
        resize = 1
        
        # Create scale tensor
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        scale = scale.to(self.device)
        
        # Generate priors
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        
        # Decode boxes
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        
        # Get scores
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        
        # Decode landmarks
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                               im_width, im_height, im_width, im_height,
                               im_width, im_height])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        
        # Filter by confidence
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        
        # Keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        
        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]
        
        # Keep top-K after NMS
        dets = dets[:self.keep_top_k, :]
        landms = landms[:self.keep_top_k, :]
        
        return dets, landms

    def infer(self, raw_image):
        """Run inference on an image
        
        Args:
            raw_image: Input image in BGR format (numpy array)
            
        Returns:
            result_boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            result_scores: List of confidence scores
            result_class_id: List of class IDs (all 'face')
            result_roi_id: List of ROI IDs (sequential indices)
        """
        try:
            if self.model is None:
                LOGGER.error("Model is not loaded")
                return [], [], [], []
            
            # Preprocess
            img, im_height, im_width = self._preprocess_image(raw_image)
            
            # Forward pass
            loc, conf, landms = self.model(img)
            
            # Post-process
            dets, landms = self._post_process(loc, conf, landms, im_height, im_width)
            
            # Filter by visualization threshold and format output
            result_boxes = []
            result_scores = []
            
            for det in dets:
                if det[4] >= self.vis_thres:
                    # Box coordinates
                    box = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
                    score = float(det[4])
                    
                    result_boxes.append(box)
                    result_scores.append(score)
            
            # Create class IDs and ROI IDs
            result_class_id = [self.class_id] * len(result_boxes)
            result_roi_id = list(range(len(result_boxes)))
            
            return result_boxes, result_scores, result_class_id, result_roi_id
            
        except Exception as e:
            LOGGER.warning(f"Model inference failed: {e}, returning empty results")
            LOGGER.exception(e)
            return [], [], [], []

