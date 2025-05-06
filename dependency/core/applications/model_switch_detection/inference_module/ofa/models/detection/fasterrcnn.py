from torchvision.models.detection import FasterRCNN

def get_faster_rcnn(backbone_with_fpn, num_classes: int = 91):
    return FasterRCNN(backbone_with_fpn, num_classes, min_size=240)