from torchvision.models.detection import FCOS

def get_fcos(backbone_with_fpn, num_classes: int = 91):
    return FCOS(backbone_with_fpn, num_classes, min_size=240)