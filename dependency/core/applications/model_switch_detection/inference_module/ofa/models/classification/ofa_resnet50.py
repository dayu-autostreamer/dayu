from models.backbone.ofa_supernet import get_ofa_supernet_resnet50
from torch import nn
def get_ofa_resnet50_classification(num_classes = 1000):
    model = get_ofa_supernet_resnet50()
    if num_classes != 1000:
        model.classifier.linear.linear = nn.Linear(2048, num_classes)
    return model
