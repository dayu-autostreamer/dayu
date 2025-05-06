from models.backbone.ofa_supernet import get_ofa_supernet_mbv3_w12
from torch import nn
def get_ofa_mbv3_w12_classification(num_classes = 1000):
    model = get_ofa_supernet_mbv3_w12()
    if num_classes != 1000:
        model.classifier.linear = nn.Linear(1536, num_classes)
    return model
