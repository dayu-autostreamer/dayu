'''
删除原模型中的多余部分，只保留backbone
'''

import torch.nn as nn

# 设置一个空操作
class Identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
        
def cleanup_backbone(backbone):

    backbone.final_expand_layer = Identity()
    backbone.global_avg_pool = Identity()
    backbone.feature_mix_layer = Identity()
    backbone.classifier = Identity()
    return backbone
