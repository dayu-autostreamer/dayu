'''
Remove redundant parts from the original model and keep only the backbone.
'''

import torch.nn as nn

# Define a no-op operation
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
