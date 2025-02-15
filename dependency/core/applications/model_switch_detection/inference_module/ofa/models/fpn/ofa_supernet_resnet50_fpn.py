from collections import OrderedDict
from typing import Dict
# from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from models.fpn.custom_fpn import CustomFPN, CustomLastLevelMaxPool
from torch import nn, Tensor

from models.ops.dynamic_conv2d import DynamicConv2d

# class Resnet50Fpn(nn.Module):
#     def __init__(self, backbone: nn.Module) -> None:
#         super().__init__()
#         in_channels_stage2 = 256
#         in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in range(1, 5)]
#         out_channels = 256
#         extra_blocks = LastLevelMaxPool()
#         self.body = backbone
#         self.fpn = FeaturePyramidNetwork(
#             in_channels_list=in_channels_list,
#             out_channels=out_channels,
#             extra_blocks=extra_blocks
#         )
#         self.out_channels = out_channels
#         self.bns = nn.ModuleList()
#         self.dynamic_convs = nn.ModuleList()
#         for i in range(4):
#             self.dynamic_convs.append(DynamicConv2d(in_channels_list[i], in_channels_list[i], kernel_size=1))
#             self.bns.append(nn.BatchNorm2d(in_channels_list[i]))

#     def forward(self, x: Tensor) -> Dict[str, Tensor]:
#         for layer in self.body.input_stem:
#             x = layer(x)
#         x = self.body.max_pooling(x)
#         mid_features = OrderedDict()
#         for stage_id, block_idx in enumerate(self.body.grouped_block_index):
#             depth_param = self.body.runtime_depth[stage_id]
#             active_idx = block_idx[: len(block_idx) - depth_param]
#             for idx in active_idx:
#                 x = self.body.blocks[idx](x)
#                 if idx == active_idx[-1]:
#                     feat = self.dynamic_convs[stage_id](x)
#                     feat = self.bns[stage_id](feat)
#                     mid_features[str(stage_id)] = feat             
#         x = self.fpn(mid_features)
#         return x

class Resnet50Fpn(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # 计算每个 stage 的通道数
        in_channels_stage2 = 256
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in range(1, 5)]
        out_channels = 192
        extra_blocks = CustomLastLevelMaxPool()
        self.body = backbone
        
        # 使用自定义 FPN 替换原来的 FPN
        self.fpn = CustomFPN(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks
        )
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # input stem
        for layer in self.body.input_stem:
            x = layer(x)
        x = self.body.max_pooling(x)
        
        # 收集每个 stage 的特征
        mid_features = OrderedDict()
        for stage_id, block_idx in enumerate(self.body.grouped_block_index):
            depth_param = self.body.runtime_depth[stage_id]
            active_idx = block_idx[: len(block_idx) - depth_param]
            for idx in active_idx:
                x = self.body.blocks[idx](x)
                if idx == active_idx[-1]:
                    # 直接使用 block 输出的特征，不再需要额外的 conv 和 bn
                    mid_features[str(stage_id)] = x
        
        x = self.fpn(mid_features)
        return x