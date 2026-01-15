from torch import nn, Tensor
from typing import Dict, List
from collections import OrderedDict
from torch.nn import functional as F
from models.ops.dynamic_conv2d import DynamicConv2d

from typing import Tuple

class CustomLastLevelMaxPool(nn.Module):
    """
    Custom MaxPool for the last level.
    """
    def forward(self, x: List[Tensor], y: List[str], z: Dict[str, Tensor]) -> Tuple[List[Tensor], List[str]]:
        names = y.copy()  # Create a copy to avoid modifying the original list
        names.append('pool')
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # stride=2
        return x, names

class CustomFPN(nn.Module):
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks=None,
    ):
        super().__init__()

        # lateral connections (replace the original 1x1 conv with dynamic conv)
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                DynamicConv2d(in_channels, out_channels, kernel_size=1)
            )

        # output convs (keep the original 3x3 conv)
        self.output_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.output_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )

        if extra_blocks is None:
            extra_blocks = CustomLastLevelMaxPool()
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """Get the result of a lateral connection."""
        return self.lateral_convs[idx](x)

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """Get the result of an output conv."""
        return self.output_convs[idx](x)

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Sort input feature maps by name
        names = list(x.keys())
        x_names = sorted(names)  

        # Top-down pathway starting from the last level
        last_inner = self.get_result_from_inner_blocks(x[x_names[-1]], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x_names) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[x_names[idx]], idx)
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:],
                                         mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, x_names = self.extra_blocks(results, x_names, x)

        # Return an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(x_names, results)])
        return out