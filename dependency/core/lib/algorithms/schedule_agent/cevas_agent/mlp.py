import torch.nn as nn


# 输入t-1到t-h时间段内AOPF/每个逻辑节点的数据输入量
# 输出预测结果,t+1时刻的各个逻辑节点:数据输入大小/边缘节点资源需求/云开销
class MLP(nn.Module):

    def __init__(self, logic_node_num=3):
        super(MLP, self).__init__()
        self.time_slot = 3
        input_size = self.time_slot * (1 + logic_node_num)  # 总共time_slot个[1,1+logic_node_num]向量 展平
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),  # 第一个隐藏层
            nn.ReLU(),
            nn.Linear(512, 128),  # 第二个隐藏层
            nn.ReLU(),
            nn.Linear(128, 4 * logic_node_num)  # 每个节点输入数据量大小/边缘节点CPU限制/内存限制/云开销
        )

    def forward(self, x):
        return self.layers(x.float())
