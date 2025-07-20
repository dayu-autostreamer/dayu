import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class LogicGNN(nn.Module):
    """Graph encoder for logical DAG (service graph)"""
    def __init__(self, in_feats, hidden_dim, out_dim, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_feats, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h

class PhysicalGNN(nn.Module):
    """Graph encoder for physical network topology"""
    def __init__(self, in_feats, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h