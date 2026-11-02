import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv


class BaselineHGATNet(nn.Module):
    """
    Baseline Hierarchical Graph Attention Network (Graph-PAVNet baseline).
    Input: 584-dim (3 coordinates + 1 radius + 4 direction/distance + 576 ViT patch tokens).
    """
    def __init__(self, in_channels=584, out_channels=2, max_hops=3):
        super().__init__()
        self.max_hops = max_hops
        self.in_channels = in_channels

        gat_input_dim = in_channels + (max_hops + 1)
        self.conv1 = GATConv(gat_input_dim, 64, heads=4, concat=True)
        self.bn1 = BatchNorm1d(256)

        self.conv2 = GATConv(256, 128, heads=4, concat=False)
        self.bn2 = BatchNorm1d(128)

        self.residual_proj = Linear(in_channels, 128)
        self.gate_linear = Linear(128 + 1, 128)

        self.conv3 = GATConv(128, 64, heads=2, concat=False)
        self.bn3 = BatchNorm1d(64)
        self.classifier = Linear(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        hop_counts = data.hierarchy_mask.argmax(dim=1).float()
        aug_x = torch.cat([x, data.hierarchy_mask], dim=1)

        h1 = F.dropout(F.elu(self.bn1(self.conv1(aug_x, edge_index))), p=0.1, training=self.training)
        h2 = self.bn2(self.conv2(h1, edge_index))

        gate = torch.sigmoid(self.gate_linear(torch.cat([h2, hop_counts.unsqueeze(1)], dim=1)))
        h2 = F.dropout(F.elu((h2 * gate) + self.residual_proj(x)), p=0.1, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        return F.log_softmax(self.classifier(h3), dim=1)

