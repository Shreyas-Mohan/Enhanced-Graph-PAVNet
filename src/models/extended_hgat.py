import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv


class MultimodalGatedAdapter(nn.Module):
    """
    Multimodal Gated Cross-Attention Adapter for Extended-HGAT.
    Fuses 576-dim ViT visual stream with handcrafted morphological/radiomic prior stream
    via learned dynamic gating weights [w_vis, w_pri].
    """
    def __init__(self, visual_dim=576, prior_dim=26, out_dim=256):
        super().__init__()
        self.visual_proj = Linear(visual_dim, out_dim)
        self.prior_proj = Linear(prior_dim, out_dim)
        self.gate = nn.Sequential(
            Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            Linear(out_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.bn = BatchNorm1d(out_dim)

    def forward(self, x):
        coords_and_dir = x[:, :8]
        visual = x[:, 8:584]
        extended = x[:, 584:]
        priors = torch.cat([coords_and_dir, extended], dim=1)

        h_vis = F.relu(self.visual_proj(visual))
        h_pri = F.relu(self.prior_proj(priors))

        gates = self.gate(torch.cat([h_vis, h_pri], dim=1))
        w_vis = gates[:, 0:1]
        w_pri = gates[:, 1:2]

        fused = w_vis * h_vis + w_pri * h_pri
        return self.bn(fused)


class ExtendedHGATNet(nn.Module):
    """
    Extended Hierarchical Graph Attention Network (Extended-HGAT / Best Features baseline).
    Utilizes 2-branch multimodal gated adapter and hierarchical attention blocks.
    Input dims: 592 (Rad), 599 (Rad+Vasc), 604 (All), 602 (Optimal / Best Features).
    """
    def __init__(self, in_channels=602, out_channels=2, max_hops=3):
        super().__init__()
        self.max_hops = max_hops
        self.in_channels = in_channels
        prior_dim = in_channels - 576
        self.adapter = MultimodalGatedAdapter(visual_dim=576, prior_dim=prior_dim, out_dim=256)

        gat_input_dim = 256 + (max_hops + 1)
        self.conv1 = GATConv(gat_input_dim, 64, heads=4, concat=True)
        self.bn1 = BatchNorm1d(256)

        self.conv2 = GATConv(256, 128, heads=4, concat=False)
        self.bn2 = BatchNorm1d(128)

        self.residual_proj = Linear(256, 128)
        self.gate_linear = Linear(128 + 1, 128)

        self.conv3 = GATConv(128, 64, heads=2, concat=False)
        self.bn3 = BatchNorm1d(64)

        self.classifier = Linear(64, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        fused_x = self.adapter(x)
        hop_counts = data.hierarchy_mask.argmax(dim=1).float()
        aug_x = torch.cat([fused_x, data.hierarchy_mask], dim=1)

        h1 = F.dropout(F.elu(self.bn1(self.conv1(aug_x, edge_index))), p=0.1, training=self.training)
        h2 = self.bn2(self.conv2(h1, edge_index))

        gate = torch.sigmoid(self.gate_linear(torch.cat([h2, hop_counts.unsqueeze(1)], dim=1)))
        h2 = F.dropout(F.elu((h2 * gate) + self.residual_proj(fused_x)), p=0.1, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        return F.log_softmax(self.classifier(h3), dim=1)


# Compatibility aliases
OriginalExtendedHGATNet = ExtendedHGATNet
OriginalMultimodalGatedAdapter = MultimodalGatedAdapter

