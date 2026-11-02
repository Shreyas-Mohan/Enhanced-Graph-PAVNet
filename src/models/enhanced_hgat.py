import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import GATConv


class DecoupledCrossGatedAdapter(nn.Module):
    """
    3-Branch Decoupled Cross-Gating Adapter.
    Separates inputs into three semantic modalities:
      1. Visual Stream (576 dims): Multi-planar ViT orthogonal patch tokens.
      2. Radiomic Stream (9 dims): Local HU, gradients, and 3D multi-scale LoG vesselness.
      3. Geometric Stream (17 dims): Coordinates, local caliber radius, tapering, and orientation.
    Caliber-conditioned gating balances visual features in central trunks with geometric continuity in distal vessels.
    """
    def __init__(self, in_channels=602, visual_dim=576, out_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.visual_dim = visual_dim
        self.out_dim = out_dim

        if in_channels == 602:
            self.rad_dim = 9
            self.geom_dim = 17
        elif in_channels == 604:
            self.rad_dim = 8
            self.geom_dim = 20
        elif in_channels == 599:
            self.rad_dim = 8
            self.geom_dim = 15
        elif in_channels == 592:
            self.rad_dim = 8
            self.geom_dim = 8
        else:
            self.rad_dim = 8
            self.geom_dim = max(8, in_channels - visual_dim)

        self.proj_vis = nn.Sequential(
            Linear(self.visual_dim, 256),
            BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.proj_rad = nn.Sequential(
            Linear(self.rad_dim, 128),
            BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.proj_geom = nn.Sequential(
            Linear(self.geom_dim, 128),
            BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.gate_vis = nn.Sequential(
            Linear(256 + 128, 256),
            nn.LeakyReLU(0.1, inplace=True),
            Linear(256, 256),
            nn.Sigmoid()
        )
        self.gate_rad = nn.Sequential(
            Linear(128 + 128, 128),
            nn.LeakyReLU(0.1, inplace=True),
            Linear(128, 128),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            Linear(256 + 128 + 128, out_dim),
            BatchNorm1d(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        coords_and_dir = x[:, :8]
        visual = x[:, 8:584]

        if self.in_channels == 602 and x.size(1) >= 602:
            geom_7 = x[:, 584:591]
            rad_8 = x[:, 591:599]
            norm_cal = x[:, 599:600]
            gauss_rad = x[:, 600:601]
            z_contrast = x[:, 601:602]
            x_rad = torch.cat([rad_8, z_contrast], dim=1)
            x_geom = torch.cat([coords_and_dir, geom_7, norm_cal, gauss_rad], dim=1)
        elif self.in_channels == 604 and x.size(1) >= 604:
            geom_7 = x[:, 584:591]
            topo_5 = x[:, 591:596]
            rad_8 = x[:, 596:604]
            x_rad = rad_8
            x_geom = torch.cat([coords_and_dir, geom_7, topo_5], dim=1)
        elif self.in_channels == 599 and x.size(1) >= 599:
            geom_7 = x[:, 584:591]
            rad_8 = x[:, 591:599]
            x_rad = rad_8
            x_geom = torch.cat([coords_and_dir, geom_7], dim=1)
        elif self.in_channels == 592 and x.size(1) >= 592:
            x_rad = x[:, 584:592]
            x_geom = coords_and_dir
        else:
            x_rad = coords_and_dir
            x_geom = coords_and_dir

        h_vis = self.proj_vis(visual)
        h_rad = self.proj_rad(x_rad)
        h_geom = self.proj_geom(x_geom)

        alpha_vis = self.gate_vis(torch.cat([h_vis, h_geom], dim=1))
        alpha_rad = self.gate_rad(torch.cat([h_rad, h_geom], dim=1))

        gated_vis = alpha_vis * h_vis
        gated_rad = alpha_rad * h_rad
        return self.fusion(torch.cat([gated_vis, gated_rad, h_geom], dim=1))


class EnhancedHGATNet(nn.Module):
    """
    Enhanced Hierarchical Graph Attention Network (Enhanced HGAT).
    Integrates 3-branch decoupled cross-gating, hierarchical attention blocks,
    and an auxiliary caliber regression head for diameter preservation.
    """
    def __init__(self, in_channels=602, out_channels=2, max_hops=3, aux_caliber=True):
        super().__init__()
        self.max_hops = max_hops
        self.in_channels = in_channels
        self.aux_caliber = aux_caliber

        self.adapter = DecoupledCrossGatedAdapter(in_channels=in_channels, visual_dim=576, out_dim=256)
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

        if self.aux_caliber:
            self.caliber_head = nn.Sequential(
                Linear(64, 32),
                nn.ReLU(inplace=True),
                Linear(32, 1),
                nn.ReLU()
            )

    def forward(self, data, return_aux=False):
        x, edge_index = data.x, data.edge_index

        fused_x = self.adapter(x)
        hop_counts = data.hierarchy_mask.argmax(dim=1).float()
        aug_x = torch.cat([fused_x, data.hierarchy_mask], dim=1)

        h1 = F.dropout(F.elu(self.bn1(self.conv1(aug_x, edge_index))), p=0.1, training=self.training)
        h2 = self.bn2(self.conv2(h1, edge_index))

        gate = torch.sigmoid(self.gate_linear(torch.cat([h2, hop_counts.unsqueeze(1)], dim=1)))
        h2 = F.dropout(F.elu((h2 * gate) + self.residual_proj(fused_x)), p=0.1, training=self.training)

        h3 = F.elu(self.bn3(self.conv3(h2, edge_index)))
        log_probs = F.log_softmax(self.classifier(h3), dim=1)

        if self.aux_caliber and (return_aux or self.training):
            pred_radii = self.caliber_head(h3).squeeze(-1)
            return log_probs, pred_radii
        return log_probs

