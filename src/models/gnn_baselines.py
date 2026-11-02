import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT as PyGGAT, GraphSAGE as PyGGraphSAGE, GCN as PyGGCN


class StandardGNNBaseline(nn.Module):
    """
    Standard GNN Baselines using official PyTorch Geometric implementations:
      - GAT (Veličković et al., ICLR 2018)
      - GraphSAGE (Hamilton et al., NeurIPS 2017)
      - GCN (Kipf & Welling, ICLR 2017)
    """
    def __init__(
        self,
        in_channels=584,
        hidden_channels=128,
        out_channels=2,
        num_layers=3,
        model_type="gat",
        heads=4,
        dropout=0.1
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.model_type == "gat":
            self.net = PyGGAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
                v2=False
            )
        elif self.model_type in ["sage", "graphsage"]:
            self.net = PyGGraphSAGE(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout
            )
        elif self.model_type == "gcn":
            self.net = PyGGCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
                out_channels=out_channels,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'gat', 'sage', or 'gcn'.")

    def forward(self, data, edge_index=None):
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            x = data.x.float()
            ei = data.edge_index
        else:
            x = data.float()
            ei = edge_index
        logits = self.net(x, ei)
        return F.log_softmax(logits, dim=-1)

