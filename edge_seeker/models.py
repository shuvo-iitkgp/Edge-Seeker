import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SmallGCN(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, out_dim: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.lin = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        logits = self.lin(x)
        return logits


class DeepGCN(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        # first layer
        self.convs.append(GCNConv(in_dim, hid_dim))
        # remaining layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hid_dim, hid_dim))

        self.lin = nn.Linear(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lin(x)
        return logits
