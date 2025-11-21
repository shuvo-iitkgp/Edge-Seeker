import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

def build_features(data) -> torch.Tensor:
    """
    Build node features for a PyG Data object.

    Priority:
    1. data.x if present
    2. one hot of data.node_label if present
    3. degree feature as fallback
    """
    if getattr(data, "x", None) is not None and data.x is not None:
        X = data.x.float()
    elif getattr(data, "node_label", None) is not None and data.node_label is not None:
        labels = data.node_label.view(-1).long()
        C = int(labels.max().item() + 1)
        X = F.one_hot(labels, num_classes=C).to(torch.float32)
    else:
        N = data.num_nodes
        deg = degree(data.edge_index[0], num_nodes=N).unsqueeze(1)
        X = deg.float()
    return X
