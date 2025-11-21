import torch
import torch.nn.functional as F
from .features import build_features
from .models import SmallGCN
from .config import DEVICE

def train_private_gcn(data, num_classes: int = 2,
                      epochs: int = 120, lr: float = 1e-2,
                      weight_decay: float = 5e-4, verbose: bool = False):
    """
    Train a small GCN on node labels for this graph.
    If no node labels, use a synthetic task based on degree thresholding.
    We only need a model that maps node features to logits.
    """
    X = build_features(data).to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    N = X.shape[0]

    if getattr(data, "y", None) is not None and data.y is not None and data.y.numel() == N:
        node_labels = data.y.view(-1).long().to(DEVICE)
    else:
        # simple proxy: degree > median vs <= median
        deg = torch.bincount(edge_index[0], minlength=N).float()
        median = deg.median()
        node_labels = (deg > median).long().to(DEVICE)

    model = SmallGCN(in_dim=X.shape[1], out_dim=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # simple train mask: all nodes
    train_mask = torch.ones(N, dtype=torch.bool, device=DEVICE)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X, edge_index)
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 20 == 0:
            pred = logits.argmax(dim=-1)
            acc = (pred == node_labels).float().mean().item()
            print(f"Epoch {ep + 1}/{epochs} loss={loss.item():.4f} acc={acc:.4f}")

    return model, X, edge_index

def make_gbb_api_for_model(model, edge_index, X):
    """
    Build the GBB API closure used by LinkTeller.
    gbb_api(node_ids, X_query) -> logits for those nodes.
    """
    def gbb_api(node_ids, X_query):
        if isinstance(node_ids, (list, tuple)):
            node_ids_t = torch.tensor(node_ids, dtype=torch.long, device=DEVICE)
        else:
            node_ids_t = node_ids.to(DEVICE)
        Xq = X_query.to(DEVICE)
        model.eval()
        with torch.no_grad():
            logits_all = model(Xq, edge_index)
            logits_sel = logits_all[node_ids_t]
        return logits_sel.detach().cpu()
    return gbb_api
