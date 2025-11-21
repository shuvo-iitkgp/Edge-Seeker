# training.py

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple
from .features import build_features
from .models import SmallGCN, DeepGCN
from .config import DEVICE


def train_private_gcn(data, num_classes: int = 2,
                      epochs: int = 120, lr: float = 1e-2,
                      weight_decay: float = 5e-4, verbose: bool = False):
    """
    Original helper. Kept for backward compat.
    """
    X = build_features(data).to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    N = X.shape[0]

    if getattr(data, "y", None) is not None and data.y is not None and data.y.numel() == N:
        node_labels = data.y.view(-1).long().to(DEVICE)
    else:
        deg = torch.bincount(edge_index[0], minlength=N).float()
        median = deg.median()
        node_labels = (deg > median).long().to(DEVICE)

    model = SmallGCN(in_dim=X.shape[1], out_dim=num_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

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


def train_gcn_with_depth(
    data,
    num_layers: int,
    num_classes: int = 2,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    patience: int = 30,
    verbose: bool = False,
) -> Tuple[DeepGCN, Dict[str, Any]]:
    """
    Train DeepGCN with given depth and simple node split.
    Uses early stopping on validation to keep accuracies in the same ballpark
    across different depths.
    """
    X = build_features(data).to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    N = X.shape[0]

    # labels
    if getattr(data, "y", None) is not None and data.y is not None and data.y.numel() == N:
        y = data.y.view(-1).long().to(DEVICE)
    else:
        deg = torch.bincount(edge_index[0], minlength=N).float()
        median = deg.median()
        y = (deg > median).long().to(DEVICE)

    # simple random split
    perm = torch.randperm(N, device=DEVICE)
    n_val = int(val_ratio * N)
    n_test = int(test_ratio * N)
    n_train = N - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)
    test_mask = torch.zeros(N, dtype=torch.bool, device=DEVICE)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    model = DeepGCN(
        in_dim=X.shape[1],
        hid_dim=64,
        out_dim=num_classes,
        num_layers=num_layers,
        dropout=0.3,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    history = {
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    for ep in range(1, epochs + 1):
        # train
        model.train()
        opt.zero_grad()
        logits = model(X, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            logits_eval = model(X, edge_index)
            pred = logits_eval.argmax(dim=-1)

            train_loss = loss.item()
            val_loss = F.cross_entropy(
                logits_eval[val_mask], y[val_mask]
            ).item()

            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)

        if verbose and ep % 20 == 0:
            print(
                f"[L={num_layers}] Epoch {ep:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} test_acc={test_acc:.3f}"
            )

        # early stopping on val
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {
                "model": model.state_dict(),
                "epoch": ep,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"[L={num_layers}] Early stopping at epoch {ep}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    final_stats = {
        "best_val_acc": best_val_acc,
        "final_test_acc": history["test_acc"][-1],
        "epoch_trained": len(history["train_loss"]),
        "history": history,
        "train_size": int(train_mask.sum().item()),
        "val_size": int(val_mask.sum().item()),
        "test_size": int(test_mask.sum().item()),
    }

    return model, final_stats


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
