# edge_seeker/noise_defense.py

from typing import Dict, Tuple, List

import torch
import torch.distributions as D

from .config import DEVICE, DEFAULT_DELTA, DEFAULT_AGG, DEFAULT_USE_PROB
from .metrics import (
    undirected_edge_set,
    top_pairs_from_score_dict,
    prf1,
    auroc_from_scores,
)
from .linkteller import linkteller_scores_for_graph
from .training import train_private_gcn, make_gbb_api_for_model


Pair = Tuple[int, int]


def make_noisy_gbb_api(base_gbb_api, sigma: float, noise_type: str = "gaussian"):
    """
    Wrap the original gbb_api to inject noise into logits.

    base_gbb_api: callable(node_ids, X_query) -> logits on CPU.
    sigma: noise scale.
    noise_type: 'gaussian' or 'laplace'.
    """
    assert noise_type in {"gaussian", "laplace"}

    def noisy_gbb_api(node_ids, X_query):
        logits = base_gbb_api(node_ids, X_query)  # CPU tensor [B, C]
        if sigma <= 0.0:
            return logits

        if noise_type == "gaussian":
            noise = torch.randn_like(logits) * sigma
        else:
            dist = D.Laplace(
                loc=torch.zeros_like(logits),
                scale=torch.full_like(logits, sigma),
            )
            noise = dist.sample()

        return logits + noise

    return noisy_gbb_api


def build_node_labels_for_graph(data) -> torch.Tensor:
    """
    Rebuild node labels in the same way as train_private_gcn.

    If data.y is present and node level, use that.
    Otherwise use high degree vs low degree as 2 class label.
    """
    edge_index = data.edge_index.to(DEVICE)
    N = data.num_nodes

    if getattr(data, "y", None) is not None and data.y is not None and data.y.numel() == N:
        node_labels = data.y.view(-1).long().to(DEVICE)
    else:
        deg = torch.bincount(edge_index[0], minlength=N).float()
        median = deg.median()
        node_labels = (deg > median).long().to(DEVICE)

    return node_labels


@torch.no_grad()
def evaluate_node_accuracy_with_logit_noise(
    model,
    data,
    sigma: float = 0.0,
    noise_type: str = "gaussian",
) -> float:
    """
    Measure how much the defense hurts the downstream node task.
    We add noise to the logits and compute accuracy on all nodes.
    """
    from .features import build_features  # avoid circular imports

    model.eval()

    X = build_features(data).to(DEVICE)
    edge_index = data.edge_index.to(DEVICE)
    y = build_node_labels_for_graph(data)

    logits = model(X, edge_index)

    if sigma > 0.0:
        if noise_type == "gaussian":
            logits = logits + torch.randn_like(logits) * sigma
        else:
            dist = D.Laplace(
                loc=torch.zeros_like(logits),
                scale=torch.full_like(logits, sigma),
            )
            logits = logits + dist.sample()

    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    return acc


def compute_attack_metrics(
    score_dict: Dict[Pair, float],
    true_pairs: "set[Pair]",
) -> Dict[str, float]:
    """
    Compute AUROC, F1, and edge recall at K, where K is number of true edges.
    """
    num_true = len(true_pairs)
    if num_true == 0:
        return {"auroc": float("nan"), "f1": 0.0, "edge_recall": 0.0}

    top_k = num_true
    pred_pairs = top_pairs_from_score_dict(score_dict, top_k=top_k)
    precision, recall, f1 = prf1(pred_pairs, true_pairs)

    edge_recall = recall
    auroc = auroc_from_scores(score_dict, true_pairs)

    return {
        "auroc": float(auroc),
        "f1": float(f1),
        "edge_recall": float(edge_recall),
    }


def run_noise_defense_for_graph(
    data,
    model,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    base_gbb_api,
    sigmas: List[float],
    noise_types: List[str],
    delta: float = DEFAULT_DELTA,
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    graph_id: int = 0,
) -> List[Dict]:
    """
    Run LinkTeller + noise defense on a single graph for all sigmas and noise types.
    Returns a list of metric dicts, one per (noise_type, sigma).
    """
    N = data.num_nodes
    true_pairs = undirected_edge_set(edge_index, num_nodes=N)
    num_edges = len(true_pairs)

    results = []

    for noise_type in noise_types:
        for sigma in sigmas:
            print(f"[graph {graph_id}] noise={noise_type} sigma={sigma:.3f}")

            noisy_gbb = make_noisy_gbb_api(
                base_gbb_api=base_gbb_api,
                sigma=sigma,
                noise_type=noise_type,
            )

            score_dict = linkteller_scores_for_graph(
                gbb_api=noisy_gbb,
                X=X,
                delta=delta,
                agg=agg,
                use_prob=use_prob,
            )

            attack_m = compute_attack_metrics(score_dict, true_pairs)
            model_acc = evaluate_node_accuracy_with_logit_noise(
                model=model,
                data=data,
                sigma=sigma,
                noise_type=noise_type,
            )

            row = {
                "graph_id": graph_id,
                "num_nodes": int(N),
                "num_edges": int(num_edges),
                "noise_type": noise_type,
                "sigma": float(sigma),
                "attack_auroc": attack_m["auroc"],
                "attack_f1": attack_m["f1"],
                "attack_edge_recall": attack_m["edge_recall"],
                "model_acc": float(model_acc),
            }
            results.append(row)

    return results


def run_noise_defense_for_dataset(
    dataset,
    sigmas: List[float],
    noise_types: List[str],
    train_epochs: int,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    delta: float = DEFAULT_DELTA,
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    verbose: bool = False,
) -> List[Dict]:
    """
    Run the full noise defense sweep on every graph in the dataset.
    One GCN per graph, same hyperparameters.
    """
    all_results: List[Dict] = []

    for idx, data in enumerate(dataset):
        print(f"Training GCN on graph {idx} with {data.num_nodes} nodes")

        model, X, edge_index = train_private_gcn(
            data,
            num_classes=2,
            epochs=train_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
        )

        base_gbb = make_gbb_api_for_model(model, edge_index, X)

        graph_results = run_noise_defense_for_graph(
            data=data,
            model=model,
            X=X,
            edge_index=edge_index,
            base_gbb_api=base_gbb,
            sigmas=sigmas,
            noise_types=noise_types,
            delta=delta,
            agg=agg,
            use_prob=use_prob,
            graph_id=idx,
        )

        all_results.extend(graph_results)

    return all_results
