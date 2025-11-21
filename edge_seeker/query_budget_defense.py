# edge_seeker/query_budget_defense.py

from typing import Dict, Tuple, List

import torch
import torch.distributions as D

from .config import DEVICE, DEFAULT_DELTA, DEFAULT_AGG, DEFAULT_USE_PROB
from .training import train_private_gcn, make_gbb_api_for_model
from .metrics import (
    undirected_edge_set,
    top_pairs_from_score_dict,
    prf1,
    auroc_from_scores,
)

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


def run_query_budget_defense_for_graph(
    data,
    model,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    base_gbb_api,
    sigma_mid: float,
    delta_list: List[float],
    query_fracs: List[float],
    noise_type: str = "gaussian",
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    delta_default: float = DEFAULT_DELTA,
    graph_id: int = 0,
) -> List[Dict]:
    """
    For a single graph:
      - Fix noise at sigma_mid
      - Vary query budget q by subsampling vi_nodes
      - Sweep delta and compute attack metrics
    Returns a list of dict rows.
    """
    N = data.num_nodes
    nodes_all = torch.arange(N, device=DEVICE)
    true_pairs = undirected_edge_set(edge_index, num_nodes=N)
    num_edges = len(true_pairs)

    results: List[Dict] = []

    # Wrap gbb with fixed mid sigma defense
    defended_gbb = make_noisy_gbb_api(
        base_gbb_api=base_gbb_api,
        sigma=sigma_mid,
        noise_type=noise_type,
    )

    # Fixed random order of nodes per graph so budgets are nested
    node_order = torch.randperm(N, device=DEVICE)

    for q in query_fracs:
        budget_size = max(1, int(round(q * N)))
        vi_nodes = node_order[:budget_size].cpu().tolist()
        query_pct = int(round(q * 100))

        for delta in delta_list:
            print(
                f"[graph {graph_id}] noise={noise_type} "
                f"sigma={sigma_mid:.3f} q={query_pct}% delta={delta:g}"
            )

            score_dict = None
            # Call LinkTeller with given vi_nodes and delta
            score_dict = __run_linkteller_with_params(
                gbb_api=defended_gbb,
                X=X,
                delta=delta,
                vi_nodes=vi_nodes,
                nodes_all=nodes_all,
                agg=agg,
                use_prob=use_prob,
            )

            attack_m = compute_attack_metrics(score_dict, true_pairs)

            row = {
                "graph_id": graph_id,
                "num_nodes": int(N),
                "num_edges": int(num_edges),
                "noise_type": noise_type,
                "sigma": float(sigma_mid),
                "query_frac": float(q),
                "query_pct": query_pct,
                "delta": float(delta),
                "attack_auroc": attack_m["auroc"],
                "attack_f1": attack_m["f1"],
                "attack_edge_recall": attack_m["edge_recall"],
            }
            results.append(row)

    return results


def __run_linkteller_with_params(
    gbb_api,
    X: torch.Tensor,
    delta: float,
    vi_nodes,
    nodes_all,
    agg: str,
    use_prob: bool,
) -> Dict[Pair, float]:
    """
    Thin wrapper around linkteller_scores_for_graph so this module stays focused.
    """
    from .linkteller import linkteller_scores_for_graph

    score_dict = linkteller_scores_for_graph(
        gbb_api=gbb_api,
        X=X,
        delta=delta,
        vi_nodes=vi_nodes,
        agg=agg,
        use_prob=use_prob,
        nodes_all=nodes_all,
    )
    return score_dict


def run_query_budget_defense_for_dataset(
    dataset,
    sigma_mid: float,
    delta_list: List[float],
    query_fracs: List[float],
    noise_type: str = "gaussian",
    train_epochs: int = 120,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    verbose: bool = False,
) -> List[Dict]:
    """
    Run the query budget + defense experiment for every graph in the dataset.

    For each graph:
      - Train a private GCN with train_private_gcn
      - Build gbb_api with make_gbb_api_for_model
      - Run run_query_budget_defense_for_graph
    """
    all_rows: List[Dict] = []

    for idx, data in enumerate(dataset):
        print(f"Training GCN on graph {idx} with {data.num_nodes} nodes")

        # Assumes train_private_gcn returns (model, X, edge_index)
        model, X, edge_index = train_private_gcn(
            data,
            num_classes=2,
            epochs=train_epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=verbose,
        )

        base_gbb = make_gbb_api_for_model(model, edge_index, X)

        rows = run_query_budget_defense_for_graph(
            data=data,
            model=model,
            X=X,
            edge_index=edge_index,
            base_gbb_api=base_gbb,
            sigma_mid=sigma_mid,
            delta_list=delta_list,
            query_fracs=query_fracs,
            noise_type=noise_type,
            agg=agg,
            use_prob=use_prob,
            delta_default=DEFAULT_DELTA,
            graph_id=idx,
        )

        all_rows.extend(rows)

    return all_rows
