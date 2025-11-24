# edge_seeker/adaptive_delta.py

from typing import Dict, Tuple, List, Set

import numpy as np
import torch

from .config import DEVICE, DEFAULT_AGG, DEFAULT_USE_PROB, DEFAULT_DELTA  # :contentReference[oaicite:1]{index=1}
from .metrics import undirected_edge_set, prf1, auroc_from_scores  # :contentReference[oaicite:2]{index=2}
from .linkteller import linkteller_scores_for_graph  # :contentReference[oaicite:3]{index=3}
from .training import train_private_gcn, make_gbb_api_for_model  # :contentReference[oaicite:4]{index=4}


Pair = Tuple[int, int]


def build_pair_labels_and_scores(
    score_dict: Dict[Pair, float],
    true_pairs: Set[Pair],
):
    """
    Turn score_dict + true edge set into arrays suitable for threshold calibration.

    y_true[k] in {0,1}, y_score[k] is continuous.
    """
    pairs = list(score_dict.keys())
    y_true = np.array([1 if p in true_pairs else 0 for p in pairs], dtype=np.int32)
    y_score = np.array([score_dict[p] for p in pairs], dtype=np.float32)
    return pairs, y_true, y_score


def calibrate_best_threshold_f1(
    score_dict: Dict[Pair, float],
    true_pairs: Set[Pair],
):
    """
    Gradient free threshold calibration.

    Sweep all unique scores as candidate thresholds.
    Pick threshold that maximizes F1 for edge recovery.
    """
    if len(score_dict) == 0 or len(true_pairs) == 0:
        return {
            "threshold": 0.0,
            "best_f1": 0.0,
            "best_precision": 0.0,
            "best_recall": 0.0,
        }

    pairs, y_true, y_score = build_pair_labels_and_scores(score_dict, true_pairs)

    # sort by decreasing score
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]
    pairs_sorted = [pairs[i] for i in order]

    # cumulative counts as we move threshold from +inf down to -inf
    tp = 0
    fp = 0
    fn = int((y_true_sorted == 1).sum())

    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0
    best_thr = float(y_score_sorted[0])

    last_score = None

    for idx, (yt, ys) in enumerate(zip(y_true_sorted, y_score_sorted)):
        if yt == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        # update only when score changes so we consider threshold at distinct values
        if last_score is None or ys != last_score:
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_prec = precision
                best_rec = recall
                best_thr = ys

        last_score = ys

    return {
        "threshold": float(best_thr),
        "best_f1": float(best_f1),
        "best_precision": float(best_prec),
        "best_recall": float(best_rec),
    }


def evaluate_scores_with_threshold(
    score_dict: Dict[Pair, float],
    true_pairs: Set[Pair],
    threshold: float,
):
    """
    Compute precision, recall, F1 at a fixed threshold on scores.
    """
    pred_pairs = {p for p, s in score_dict.items() if s >= threshold}
    precision, recall, f1 = prf1(pred_pairs, true_pairs)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_pred_pairs": int(len(pred_pairs)),
    }


def run_linkteller_for_graph_with_delta(
    data,
    base_gbb_api,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    delta: float,
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    graph_id: int = 0,
):
    """
    Helper to run LinkTeller on a single graph at a given delta
    and compute AUROC + best F1 via threshold calibration.
    """
    N = data.num_nodes
    true_pairs = undirected_edge_set(edge_index, num_nodes=N)
    num_edges = len(true_pairs)

    score_dict = linkteller_scores_for_graph(
        gbb_api=base_gbb_api,
        X=X,
        delta=delta,
        agg=agg,
        use_prob=use_prob,
    )

    # AUROC uses your existing helper
    auroc = auroc_from_scores(score_dict, true_pairs)

    # gradient free threshold search on per pair scores
    thr_stats = calibrate_best_threshold_f1(score_dict, true_pairs)

    row = {
        "graph_id": graph_id,
        "num_nodes": int(N),
        "num_edges": int(num_edges),
        "delta": float(delta),
        "attack_auroc": float(auroc),
        "attack_f1": thr_stats["best_f1"],
        "attack_precision": thr_stats["best_precision"],
        "attack_recall": thr_stats["best_recall"],
        "threshold": thr_stats["threshold"],
    }
    return row, score_dict


def gradient_free_delta_search_for_graph(
    data,
    base_gbb_api,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    delta_grid: List[float],
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    graph_id: int = 0,
):
    """
    Gradient free search over delta values on a single graph.

    For each delta:
      - run LinkTeller
      - calibrate best threshold on scores
      - record AUROC and calibrated F1

    Returns:
      best_row: row dict for best delta by F1
      all_rows: list of all row dicts for full grid
    """
    all_rows = []
    best_row = None
    best_f1 = -1.0

    for delta in delta_grid:
        print(f"[graph {graph_id}] searching delta={delta:g}")
        row, _ = run_linkteller_for_graph_with_delta(
            data=data,
            base_gbb_api=base_gbb_api,
            X=X,
            edge_index=edge_index,
            delta=delta,
            agg=agg,
            use_prob=use_prob,
            graph_id=graph_id,
        )
        all_rows.append(row)
        if row["attack_f1"] > best_f1:
            best_f1 = row["attack_f1"]
            best_row = row

    return best_row, all_rows


def baseline_fixed_delta_for_graph(
    data,
    base_gbb_api,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    delta: float = DEFAULT_DELTA,
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    graph_id: int = 0,
):
    """
    Baseline LinkTeller with fixed global delta and default calibration.
    Uses the same threshold calibration routine so comparison is fair.
    """
    row, _ = run_linkteller_for_graph_with_delta(
        data=data,
        base_gbb_api=base_gbb_api,
        X=X,
        edge_index=edge_index,
        delta=delta,
        agg=agg,
        use_prob=use_prob,
        graph_id=graph_id,
    )
    row["mode"] = "baseline_fixed_delta"
    return row


def adaptive_delta_for_graph(
    data,
    base_gbb_api,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    delta_grid: List[float],
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    graph_id: int = 0,
):
    """
    Run the gradient free search and return the best row.
    """
    best_row, all_rows = gradient_free_delta_search_for_graph(
        data=data,
        base_gbb_api=base_gbb_api,
        X=X,
        edge_index=edge_index,
        delta_grid=delta_grid,
        agg=agg,
        use_prob=use_prob,
        graph_id=graph_id,
    )
    best_row["mode"] = "adaptive_delta"
    return best_row, all_rows


def run_adaptive_delta_for_dataset(
    dataset,
    delta_grid: List[float],
    train_epochs: int,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    agg: str = DEFAULT_AGG,
    use_prob: bool = DEFAULT_USE_PROB,
    verbose: bool = False,
):
    """
    Full pipeline on a TU dataset.

    For each graph:
      - train private GCN
      - build gbb_api
      - run baseline fixed delta
      - run adaptive delta grid search
    """
    all_rows = []

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

        # baseline
        baseline_row = baseline_fixed_delta_for_graph(
            data=data,
            base_gbb_api=base_gbb,
            X=X,
            edge_index=edge_index,
            delta=DEFAULT_DELTA,
            agg=agg,
            use_prob=use_prob,
            graph_id=idx,
        )
        all_rows.append(baseline_row)

        # adaptive search
        best_row, all_search_rows = adaptive_delta_for_graph(
            data=data,
            base_gbb_api=base_gbb,
            X=X,
            edge_index=edge_index,
            delta_grid=delta_grid,
            agg=agg,
            use_prob=use_prob,
            graph_id=idx,
        )
        all_rows.append(best_row)

        # optionally you can also log the intermediate grid rows
        for r in all_search_rows:
            r2 = dict(r)
            r2["mode"] = "search_grid"
            all_rows.append(r2)

    return all_rows
