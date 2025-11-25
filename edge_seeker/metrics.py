import random
from typing import Dict, Tuple, Set, Iterable
import torch
from torch_geometric.utils import to_undirected, remove_self_loops
from sklearn.metrics import roc_auc_score

Pair = Tuple[int, int]

def per_relation_f1_report(
    score_dict: Dict[Pair, float],
    relation_edge_index_dict: Dict[str, torch.Tensor],
    num_nodes: int,
    top_k_mode: str = "num_edges",
    top_k_factor: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compute F1 for each relation using a single LinkTeller score_dict.

    Arguments
    ----------
    score_dict:
        Dict[(i, j)] -> score on a single node set of size num_nodes.
        Typically output of linkteller_scores_for_graph or linkteller_scores_for_hetero
        restricted to a single node type.

    relation_edge_index_dict:
        Dict[relation_name] -> edge_index for that relation in the same node
        index space of size num_nodes. For example for OGB-MAG paper-paper
        relations after reindexing papers to [0, N_paper).

    num_nodes:
        Number of nodes in the target node set.

    top_k_mode:
        "num_edges"  use top_k equal to number of true edges per relation.
        "factor"     use top_k = top_k_factor * |E_rel|.

    top_k_factor:
        Factor used when top_k_mode == "factor".

    Returns
    -------
    report:
        Dict[relation_name] -> {
            "precision": ...,
            "recall": ...,
            "f1": ...,
            "num_edges": ...,
            "top_k": ...
        }
    """
    report: Dict[str, Dict[str, float]] = {}

    for rel_name, edge_index in relation_edge_index_dict.items():
        true_pairs = undirected_edge_set(edge_index, num_nodes=num_nodes)
        m = len(true_pairs)
        if m == 0:
            continue

        if top_k_mode == "num_edges":
            top_k = m
        elif top_k_mode == "factor":
            top_k = max(1, int(top_k_factor * m))
        else:
            raise ValueError("top_k_mode must be 'num_edges' or 'factor'")

        pred_pairs = top_pairs_from_score_dict(score_dict, top_k=top_k)
        precision, recall, f1 = prf1(pred_pairs, true_pairs)

        report[rel_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "num_edges": float(m),
            "top_k": float(top_k),
        }

    return report


def undirected_edge_set(edge_index, num_nodes: int) -> Set[Pair]:
    ei, _ = remove_self_loops(edge_index)
    ei = to_undirected(ei, num_nodes=num_nodes)
    i, j = ei[0].cpu().tolist(), ei[1].cpu().tolist()
    pairs = set()
    for u, v in zip(i, j):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.add((a, b))
    return pairs

def top_pairs_from_score_dict(score_dict: Dict[Pair, float], top_k: int) -> Set[Pair]:
    items = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
    top = [ij for (ij, _) in items[:min(top_k, len(items))]]
    return set(top)

def prf1(pred_pairs: Set[Pair], true_pairs: Set[Pair]):
    tp = len(pred_pairs & true_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def auroc_from_scores(score_dict: Dict[Pair, float], true_pairs: Set[Pair],
                      num_neg: int = None) -> float:
    """
    Simple AUROC estimator. Use all positives and sample negatives.
    """
    pos_pairs = list(true_pairs)
    pos_scores = [score_dict.get(p, 0.0) for p in pos_pairs]

    all_pairs = list(score_dict.keys())
    neg_candidates = [p for p in all_pairs if p not in true_pairs]
    if num_neg is None:
        num_neg = len(pos_pairs)
    if len(neg_candidates) == 0:
        return float("nan")
    neg_samples = random.sample(neg_candidates, min(num_neg, len(neg_candidates)))
    neg_scores = [score_dict.get(p, 0.0) for p in neg_samples]

    y_true = [1] * len(pos_scores) + [0] * len(neg_scores)
    y_scores = pos_scores + neg_scores

    return roc_auc_score(y_true, y_scores)
