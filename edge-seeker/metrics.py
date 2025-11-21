import random
from typing import Dict, Tuple, Set, Iterable
import torch
from torch_geometric.utils import to_undirected, remove_self_loops
from sklearn.metrics import roc_auc_score

Pair = Tuple[int, int]

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
