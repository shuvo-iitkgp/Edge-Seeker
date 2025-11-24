import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from torch_geometric.utils import degree
from .config import DEVICE  # or from config import DEVICE if top-level

# @torch.no_grad()
# def linkteller_scores_for_graph(
#     gbb_api,
#     X: torch.Tensor,
#     delta: float = 1e-2,
#     vi_nodes=None,
#     agg: str = "max",
#     use_prob: bool = True,
#     nodes_all=None,
# ) -> Dict[Tuple[int, int], float]:
#     """
#     Compute LinkTeller influence scores for all node pairs.

#     gbb_api: callable (node_ids, X_query) -> logits
#     X: [N, d]
#     returns dict[(i, j)] = score with i < j
#     """
#     N = X.shape[0]
#     if nodes_all is None:
#         nodes_all = torch.arange(N, device=DEVICE)
#     else:
#         nodes_all = torch.tensor(nodes_all, dtype=torch.long, device=DEVICE)

#     if vi_nodes is None:
#         vi_nodes = nodes_all
#     else:
#         vi_nodes = torch.tensor(vi_nodes, dtype=torch.long, device=DEVICE)

#     X_orig = X.clone().to(DEVICE)
#     logits_base = gbb_api(nodes_all, X_orig)
#     if use_prob:
#         probs_base = F.softmax(logits_base, dim=-1)
#         base_scalar = probs_base[:, 1]
#     else:
#         base_scalar = logits_base[:, 1]

#     Sdir = torch.zeros((N, N), dtype=torch.float32)

#     for v in vi_nodes.tolist():
#         x_plus = X_orig.clone()
#         x_minus = X_orig.clone()

#         x_plus[v] = X_orig[v] + delta
#         x_minus[v] = X_orig[v] - delta

#         logits_plus = gbb_api(nodes_all, x_plus)
#         logits_minus = gbb_api(nodes_all, x_minus)

#         if use_prob:
#             probs_plus = F.softmax(logits_plus, dim=-1)
#             probs_minus = F.softmax(logits_minus, dim=-1)
#             scalar_plus = probs_plus[:, 1]
#             scalar_minus = probs_minus[:, 1]
#         else:
#             scalar_plus = logits_plus[:, 1]
#             scalar_minus = logits_minus[:, 1]

#         diff = 0.5 * (scalar_plus - scalar_minus)
#         Sdir[v, :] = diff

#     score_dict: Dict[Tuple[int, int], float] = {}
#     for i in range(N):
#         for j in range(i + 1, N):
#             a = float(Sdir[i, j].item())
#             b = float(Sdir[j, i].item())
#             if agg == "max":
#                 s = max(a, b)
#             elif agg == "mean":
#                 s = 0.5 * (a + b)
#             elif agg == "sum":
#                 s = a + b
#             else:
#                 raise ValueError("agg must be max, mean or sum")
#             score_dict[(i, j)] = s
#     return score_dict

# def mat_to_score_dict(S: torch.Tensor) -> Dict[Tuple[int, int], float]:
#     """
#     Convert NxN score matrix to dict for i < j.
#     """
#     N = S.shape[0]
#     result: Dict[Tuple[int, int], float] = {}
#     for i in range(N):
#         for j in range(i + 1, N):
#             result[(i, j)] = float(S[i, j].item())
#     return result
# linkteller.py

@torch.no_grad()
def linkteller_scores_for_graph(
    gbb_api,
    X: torch.Tensor,
    delta: float = 1e-2,
    vi_nodes=None,
    agg: str = "max",
    use_prob: bool = True,
    nodes_all=None,
    edge_index: torch.Tensor = None,
    norm_by_degree: bool = False,
    degree_power: float = 1.0,
) -> Dict[Tuple[int, int], float]:
    """
    Compute LinkTeller influence scores for all node pairs.

    gbb_api: callable (node_ids, X_query) -> logits
    X: [N, d]

    Maui normalization option:
        If norm_by_degree is True, we divide the influence row for
        perturbed node v by deg[v] ** degree_power.

    Returns dict[(i, j)] = score with i < j.
    """
    N = X.shape[0]
    if nodes_all is None:
        nodes_all = torch.arange(N, device=DEVICE)
    else:
        nodes_all = torch.tensor(nodes_all, dtype=torch.long, device=DEVICE)

    if vi_nodes is None:
        vi_nodes = nodes_all
    else:
        vi_nodes = torch.tensor(vi_nodes, dtype=torch.long, device=DEVICE)

    # degree for Maui normalization
    deg = None
    if norm_by_degree:
        if edge_index is None:
            raise ValueError("edge_index is required for norm_by_degree=True")
        deg = degree(edge_index[0], num_nodes=N).to(DEVICE)
        deg = deg.clamp(min=1.0)  # avoid divide-by-zero

    X_orig = X.clone().to(DEVICE)
    logits_base = gbb_api(nodes_all, X_orig)
    if use_prob:
        probs_base = F.softmax(logits_base, dim=-1)
        base_scalar = probs_base[:, 1]
    else:
        base_scalar = logits_base[:, 1]

    Sdir = torch.zeros((N, N), dtype=torch.float32)

    for v in vi_nodes.tolist():
        x_plus = X_orig.clone()
        x_minus = X_orig.clone()

        x_plus[v] = X_orig[v] + delta
        x_minus[v] = X_orig[v] - delta

        logits_plus = gbb_api(nodes_all, x_plus)
        logits_minus = gbb_api(nodes_all, x_minus)

        if use_prob:
            probs_plus = F.softmax(logits_plus, dim=-1)
            probs_minus = F.softmax(logits_minus, dim=-1)
            scalar_plus = probs_plus[:, 1]
            scalar_minus = probs_minus[:, 1]
        else:
            scalar_plus = logits_plus[:, 1]
            scalar_minus = logits_minus[:, 1]

        diff = 0.5 * (scalar_plus - scalar_minus)

        # Maui Normalization: normalize influence by source degree
        if norm_by_degree:
            scale = deg[v] ** degree_power
            diff = diff / scale

        Sdir[v, :] = diff.cpu()

    score_dict: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            a = float(Sdir[i, j].item())
            b = float(Sdir[j, i].item())
            if agg == "max":
                s = max(a, b)
            elif agg == "mean":
                s = 0.5 * (a + b)
            elif agg == "sum":
                s = a + b
            else:
                raise ValueError("agg must be max, mean or sum")
            score_dict[(i, j)] = s
    return score_dict
