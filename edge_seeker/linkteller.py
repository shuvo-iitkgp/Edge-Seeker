import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Iterable, Optional
from torch_geometric.utils import degree
from .config import DEVICE  # or from config import DEVICE if top-level

Pair = Tuple[int, int]

@torch.no_grad()
def linkteller_scores_for_hetero(
    gbb_api,
    x_dict: Dict[str, torch.Tensor],
    target_ntype: str,
    delta: float = 1e-2,
    vi_nodes: Optional[Iterable[int]] = None,
    agg: str = "max",
    use_prob: bool = True,
    nodes_all: Optional[Iterable[int]] = None,
    rel_mask: Optional[Iterable[Tuple[str, str, str]]] = None,
) -> Dict[Pair, float]:
    """
    LinkTeller for a heterogeneous graph with relation masking.

    Arguments
    ----------
    gbb_api:
        Callable with signature
            gbb_api(
                target_ntype: str,
                node_ids: Tensor,
                x_dict_query: Dict[str, Tensor],
                rel_mask: Optional[Iterable[EdgeType]]
            ) -> logits[target_ntype][node_ids]
        EdgeType is a tuple (src_type, rel_name, dst_type).

    x_dict:
        Dict[str, Tensor] of node features per node type.
        At minimum must contain x_dict[target_ntype] of shape [N_t, d].

    target_ntype:
        Node type we attack, for example "paper" in OGB-MAG.

    delta:
        Finite difference step.

    vi_nodes:
        Node indices (in the target_ntype index space) whose features we poke.
        If None uses all nodes of target_ntype.

    agg:
        How to symmetrize directed influences: "max", "mean" or "sum".

    use_prob:
        If True uses softmax probability of class 1, else raw logits[:, 1].

    nodes_all:
        Subset of target_ntype nodes whose predictions we read.
        If None uses all nodes of target_ntype.

    rel_mask:
        Optional iterable of edge types to keep during the attack.
        Passed through to gbb_api. If None gbb_api should use all relations.

    Returns
    -------
    score_dict:
        Dict[(i, j)] -> influence score between nodes i and j of target_ntype
        with i < j, using the same semantics as linkteller_scores_for_graph.
    """
    x_target = x_dict[target_ntype].to(DEVICE)
    N = x_target.size(0)

    if nodes_all is None:
        nodes_all_t = torch.arange(N, device=DEVICE)
    else:
        nodes_all_t = torch.as_tensor(nodes_all, dtype=torch.long, device=DEVICE)

    if vi_nodes is None:
        vi_nodes_t = nodes_all_t
    else:
        vi_nodes_t = torch.as_tensor(vi_nodes, dtype=torch.long, device=DEVICE)

    # Clone features dict once on device
    x_dict_base = {k: v.to(DEVICE).clone() for k, v in x_dict.items()}

    # Base logits, mainly to mirror the homogeneous implementation
    logits_base = gbb_api(
        target_ntype=target_ntype,
        node_ids=nodes_all_t,
        x_dict_query=x_dict_base,
        rel_mask=rel_mask,
    )
    if use_prob:
        probs_base = F.softmax(logits_base, dim=-1)
        base_scalar = probs_base[:, 1]
    else:
        base_scalar = logits_base[:, 1]
    _ = base_scalar  # kept for parity with homogeneous code

    Sdir = torch.zeros((N, N), dtype=torch.float32)

    for v in vi_nodes_t.tolist():
        # Fresh copies per perturbation
        x_plus = {k: v.clone() for k, v in x_dict_base.items()}
        x_minus = {k: v.clone() for k, v in x_dict_base.items()}

        x_plus[target_ntype][v] = x_dict_base[target_ntype][v] + delta
        x_minus[target_ntype][v] = x_dict_base[target_ntype][v] - delta

        logits_plus = gbb_api(
            target_ntype=target_ntype,
            node_ids=nodes_all_t,
            x_dict_query=x_plus,
            rel_mask=rel_mask,
        )
        logits_minus = gbb_api(
            target_ntype=target_ntype,
            node_ids=nodes_all_t,
            x_dict_query=x_minus,
            rel_mask=rel_mask,
        )

        if use_prob:
            probs_plus = F.softmax(logits_plus, dim=-1)
            probs_minus = F.softmax(logits_minus, dim=-1)
            scalar_plus = probs_plus[:, 1]
            scalar_minus = probs_minus[:, 1]
        else:
            scalar_plus = logits_plus[:, 1]
            scalar_minus = logits_minus[:, 1]

        diff = 0.5 * (scalar_plus - scalar_minus)  # effect of poking node v
        Sdir[v, :] = diff

    # Symmetrize into edge scores
    score_dict: Dict[Pair, float] = {}
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


@torch.no_grad()
def linkteller_scores_for_graph(
    gbb_api,
    X: torch.Tensor,
    delta: float = 1e-2,
    vi_nodes=None,
    agg: str = "max",
    use_prob: bool = True,
    nodes_all=None,
) -> Dict[Tuple[int, int], float]:
    """
    Compute LinkTeller influence scores for all node pairs.

    gbb_api: callable (node_ids, X_query) -> logits
    X: [N, d]
    returns dict[(i, j)] = score with i < j
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
        Sdir[v, :] = diff

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

def mat_to_score_dict(S: torch.Tensor) -> Dict[Tuple[int, int], float]:
    """
    Convert NxN score matrix to dict for i < j.
    """
    N = S.shape[0]
    result: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            result[(i, j)] = float(S[i, j].item())
    return result
    