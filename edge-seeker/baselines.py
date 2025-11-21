import torch
from torch_geometric.nn.models import Node2Vec

def adjacency_matrix_from_edge_index(edge_index, N: int, device="cpu") -> torch.Tensor:
    A = torch.zeros((N, N), dtype=torch.float32, device=device)
    i, j = edge_index[0].long(), edge_index[1].long()
    A[i, j] = 1.0
    A[j, i] = 1.0
    A.fill_diagonal_(0.0)
    return A

def common_neighbors_score(A: torch.Tensor) -> torch.Tensor:
    S = A @ A
    S.fill_diagonal_(0.0)
    return S

def adamic_adar_score(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(dim=1)
    inv_log_deg = torch.zeros_like(deg)
    mask = deg > 1
    inv_log_deg[mask] = 1.0 / torch.log(deg[mask])
    W = torch.diag(inv_log_deg)
    S = A @ W @ A
    S.fill_diagonal_(0.0)
    return S

def resource_allocation_score(A: torch.Tensor) -> torch.Tensor:
    deg = A.sum(dim=1)
    inv_deg = torch.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    W = torch.diag(inv_deg)
    S = A @ W @ A
    S.fill_diagonal_(0.0)
    return S

def node2vec_scores(edge_index, num_nodes: int, emb_dim: int = 64,
                    epochs: int = 80, device="cpu", batch_size: int = 128) -> torch.Tensor:
    """
    Node2Vec similarity matrix. Handles tiny graphs gracefully.
    """
    if edge_index.numel() == 0 or edge_index.size(1) < 2 or num_nodes < 3:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=emb_dim,
        walk_length=10,
        context_size=5,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        sparse=True,
    ).to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    try:
        for _ in range(epochs):
            model.train()
            for pos_rw, neg_rw in loader:
                opt.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                opt.step()
    except Exception:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        if hasattr(model, "embedding"):
            Z = model.embedding.weight.detach().cpu()
        elif hasattr(model, "embeddings"):
            Z = model.embeddings.weight.detach().cpu()
        else:
            Z = None
            for _, param in model.named_parameters():
                if param.ndim == 2 and param.shape[0] == num_nodes:
                    Z = param.detach().cpu()
                    break
        if Z is None:
            return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    S = Z @ Z.T
    S.fill_diagonal_(0.0)
    return S
