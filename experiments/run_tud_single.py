import argparse
import os
import pandas as pd
from tqdm import tqdm

from edge_seeker import (
    set_seed,
    DEVICE,
    load_tud_dataset,
    build_features,
    SmallGCN,
    train_private_gcn,
    make_gbb_api_for_model,
    linkteller_scores_for_graph,
    adjacency_matrix_from_edge_index,
    common_neighbors_score,
    adamic_adar_score,
    resource_allocation_score,
    node2vec_scores,
    undirected_edge_set,
    top_pairs_from_score_dict,
    prf1,
    auroc_from_scores,
)
from edge_seeker.config import (
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_DELTA,
    DEFAULT_AGG,
    DEFAULT_USE_PROB,
    DEFAULT_NODE2VEC_EPOCHS,
)

def run_dataset(name: str, out_csv: str, limit_graphs: int = None):
    set_seed()
    dataset = load_tud_dataset(name)
    print(f"{name} graphs: {len(dataset)}")

    records = []

    for gid, data in enumerate(tqdm(dataset, desc=f"{name}")):
        if limit_graphs is not None and gid >= limit_graphs:
            break

        data = data.clone()
        N = data.num_nodes
        if N < 4:
            continue

        # Train small GCN
        model, X_model, edge_index_model = train_private_gcn(
            data,
            epochs=DEFAULT_TRAIN_EPOCHS,
            verbose=False,
        )
        gbb_api = make_gbb_api_for_model(model, edge_index_model, X_model)

        # True edge set
        true_pairs = undirected_edge_set(data.edge_index, num_nodes=N)
        m_hat = len(true_pairs)

        # LinkTeller
        lt_scores = linkteller_scores_for_graph(
            gbb_api,
            X_model,
            delta=DEFAULT_DELTA,
            agg=DEFAULT_AGG,
            use_prob=DEFAULT_USE_PROB,
        )

        # Baselines on adjacency
        A = adjacency_matrix_from_edge_index(data.edge_index, N, device=DEVICE)
        cn_S = common_neighbors_score(A)
        aa_S = adamic_adar_score(A)
        ra_S = resource_allocation_score(A)

        cn_scores = {k: float(v) for k, v in
                     {**{(i, j): float(cn_S[i, j]) for i in range(N) for j in range(i + 1, N)}}.items()}
        aa_scores = {k: float(v) for k, v in
                     {**{(i, j): float(aa_S[i, j]) for i in range(N) for j in range(i + 1, N)}}.items()}
        ra_scores = {k: float(v) for k, v in
                     {**{(i, j): float(ra_S[i, j]) for i in range(N) for j in range(i + 1, N)}}.items()}

        # Node2Vec
        try:
            emb_S = node2vec_scores(data.edge_index, N,
                                    emb_dim=64,
                                    epochs=DEFAULT_NODE2VEC_EPOCHS,
                                    device=DEVICE)
            emb_scores = { (i, j): float(emb_S[i, j])
                           for i in range(N) for j in range(i + 1, N) }
        except Exception:
            emb_scores = None

        methods = {
            "LinkTeller": lt_scores,
            "CN": cn_scores,
            "AA": aa_scores,
            "RA": ra_scores,
        }
        if emb_scores is not None:
            methods["Node2Vec"] = emb_scores

        for method_name, scores in methods.items():
            pred_pairs = top_pairs_from_score_dict(scores, m_hat)
            p, r, f1 = prf1(pred_pairs, true_pairs)
            au = auroc_from_scores(scores, true_pairs)

            records.append({
                "dataset": name,
                "graph_id": gid,
                "n_nodes": N,
                "n_edges_true": m_hat,
                "method": method_name,
                "precision": p,
                "recall": r,
                "f1": f1,
                "auroc": au,
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--out", type=str, default="results/linkteller_tud_results.csv")
    parser.add_argument("--limit_graphs", type=int, default=None)
    args = parser.parse_args()
    run_dataset(args.dataset, args.out, args.limit_graphs)

if __name__ == "__main__":
    main()
