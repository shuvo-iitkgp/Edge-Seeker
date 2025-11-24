# experiments/proteins_maui_normalization.py

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import set_seed, DEFAULT_DELTA, DEFAULT_AGG, DEFAULT_USE_PROB  # adjust import path
from data import load_tud_dataset
from training import train_private_gcn, make_gbb_api_for_model
from linkteller import linkteller_scores_for_graph
from metrics import undirected_edge_set, top_pairs_from_score_dict, prf1, auroc_from_scores

def run_proteins_maui_experiment(
    seed: int = 42,
    max_graphs: int = None,
    degree_power: float = 1.0,
):
    set_seed(seed)

    dataset = load_tud_dataset("PROTEINS")
    num_graphs = len(dataset)
    if max_graphs is not None:
        num_graphs = min(num_graphs, max_graphs)

    base_f1_list = []
    base_auc_list = []

    maui_f1_list = []
    maui_auc_list = []

    for idx in tqdm(range(num_graphs), desc="PROTEINS graphs"):
        data = dataset[idx]
        # train private GCN as in your previous experiments
        model, X, edge_index = train_private_gcn(
            data,
            epochs=120,  # or DEFAULT_TRAIN_EPOCHS
        )
        gbb_api = make_gbb_api_for_model(model, edge_index, X)

        num_nodes = X.shape[0]

        # true undirected edges
        true_pairs = undirected_edge_set(edge_index, num_nodes=num_nodes)
        top_k = len(true_pairs)

        # 1. vanilla LinkTeller
        scores_base = linkteller_scores_for_graph(
            gbb_api=gbb_api,
            X=X,
            delta=DEFAULT_DELTA,
            agg=DEFAULT_AGG,
            use_prob=DEFAULT_USE_PROB,
        )

        pred_pairs_base = top_pairs_from_score_dict(scores_base, top_k)
        p_b, r_b, f1_b = prf1(pred_pairs_base, true_pairs)
        auc_b = auroc_from_scores(scores_base, true_pairs)

        base_f1_list.append(f1_b)
        base_auc_list.append(auc_b)

        # 2. Maui-normalized LinkTeller
        scores_maui = linkteller_scores_for_graph(
            gbb_api=gbb_api,
            X=X,
            delta=DEFAULT_DELTA,
            agg=DEFAULT_AGG,
            use_prob=DEFAULT_USE_PROB,
            edge_index=edge_index,
            norm_by_degree=True,
            degree_power=degree_power,
        )

        pred_pairs_maui = top_pairs_from_score_dict(scores_maui, top_k)
        p_m, r_m, f1_m = prf1(pred_pairs_maui, true_pairs)
        auc_m = auroc_from_scores(scores_maui, true_pairs)

        maui_f1_list.append(f1_m)
        maui_auc_list.append(auc_m)

    results = {
        "base_f1": np.array(base_f1_list),
        "base_auc": np.array(base_auc_list),
        "maui_f1": np.array(maui_f1_list),
        "maui_auc": np.array(maui_auc_list),
    }

    return results


def summarize_results(results):
    def mean_std(x):
        return float(np.mean(x)), float(np.std(x))

    base_f1_mu, base_f1_sd = mean_std(results["base_f1"])
    base_auc_mu, base_auc_sd = mean_std(results["base_auc"])
    maui_f1_mu, maui_f1_sd = mean_std(results["maui_f1"])
    maui_auc_mu, maui_auc_sd = mean_std(results["maui_auc"])

    print("=== PROTEINS LinkTeller vs Maui-normalized ===")
    print(f"Base   F1:   {base_f1_mu:.4f} ± {base_f1_sd:.4f}")
    print(f"Maui   F1:   {maui_f1_mu:.4f} ± {maui_f1_sd:.4f}")
    print(f"Base  AUROC:{base_auc_mu:.4f} ± {base_auc_sd:.4f}")
    print(f"Maui  AUROC:{maui_auc_mu:.4f} ± {maui_auc_sd:.4f}")


def plot_results(results, out_prefix: str = "proteins_maui"):
    base_f1 = results["base_f1"]
    maui_f1 = results["maui_f1"]
    base_auc = results["base_auc"]
    maui_auc = results["maui_auc"]

    # 1. Bar plot of mean F1 and AUROC
    metrics = ["F1", "AUROC"]
    base_means = [np.mean(base_f1), np.mean(base_auc)]
    maui_means = [np.mean(maui_f1), np.mean(maui_auc)]
    base_stds = [np.std(base_f1), np.std(base_auc)]
    maui_stds = [np.std(maui_f1), np.std(maui_auc)]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, base_means, width, yerr=base_stds, label="LinkTeller")
    plt.bar(x + width / 2, maui_means, width, yerr=maui_stds, label="LinkTeller + Maui norm")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("PROTEINS edge recovery: effect of Maui normalization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bar.png", dpi=300)

    # 2. Scatter plot: per-graph F1 base vs Maui
    plt.figure(figsize=(5, 5))
    plt.scatter(base_f1, maui_f1, alpha=0.6)
    lims = [
        min(base_f1.min(), maui_f1.min()),
        max(base_f1.max(), maui_f1.max()),
    ]
    plt.plot(lims, lims)  # y = x
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Base F1")
    plt.ylabel("Maui normalized F1")
    plt.title("Per graph F1: Maui norm vs base")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scatter_f1.png", dpi=300)

    # 3. Histogram of F1 improvement
    delta_f1 = maui_f1 - base_f1
    plt.figure(figsize=(6, 4))
    plt.hist(delta_f1, bins=30)
    plt.xlabel("ΔF1 = F1_maui - F1_base")
    plt.ylabel("Count")
    plt.title("Distribution of F1 change with Maui normalization")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_delta_f1_hist.png", dpi=300)

    # same for AUROC if you care
    delta_auc = maui_auc - base_auc
    plt.figure(figsize=(6, 4))
    plt.hist(delta_auc, bins=30)
    plt.xlabel("ΔAUROC = AUROC_maui - AUROC_base")
    plt.ylabel("Count")
    plt.title("Distribution of AUROC change with Maui normalization")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_delta_auc_hist.png", dpi=300)


if __name__ == "__main__":
    results = run_proteins_maui_experiment(seed=42, max_graphs=None, degree_power=1.0)
    summarize_results(results)
    plot_results(results)
