# experiments/depth_sweep_all_graphs.py

import os
import json
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from edge_seeker.config import (
    set_seed,
    DEVICE,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_DELTA,
    DEFAULT_AGG,
    DEFAULT_USE_PROB,
)
from edge_seeker.data import load_tud_dataset
from edge_seeker.features import build_features
from edge_seeker.training import train_gcn_with_depth, make_gbb_api_for_model
from edge_seeker.linkteller import linkteller_scores_for_graph
from edge_seeker.metrics import undirected_edge_set, auroc_from_scores


def run_depth_sweep_all_graphs(
    dataset_name: str,
    depths=(1, 2, 3, 4, 5),
    out_dir: str = "results/depth_sweep_all",
    delta=None,
    use_prob=None,
    agg=None,
    verbose: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    if delta is None:
        delta = DEFAULT_DELTA
    if use_prob is None:
        use_prob = DEFAULT_USE_PROB
    if agg is None:
        agg = DEFAULT_AGG

    dataset = load_tud_dataset(dataset_name)
    num_graphs = len(dataset)
    if verbose:
        print(f"Dataset {dataset_name} has {num_graphs} graphs")

    # metrics[L] is a list over graphs
    acc_per_L = {L: [] for L in depths}
    lt_auroc_per_L = {L: [] for L in depths}

    per_graph_results = {}

    for g_idx, data in enumerate(dataset):
        if verbose:
            print(f"\n=== Graph {g_idx + 1}/{num_graphs} (nodes={data.num_nodes}, edges={data.num_edges}) ===")

        X_full = build_features(data).to(DEVICE)
        edge_index = data.edge_index.to(DEVICE)
        num_nodes = data.num_nodes
        true_pairs = undirected_edge_set(edge_index, num_nodes)

        per_graph_results[g_idx] = {}

        for L in depths:
            if verbose:
                print(f"  -> Training depth L={L}")

            model, stats = train_gcn_with_depth(
                data,
                num_layers=L,
                num_classes=2,
                epochs=DEFAULT_TRAIN_EPOCHS,
                lr=1e-2,
                weight_decay=5e-4,
                val_ratio=0.2,
                test_ratio=0.2,
                patience=40,
                verbose=False,
            )

            test_acc = stats["final_test_acc"]
            acc_per_L[L].append(test_acc)

            gbb_api = make_gbb_api_for_model(model, edge_index, X_full)
            score_dict = linkteller_scores_for_graph(
                gbb_api,
                X_full,
                delta=delta,
                agg=agg,
                use_prob=use_prob,
            )

            lt_auroc = auroc_from_scores(score_dict, true_pairs, num_neg=None)
            lt_auroc_per_L[L].append(lt_auroc)

            if verbose:
                print(f"    L={L}: test_acc={test_acc:.3f}, LinkTeller AUROC={lt_auroc:.3f}")

            per_graph_results[g_idx][L] = {
                "test_acc": float(test_acc),
                "lt_auroc": float(lt_auroc),
                "best_val_acc": float(stats["best_val_acc"]),
                "epoch_trained": int(stats["epoch_trained"]),
            }

    # aggregate over graphs
    def safe_mean_std(values):
        vals = np.array([v for v in values if not (math.isnan(v) or math.isinf(v))], dtype=float)
        if len(vals) == 0:
            return float("nan"), float("nan")
        return float(vals.mean()), float(vals.std())

    summary = {
        "dataset": dataset_name,
        "num_graphs": num_graphs,
        "depths": list(depths),
        "per_graph": per_graph_results,
        "aggregate": {},
    }

    for L in depths:
        acc_mean, acc_std = safe_mean_std(acc_per_L[L])
        lt_mean, lt_std = safe_mean_std(lt_auroc_per_L[L])
        summary["aggregate"][L] = {
            "test_acc_mean": acc_mean,
            "test_acc_std": acc_std,
            "lt_auroc_mean": lt_mean,
            "lt_auroc_std": lt_std,
            "num_graphs_used": len(acc_per_L[L]),
        }

    json_path = os.path.join(out_dir, f"{dataset_name}_depth_sweep_all_graphs.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # good plots for paper

    depths_list = list(depths)

    acc_means = [summary["aggregate"][L]["test_acc_mean"] for L in depths_list]
    acc_stds = [summary["aggregate"][L]["test_acc_std"] for L in depths_list]

    auroc_means = [summary["aggregate"][L]["lt_auroc_mean"] for L in depths_list]
    auroc_stds = [summary["aggregate"][L]["lt_auroc_std"] for L in depths_list]

    # accuracy vs depth
    plt.figure(figsize=(4.0, 3.2), dpi=200)
    plt.errorbar(depths_list, acc_means, yerr=acc_stds, marker="o", capsize=4)
    plt.xlabel("GCN depth L")
    plt.ylabel("Node accuracy (mean ± std)")
    plt.title(f"{dataset_name}  all graphs  accuracy vs depth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    acc_png = os.path.join(out_dir, f"{dataset_name}_acc_vs_depth_all_graphs.png")
    plt.savefig(acc_png, bbox_inches="tight")
    plt.close()

    # LinkTeller AUROC vs depth
    plt.figure(figsize=(4.0, 3.2), dpi=200)
    plt.errorbar(depths_list, auroc_means, yerr=auroc_stds, marker="s", capsize=4)
    plt.xlabel("GCN depth L")
    plt.ylabel("LinkTeller AUROC (mean ± std)")
    plt.title(f"{dataset_name}  all graphs  LinkTeller AUROC vs depth")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    lt_png = os.path.join(out_dir, f"{dataset_name}_lt_auroc_vs_depth_all_graphs.png")
    plt.savefig(lt_png, bbox_inches="tight")
    plt.close()

    # combined figure
    plt.figure(figsize=(5.0, 3.2), dpi=200)
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    line1 = ax1.errorbar(depths_list, acc_means, yerr=acc_stds, marker="o", capsize=4, label="Accuracy")
    line2 = ax2.errorbar(depths_list, auroc_means, yerr=auroc_stds, marker="s", capsize=4, linestyle="--", label="LinkTeller AUROC")

    ax1.set_xlabel("GCN depth L")
    ax1.set_ylabel("Node accuracy (mean)")
    ax2.set_ylabel("LinkTeller AUROC (mean)")

    lines = [line1, line2]
    labels = ["Accuracy", "LinkTeller AUROC"]
    ax1.legend(lines, labels, loc="best")

    plt.title(f"{dataset_name}  all graphs  depth sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    combo_png = os.path.join(out_dir, f"{dataset_name}_depth_sweep_all_graphs_combined.png")
    plt.savefig(combo_png, bbox_inches="tight")
    plt.close()

    return {
        "json_path": json_path,
        "acc_png": acc_png,
        "lt_png": lt_png,
        "combo_png": combo_png,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG")
    parser.add_argument("--out-dir", type=str, default="results/depth_sweep_all")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    res = run_depth_sweep_all_graphs(
        dataset_name=args.dataset,
        depths=tuple(args.depths),
        out_dir=args.out_dir,
        verbose=args.verbose,
    )

    if args.verbose:
        print("\nFinished depth sweep on all graphs.")
        print("Results JSON:", res["json_path"])
        print("Accuracy plot:", res["acc_png"])
        print("LinkTeller plot:", res["lt_png"])
        print("Combined plot:", res["combo_png"])


if __name__ == "__main__":
    main()
