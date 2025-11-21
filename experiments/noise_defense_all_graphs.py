# experiments/proteins_noise_defense_all_graphs.py

import os
import sys
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

# Make repo root importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import edge_seeker as es
from edge_seeker.noise_defense import run_noise_defense_for_dataset


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by noise_type and sigma and compute mean and std
    for all metrics across graphs.
    """
    agg_df = (
        df.groupby(["noise_type", "sigma"])
        .agg(
            attack_auroc_mean=("attack_auroc", "mean"),
            attack_auroc_std=("attack_auroc", "std"),
            attack_f1_mean=("attack_f1", "mean"),
            attack_f1_std=("attack_f1", "std"),
            attack_edge_recall_mean=("attack_edge_recall", "mean"),
            attack_edge_recall_std=("attack_edge_recall", "std"),
            model_acc_mean=("model_acc", "mean"),
            model_acc_std=("model_acc", "std"),
        )
        .reset_index()
    )
    return agg_df


def plot_metric_with_errorbars(
    agg_df: pd.DataFrame,
    metric_mean: str,
    metric_std: str,
    title: str,
    ylabel: str,
    out_path: str,
):
    """
    Plot mean +/- std vs sigma for each noise type.
    """
    plt.figure(figsize=(5, 4))

    noise_types: List[str] = sorted(agg_df["noise_type"].unique())
    for nt in noise_types:
        sub = agg_df[agg_df["noise_type"] == nt].sort_values("sigma")
        sigmas = sub["sigma"].values
        means = sub[metric_mean].values
        stds = sub[metric_std].values

        plt.errorbar(
            sigmas,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=nt.capitalize(),
        )

    plt.xlabel(r"$\sigma$ (logit noise scale)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    es.set_seed(es.SEED)

    print("Loading PROTEINS dataset")
    dataset = es.load_tud_dataset("PROTEINS")
    print(f"Total graphs in PROTEINS: {len(dataset)}")

    sigmas = [0.0, 0.05, 0.1, 0.2, 0.3]
    noise_types = ["gaussian", "laplace"]

    print("Running noise defense sweep on all graphs")
    all_results = run_noise_defense_for_dataset(
        dataset=dataset,
        sigmas=sigmas,
        noise_types=noise_types,
        train_epochs=es.DEFAULT_TRAIN_EPOCHS,
        lr=1e-2,
        weight_decay=5e-4,
        delta=es.DEFAULT_DELTA,
        agg=es.DEFAULT_AGG,
        use_prob=es.DEFAULT_USE_PROB,
        verbose=False,
    )

    df = pd.DataFrame(all_results)

    results_dir = os.path.join(ROOT_DIR, "results", "proteins_noise_defense")
    os.makedirs(results_dir, exist_ok=True)

    per_graph_csv = os.path.join(results_dir, "proteins_noise_defense_per_graph.csv")
    df.to_csv(per_graph_csv, index=False)
    print(f"Saved per graph metrics to {per_graph_csv}")

    agg_df = aggregate_results(df)
    agg_csv = os.path.join(results_dir, "proteins_noise_defense_agg.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregated metrics to {agg_csv}")

    # Plots for report
    plot_metric_with_errorbars(
        agg_df=agg_df,
        metric_mean="attack_auroc_mean",
        metric_std="attack_auroc_std",
        title="Attack AUROC vs logit noise (PROTEINS, 1113 graphs)",
        ylabel="Attack AUROC",
        out_path=os.path.join(results_dir, "attack_auroc_vs_sigma.png"),
    )

    plot_metric_with_errorbars(
        agg_df=agg_df,
        metric_mean="attack_f1_mean",
        metric_std="attack_f1_std",
        title="Attack F1 vs logit noise (PROTEINS, 1113 graphs)",
        ylabel="Attack F1",
        out_path=os.path.join(results_dir, "attack_f1_vs_sigma.png"),
    )

    plot_metric_with_errorbars(
        agg_df=agg_df,
        metric_mean="attack_edge_recall_mean",
        metric_std="attack_edge_recall_std",
        title="Attack edge recall vs logit noise (PROTEINS, 1113 graphs)",
        ylabel="Edge recall at K",
        out_path=os.path.join(results_dir, "attack_edge_recall_vs_sigma.png"),
    )

    plot_metric_with_errorbars(
        agg_df=agg_df,
        metric_mean="model_acc_mean",
        metric_std="model_acc_std",
        title="Node accuracy vs logit noise (PROTEINS, 1113 graphs)",
        ylabel="Node classification accuracy",
        out_path=os.path.join(results_dir, "model_acc_vs_sigma.png"),
    )

    print("Finished noise defense experiment for PROTEINS")


if __name__ == "__main__":
    main()
