# experiments/proteins_query_budget_defense.py

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

from edge_seeker.config import (
    set_seed,
    SEED,
    DEFAULT_DELTA,
    DEFAULT_AGG,
    DEFAULT_USE_PROB,
    DEFAULT_TRAIN_EPOCHS,
)
from edge_seeker.data import load_tud_dataset
from edge_seeker.query_budget_defense import run_query_budget_defense_for_dataset


def aggregate_query_budget_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (query_frac, delta) and compute mean and std of metrics across graphs.
    """
    agg_df = (
        df.groupby(["query_frac", "query_pct", "delta"])
        .agg(
            attack_auroc_mean=("attack_auroc", "mean"),
            attack_auroc_std=("attack_auroc", "std"),
            attack_f1_mean=("attack_f1", "mean"),
            attack_f1_std=("attack_f1", "std"),
            attack_edge_recall_mean=("attack_edge_recall", "mean"),
            attack_edge_recall_std=("attack_edge_recall", "std"),
        )
        .reset_index()
    )
    return agg_df


def plot_delta_vs_f1(
    agg_df: pd.DataFrame,
    out_path: str,
    title: str,
):
    """
    Plot Δ vs F1 with different curves for each query budget q.
    """
    plt.figure(figsize=(6, 4))

    query_fracs: List[float] = sorted(agg_df["query_frac"].unique())
    for q in query_fracs:
        sub = agg_df[agg_df["query_frac"] == q].sort_values("delta")
        deltas = sub["delta"].values
        means = sub["attack_f1_mean"].values
        stds = sub["attack_f1_std"].values
        label = f"{int(round(q * 100))}% queries"

        plt.errorbar(
            deltas,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=label,
        )

    plt.xscale("log")
    plt.xlabel(r"$\Delta$ (LinkTeller injection magnitude)")
    plt.ylabel("Attack F1 score")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    set_seed(SEED)

    print("Loading PROTEINS dataset")
    dataset = load_tud_dataset("PROTEINS")
    print(f"Total graphs in PROTEINS: {len(dataset)}")

    # Same sigma grid as your previous noise experiment
    sigma_grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    sigma_mid = sigma_grid[len(sigma_grid) // 2]  # 0.1
    noise_type = "gaussian"

    # Query budgets q
    query_fracs = [0.10, 0.25, 0.50, 1.00]

    # Delta sweep for the x axis
    # You can tweak these if LinkTeller behaves badly
    delta_list = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

    print(
        f"Running Query Budget + Defense experiment on PROTEINS "
        f"with sigma_mid={sigma_mid} ({noise_type})"
    )

    all_rows = run_query_budget_defense_for_dataset(
        dataset=dataset,
        sigma_mid=sigma_mid,
        delta_list=delta_list,
        query_fracs=query_fracs,
        noise_type=noise_type,
        train_epochs=DEFAULT_TRAIN_EPOCHS,
        lr=1e-2,
        weight_decay=5e-4,
        agg=DEFAULT_AGG,
        use_prob=DEFAULT_USE_PROB,
        verbose=False,
    )

    df = pd.DataFrame(all_rows)

    results_dir = os.path.join(ROOT_DIR, "results", "proteins_query_budget_defense")
    os.makedirs(results_dir, exist_ok=True)

    per_graph_csv = os.path.join(results_dir, "proteins_query_budget_defense_per_graph.csv")
    df.to_csv(per_graph_csv, index=False)
    print(f"Saved per graph metrics to {per_graph_csv}")

    agg_df = aggregate_query_budget_results(df)
    agg_csv = os.path.join(results_dir, "proteins_query_budget_defense_agg.csv")
    agg_df.to_csv(agg_csv, index=False)
    print(f"Saved aggregated metrics to {agg_csv}")

    # Δ vs F1 plot
    fig_path = os.path.join(results_dir, "delta_vs_f1_sigma_mid_gaussian.png")
    plot_delta_vs_f1(
        agg_df=agg_df,
        out_path=fig_path,
        title="Query budget + noise defense on PROTEINS (Δ vs F1)",
    )

    print("Finished Query Budget + Defense experiment for PROTEINS")
    print("Key outputs:")
    print(f"  Per graph CSV: {per_graph_csv}")
    print(f"  Aggregated CSV: {agg_csv}")
    print(f"  Plot: {fig_path}")


if __name__ == "__main__":
    main()
