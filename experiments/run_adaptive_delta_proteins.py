# experiments/run_adaptive_delta_proteins.py

import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from edge_seeker.config import set_seed, DEFAULT_TRAIN_EPOCHS  # :contentReference[oaicite:5]{index=5}
from edge_seeker.data import load_tud_dataset  # :contentReference[oaicite:6]{index=6}
from edge_seeker.adaptive_delta import run_adaptive_delta_for_dataset


def main():
    set_seed(42)

    results_dir = "results/proteins_adaptive_delta"
    figs_dir = os.path.join(results_dir, "figs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    print("Loading PROTEINS dataset")
    dataset = load_tud_dataset("PROTEINS")

    # delta search grid
    delta_grid = [
        1e-4,
        5e-4,
        1e-3,
        2e-3,
        5e-3,
        1e-2,
        2e-2,
        5e-2,
        1e-1,
    ]

    print("Running adaptive delta search on PROTEINS")
    rows = run_adaptive_delta_for_dataset(
        dataset=dataset,
        delta_grid=delta_grid,
        train_epochs=DEFAULT_TRAIN_EPOCHS,
        lr=1e-2,
        weight_decay=5e-4,
        agg="max",
        use_prob=True,
        verbose=False,
    )

    df = pd.DataFrame(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"proteins_adaptive_delta_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics table to {csv_path}")

    # Filter only baseline and adaptive rows for clean plots
    df_cmp = df[df["mode"].isin(["baseline_fixed_delta", "adaptive_delta"])].copy()

    # pivot to have columns baseline_*, adaptive_* per graph
    df_wide = (
        df_cmp
        .pivot_table(
            index="graph_id",
            columns="mode",
            values=["attack_auroc", "attack_f1", "delta", "threshold"],
            aggfunc="first",
        )
    )
    # flatten MultiIndex columns
    df_wide.columns = [
        f"{metric}_{mode}"
        for metric, mode in df_wide.columns.to_flat_index()
    ]
    df_wide = df_wide.reset_index()

    # 1. Scatter plot: AUROC baseline vs adaptive
    plt.figure(figsize=(5, 5))
    sns.scatterplot(
        data=df_wide,
        x="attack_auroc_baseline_fixed_delta",
        y="attack_auroc_adaptive_delta",
    )
    max_val = max(
        df_wide["attack_auroc_baseline_fixed_delta"].max(),
        df_wide["attack_auroc_adaptive_delta"].max(),
    )
    min_val = min(
        df_wide["attack_auroc_baseline_fixed_delta"].min(),
        df_wide["attack_auroc_adaptive_delta"].min(),
    )
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("AUROC baseline Δ")
    plt.ylabel("AUROC adaptive Δ")
    plt.title("PROTEINS LinkTeller AUROC: baseline vs adaptive Δ")
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, "proteins_auroc_baseline_vs_adaptive.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure {fig_path}")

    # 2. Boxplot: F1 baseline vs adaptive
    df_box = df_cmp[["graph_id", "mode", "attack_f1"]].copy()
    plt.figure(figsize=(5, 4))
    sns.boxplot(
        data=df_box,
        x="mode",
        y="attack_f1",
    )
    plt.xlabel("")
    plt.ylabel("Attack F1")
    plt.title("PROTEINS LinkTeller F1: baseline vs adaptive Δ")
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, "proteins_f1_baseline_vs_adaptive_box.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure {fig_path}")

    # 3. Scatter: best delta vs number of nodes (to show data driven behavior)
    # need per graph info, join with size stats
    size_cols = ["graph_id", "num_nodes", "num_edges"]
    df_sizes = (
        df_cmp[size_cols]
        .drop_duplicates(subset=["graph_id"])
        .reset_index(drop=True)
    )
    df_delta = df_wide[["graph_id", "delta_adaptive_delta"]].rename(
        columns={"delta_adaptive_delta": "delta_best"}
    )
    df_delta = df_delta.merge(df_sizes, on="graph_id", how="left")

    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=df_delta,
        x="num_nodes",
        y="delta_best",
    )
    plt.xlabel("Number of nodes")
    plt.ylabel("Best Δ per graph")
    plt.title("PROTEINS: data driven Δ vs graph size")
    plt.tight_layout()
    fig_path = os.path.join(figs_dir, "proteins_delta_vs_numnodes.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure {fig_path}")

    print("Done.")


if __name__ == "__main__":
    main()
