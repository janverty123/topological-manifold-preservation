"""
visualize.py
------------
Implements the final step of the Simulation section: "exporting all
log files into matplotlib to generate comparative line graphs and
heatmaps."
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_over_epochs(df: pd.DataFrame, metric: str, ylabel: str, out_path: str, title=None):
    plt.figure(figsize=(7, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group[metric], marker="o", label=method)
    plt.xlabel("Task-2 Epoch")
    plt.ylabel(ylabel)
    plt.title(title or f"{ylabel} vs. Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_retention_accuracy(df, out_path):
    plot_metric_over_epochs(df, "retention_accuracy", "Retention Accuracy (Task 1)", out_path,
                             title="Retention Accuracy: TMP vs. EWC vs. Finetune")


def plot_learning_accuracy(df, out_path):
    plot_metric_over_epochs(df, "learning_accuracy", "Learning Accuracy (Task 2)", out_path,
                             title="Learning Rate Efficiency: TMP vs. EWC vs. Finetune")


def plot_computational_overhead(df, out_path):
    plot_metric_over_epochs(df, "epoch_time_sec", "Epoch Time (s)", out_path,
                             title="Computational Overhead: Epoch Time")


def plot_memory_overhead(df, out_path):
    plot_metric_over_epochs(df, "memory_mb", "Resident Memory (MB)", out_path,
                             title="Computational Overhead: Memory Usage")


def plot_feature_space_drift(df, out_path):
    tmp_df = df[df.method == "tmp"].sort_values("epoch")
    if tmp_df["feature_space_drift_w_inf"].isna().all():
        return
    plt.figure(figsize=(7, 5))
    plt.plot(tmp_df["epoch"], tmp_df["feature_space_drift_w_inf"], marker="o", color="crimson")
    plt.xlabel("Task-2 Epoch")
    plt.ylabel("Bottleneck Distance W_inf(D_base, D_current)")
    plt.title("Feature Space Drift Over Task-2 Training (TMP)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_summary_heatmap(summary_df: pd.DataFrame, out_path: str):
    """
    `summary_df` is the multi-indexed mean/std table from evaluate.summarize().
    Plots a heatmap of mean values across methods x metrics.
    """
    means = summary_df.xs("mean", axis=1, level=1)
    plt.figure(figsize=(6, 4))
    plt.imshow(means.values, cmap="viridis", aspect="auto")
    plt.colorbar(label="Mean Value")
    plt.xticks(range(len(means.columns)), means.columns, rotation=30, ha="right")
    plt.yticks(range(len(means.index)), means.index)
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            plt.text(j, i, f"{means.values[i, j]:.3f}", ha="center", va="center",
                      color="white", fontsize=8)
    plt.title("Mean Metric Comparison Heatmap")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_plots(df: pd.DataFrame, summary_df: pd.DataFrame, plot_dir: str):
    plot_retention_accuracy(df, os.path.join(plot_dir, "retention_accuracy.png"))
    plot_learning_accuracy(df, os.path.join(plot_dir, "learning_accuracy.png"))
    plot_computational_overhead(df, os.path.join(plot_dir, "epoch_time.png"))
    plot_memory_overhead(df, os.path.join(plot_dir, "memory_usage.png"))
    plot_feature_space_drift(df, os.path.join(plot_dir, "feature_space_drift.png"))
    plot_summary_heatmap(summary_df, os.path.join(plot_dir, "summary_heatmap.png"))
