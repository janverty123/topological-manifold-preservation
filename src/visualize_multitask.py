"""
visualize_multitask.py
------------------------
Plotting for the N-task comparison (src/evaluate_multitask.py).
Separate from src/visualize.py, which is built for the 2-task format.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _task_boundaries(df, method):
    """Global-step positions where a new task starts, for drawing
    vertical reference lines on the continuous-sequence plots."""
    sub = df[df.method == method].sort_values("global_step")
    boundaries = sub.groupby("task_idx")["global_step"].min().tolist()
    return boundaries


def plot_metric_over_sequence(df: pd.DataFrame, metric: str, ylabel: str, out_path: str, title=None):
    plt.figure(figsize=(9, 5))
    for method, group in df.groupby("method"):
        group = group.sort_values("global_step")
        plt.plot(group["global_step"], group[metric], marker="o", markersize=3, label=method)

    # Draw task-boundary reference lines using the first method's structure
    # (all methods share the same task/epoch structure by construction)
    first_method = df["method"].iloc[0]
    for b in _task_boundaries(df, first_method):
        plt.axvline(x=b, color="gray", linestyle=":", alpha=0.4)

    plt.xlabel("Global training step (task boundaries marked)")
    plt.ylabel(ylabel)
    plt.title(title or f"{ylabel} across the task sequence")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_retention_over_sequence(df, out_path):
    plot_metric_over_sequence(df, "avg_retention_accuracy", "Avg. Retention Accuracy (all seen tasks)",
                               out_path, title="Retention Across the Task Sequence")


def plot_learning_over_sequence(df, out_path):
    plot_metric_over_sequence(df, "learning_accuracy", "Learning Accuracy (current task)",
                               out_path, title="New-Task Learning Across the Sequence")


def plot_per_task_retention_final(final_per_task: pd.DataFrame, out_path: str):
    """
    Grouped bar chart: for each method, retention on each PREVIOUSLY
    SEEN task, measured at the end of the LAST task's training (i.e.
    the final, most-forgotten state of every task). This is the
    "does protection decay with distance" plot.
    """
    last_task_idx = final_per_task["task_idx"].max()
    last_rows = final_per_task[final_per_task.task_idx == last_task_idx]

    retention_cols = [c for c in final_per_task.columns if c.startswith("retention_task")]
    retention_cols = sorted(retention_cols, key=lambda c: int(c.replace("retention_task", "")))

    methods = last_rows["method"].tolist()
    x = range(len(retention_cols))
    width = 0.8 / max(len(methods), 1)

    plt.figure(figsize=(9, 5))
    for i, method in enumerate(methods):
        row = last_rows[last_rows.method == method].iloc[0]
        values = [row[c] if pd.notna(row.get(c)) else 0 for c in retention_cols]
        offsets = [xi + i * width for xi in x]
        plt.bar(offsets, values, width=width, label=method)

    plt.xticks([xi + width * (len(methods) - 1) / 2 for xi in x],
               [c.replace("retention_", "") for c in retention_cols])
    plt.xlabel("Task (in order learned)")
    plt.ylabel(f"Retention accuracy (measured after task {last_task_idx} training)")
    plt.title("Retention by Task Age -- Does Protection Decay With Distance?")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_retention_vs_learning_tradeoff(df: pd.DataFrame, out_path: str):
    """
    Scatter: mean avg_retention_accuracy vs. mean learning_accuracy per
    method, across the whole sequence -- the stability-plasticity
    tradeoff view. Also draws the per-task-idx trajectory for each
    method as a faint connected path, in addition to the overall
    method-mean marker.
    """
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    for i, (method, group) in enumerate(df.groupby("method")):
        color = colors[i % len(colors)]
        per_task = group.groupby("task_idx")[["avg_retention_accuracy", "learning_accuracy"]].mean()
        plt.plot(per_task["learning_accuracy"], per_task["avg_retention_accuracy"],
                  linestyle="--", alpha=0.4, color=color)
        for task_idx, row in per_task.iterrows():
            plt.scatter(row["learning_accuracy"], row["avg_retention_accuracy"],
                        color=color, alpha=0.5, s=40)

        overall_mean = group[["avg_retention_accuracy", "learning_accuracy"]].mean()
        plt.scatter(overall_mean["learning_accuracy"], overall_mean["avg_retention_accuracy"],
                    color=color, marker="*", s=400, edgecolor="black", linewidth=1, label=f"{method} (mean)")

    plt.xlabel("Learning Accuracy (new-task performance)")
    plt.ylabel("Avg. Retention Accuracy (old-task performance)")
    plt.title("Stability-Plasticity Tradeoff\n(top-right is ideal: learns new tasks AND retains old ones)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_space_drift(df, out_path):
    tmp_df = df[(df.method == "tmp") & df["feature_space_drift_w_inf"].notna()].sort_values("global_step")
    if tmp_df.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(tmp_df["global_step"], tmp_df["feature_space_drift_w_inf"], marker="o", markersize=3, color="crimson")
    for b in _task_boundaries(df, "tmp"):
        plt.axvline(x=b, color="gray", linestyle=":", alpha=0.4)
    plt.xlabel("Global training step (task boundaries marked)")
    plt.ylabel("Bottleneck Distance W_inf (pooled reference)")
    plt.title("TMP: Feature Space Drift Across the Task Sequence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_task_retention_trajectories(trajectories: dict, method: str, out_path: str):
    """
    Recommendation 5: one line per task, tracing its retention accuracy
    across every subsequent task boundary, so you can see WHEN a task's
    protection collapses (sudden cliff vs. gradual erosion) rather than
    only its final value.

    `trajectories` comes from evaluate_multitask.build_retention_trajectory_matrix.
    """
    plt.figure(figsize=(8, 5))
    for task_id, points in sorted(trajectories.items()):
        points = sorted(points, key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=task_id)

    plt.xlabel("Task index just finished training")
    plt.ylabel("Retention accuracy")
    plt.title(f"Individual Task Retention Trajectories ({method})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(range(int(plt.xlim()[0]), int(plt.xlim()[1]) + 1))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def generate_all_multitask_plots(df: pd.DataFrame, final_per_task: pd.DataFrame, plot_dir: str):
    from src.evaluate_multitask import build_retention_trajectory_matrix

    plot_retention_over_sequence(df, os.path.join(plot_dir, "mt_retention_over_sequence.png"))
    plot_learning_over_sequence(df, os.path.join(plot_dir, "mt_learning_over_sequence.png"))
    plot_per_task_retention_final(final_per_task, os.path.join(plot_dir, "mt_per_task_retention_final.png"))
    plot_retention_vs_learning_tradeoff(df, os.path.join(plot_dir, "mt_tradeoff.png"))
    plot_feature_space_drift(df, os.path.join(plot_dir, "mt_feature_space_drift.png"))

    for method in df["method"].unique():
        trajectories = build_retention_trajectory_matrix(df, method)
        plot_task_retention_trajectories(
            trajectories, method,
            os.path.join(plot_dir, f"mt_retention_trajectories_{method}.png"),
        )
