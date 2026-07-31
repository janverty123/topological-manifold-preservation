"""
evaluate_multitask.py
-----------------------
Comparison and statistical analysis for the N-task continual learning
logs produced by run_multitask.py / src/train_general.py (e.g. 5-task
Split-MNIST). This is SEPARATE from src/evaluate.py, which is built
for the original 2-task Split-MNIST log format (a flat
`retention_accuracy` field). Multi-task logs instead have
`avg_retention_accuracy`, `per_task_retention` (a dict), and `task_idx`
-- this module is built around that shape.

Auto-detects whichever of {finetune, ewc, tmp}.jsonl are present in the
given log directory, so it works even if you've only run some of the
three methods so far.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


KNOWN_METHODS = ["finetune", "ewc", "tmp"]


def load_multitask_log(jsonl_path):
    """Loads only the per-epoch task-training records (skips the
    task0_pretrain records, which have a different shape)."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if "avg_retention_accuracy" in r:
                records.append(r)
    return records


def build_comparison_table(log_dir: str, methods=None):
    """
    Returns a long-format DataFrame, one row per (method, task_idx, epoch),
    with a `global_step` column (task_idx and epoch combined into a
    single increasing index) for easy plotting across the whole
    sequence, plus expanded per-task retention columns
    (`retention_task0`, `retention_task1`, ...).
    """
    methods = methods or KNOWN_METHODS
    rows = []
    found_methods = []

    for method in methods:
        path = os.path.join(log_dir, f"{method}.jsonl")
        if not os.path.exists(path):
            continue
        found_methods.append(method)
        records = load_multitask_log(path)
        for r in records:
            row = {
                "method": method,
                "task_idx": r["task_idx"],
                "epoch": r["epoch"],
                "avg_retention_accuracy": r["avg_retention_accuracy"],
                "learning_accuracy": r["learning_accuracy"],
                "avg_ce_loss": r.get("avg_ce_loss"),
                "avg_extra_term": r.get("avg_extra_term"),
                "epoch_time_sec": r.get("epoch_time_sec"),
                "memory_mb": r.get("memory_mb"),
                "feature_space_drift_w_inf": r.get("feature_space_drift_w_inf"),
            }
            for task_key, acc in r.get("per_task_retention", {}).items():
                row[f"retention_{task_key}"] = acc
            rows.append(row)

    if not found_methods:
        raise FileNotFoundError(
            f"No multi-task logs found in {log_dir}. Expected one or more of: "
            f"{[m + '.jsonl' for m in methods]}"
        )

    df = pd.DataFrame(rows)
    # global_step: cumulative epoch count across the whole sequence,
    # so task boundaries are visible when plotted continuously
    df = df.sort_values(["method", "task_idx", "epoch"]).reset_index(drop=True)
    df["global_step"] = df.groupby("method").cumcount() + 1

    print(f"Loaded methods: {found_methods}")
    return df, found_methods


def summarize(df: pd.DataFrame):
    """Per-method mean/std of the key metrics, across all tasks/epochs."""
    cols = ["avg_retention_accuracy", "learning_accuracy", "epoch_time_sec", "memory_mb"]
    return df.groupby("method")[cols].agg(["mean", "std"])


def summarize_final_epoch_per_task(df: pd.DataFrame):
    """
    For each method and task_idx, the metrics at the FINAL epoch of
    that task's training -- i.e. "how well did retention/learning end
    up once the model moved on from this task". This is usually the
    more meaningful number to report/plot than an average across all
    epochs (which includes early, still-converging epochs).
    """
    idx = df.groupby(["method", "task_idx"])["epoch"].idxmax()
    return df.loc[idx].sort_values(["method", "task_idx"]).reset_index(drop=True)


def significance_test(df: pd.DataFrame, metric="avg_retention_accuracy",
                       method_a="tmp", method_b="ewc"):
    """
    Paired Wilcoxon signed-rank test on `metric`, matched by
    (task_idx, epoch) between two methods. Requires both methods to
    have the same task/epoch structure (same num_tasks, same
    task2_epochs) -- true by construction if both were run from the
    same config.
    """
    a_df = df[df.method == method_a].sort_values(["task_idx", "epoch"])
    b_df = df[df.method == method_b].sort_values(["task_idx", "epoch"])
    n = min(len(a_df), len(b_df))
    a = a_df[metric].to_numpy()[:n]
    b = b_df[metric].to_numpy()[:n]

    if n < 2 or np.allclose(a, b):
        return {"statistic": None, "p_value": None,
                "verdict": "Insufficient or identical paired samples for a Wilcoxon test."}

    stat, p = wilcoxon(a, b)
    verdict = (
        f"Reject H0 (p={p:.4f} < 0.05): significant difference in {metric} "
        f"between {method_a} and {method_b}."
        if p < 0.05 else
        f"Fail to reject H0 (p={p:.4f} >= 0.05): no significant difference in "
        f"{metric} between {method_a} and {method_b}."
    )
    return {"statistic": float(stat), "p_value": float(p), "verdict": verdict}


def build_retention_trajectory_matrix(df: pd.DataFrame, method: str):
    """
    Recommendation 5: sequential task-retention trajectories.

    Returns a dict {task_id: [(task_idx_trained, retention_accuracy), ...]}
    tracing how EACH task's retention accuracy evolves as training moves
    on to each SUBSEQUENT task (measured at the final epoch of each
    task's training) -- i.e. row i, column j of the conceptual matrix
    A[i][j] = accuracy on task j after finishing training on task i.

    This shows WHEN a task's retention drops, not just its final value,
    which a single before/after number can hide (e.g. a sudden
    collapse right after one specific task vs. gradual erosion).
    """
    sub = df[df.method == method]
    final_per_task = sub.loc[sub.groupby("task_idx")["epoch"].idxmax()].sort_values("task_idx")

    trajectories = {}
    for _, row in final_per_task.iterrows():
        trained_up_to = int(row["task_idx"])
        for col in [c for c in df.columns if c.startswith("retention_task")]:
            if pd.notna(row.get(col)):
                task_id = col.replace("retention_", "")
                trajectories.setdefault(task_id, []).append((trained_up_to, row[col]))

    return trajectories



def run_full_multitask_evaluation(log_dir: str, output_path: str):
    df, found_methods = build_comparison_table(log_dir)
    summary = summarize(df)
    summary_flat = summary.copy()
    summary_flat.columns = [f"{metric}_{stat}" for metric, stat in summary_flat.columns]

    final_per_task = summarize_final_epoch_per_task(df)

    results = {
        "methods_compared": found_methods,
        "summary_all_epochs": summary_flat.to_dict(orient="index"),
        "final_epoch_per_task": final_per_task.to_dict(orient="records"),
    }

    if "tmp" in found_methods and "ewc" in found_methods:
        results["retention_tmp_vs_ewc"] = significance_test(df, "avg_retention_accuracy", "tmp", "ewc")
        results["learning_tmp_vs_ewc"] = significance_test(df, "learning_accuracy", "tmp", "ewc")
    if "tmp" in found_methods and "finetune" in found_methods:
        results["retention_tmp_vs_finetune"] = significance_test(df, "avg_retention_accuracy", "tmp", "finetune")
    if "ewc" in found_methods and "finetune" in found_methods:
        results["retention_ewc_vs_finetune"] = significance_test(df, "avg_retention_accuracy", "ewc", "finetune")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps(results, indent=2, default=str))
    return df, results
