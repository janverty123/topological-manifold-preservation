"""
evaluate.py
-----------
Implements "Data Analysis": statistical comparison of TMP against the
Finetune and EWC baselines, addressing Research Questions 2 and 3 and
the study's Hypotheses (H0 / H1).

Loads the per-method JSONL logs produced by train.py, builds a tidy
comparison table, and runs a paired significance test (Wilcoxon
signed-rank, appropriate for small paired epoch-wise samples without a
normality assumption) on Retention Accuracy between TMP and each
baseline.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_epoch_records(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if "retention_accuracy" in r:
                records.append(r)
    return pd.DataFrame(records)


def build_comparison_table(log_paths: dict):
    """
    log_paths: {"finetune": path, "ewc": path, "tmp": path}
    Returns a long-format DataFrame with columns:
        method, epoch, retention_accuracy, learning_accuracy,
        epoch_time_sec, memory_mb, feature_space_drift_w_inf
    """
    frames = []
    for method, path in log_paths.items():
        df = load_epoch_records(path)
        df["method"] = method
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame):
    """Per-method mean/std summary of the four primary metrics."""
    cols = ["retention_accuracy", "learning_accuracy", "epoch_time_sec", "memory_mb"]
    return df.groupby("method")[cols].agg(["mean", "std"])


def significance_test(df: pd.DataFrame, metric="retention_accuracy",
                       method_a="tmp", method_b="finetune"):
    """
    Paired Wilcoxon signed-rank test on `metric` between two methods,
    matched by epoch. Addresses Research Question 2 / Hypothesis
    testing (H0 vs H1).

    Returns dict with statistic, p_value, and a plain-language verdict
    at alpha = 0.05.
    """
    a = df[df.method == method_a].sort_values("epoch")[metric].to_numpy()
    b = df[df.method == method_b].sort_values("epoch")[metric].to_numpy()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

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


def run_full_evaluation(log_dir: str, output_path: str):
    log_paths = {
        "finetune": os.path.join(log_dir, "finetune.jsonl"),
        "ewc": os.path.join(log_dir, "ewc.jsonl"),
        "tmp": os.path.join(log_dir, "tmp.jsonl"),
    }
    for m, p in log_paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing log for method '{m}': {p}. Run scripts/run_{m}.py first.")

    df = build_comparison_table(log_paths)
    summary = summarize(df)
    # flatten MultiIndex columns (metric, stat) -> "metric_stat" so the
    # summary table is JSON-serializable
    summary_flat = summary.copy()
    summary_flat.columns = [f"{metric}_{stat}" for metric, stat in summary_flat.columns]

    results = {
        "summary": summary_flat.to_dict(orient="index"),
        "retention_vs_finetune": significance_test(df, "retention_accuracy", "tmp", "finetune"),
        "retention_vs_ewc": significance_test(df, "retention_accuracy", "tmp", "ewc"),
        "drift_vs_finetune_note": (
            "Feature Space Drift (W_inf) is only computed for the 'tmp' method "
            "in this reference implementation; extend train.py to compute it "
            "for finetune/ewc as well if a drift comparison is required for "
            "Research Question 2."
        ),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    return df, results
