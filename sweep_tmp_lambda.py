"""
sweep_tmp_lambda.py
--------------------
Runs the TMP method across several `tmp.lambda_` values (including a
0.0 ablation) using the SAME cached Task-1 baseline model and the SAME
baseline persistence diagram/point cloud for every run, so the only
thing that varies is lambda. This directly tests:

  1. Does retention accuracy actually respond to lambda in a sensible,
     monotonic-ish way (evidence the regularizer is doing real work),
     rather than being disconnected from it (the earlier symptom of the
     sampling-noise bug)?
  2. At lambda=0.0, does TMP collapse back toward Finetune-level
     retention (~24% in this project's runs so far)? If yes, that
     confirms the improvement at higher lambda comes from the
     regularization term itself, not merely from the extra Task-1
     forward passes injected into the training loop.

Usage:
    python sweep_tmp_lambda.py --config configs/baseline/split_mnist_2task.yaml
    python sweep_tmp_lambda.py --config configs/baseline/split_mnist_2task.yaml --lambdas 0.0,0.5,1.0,2.0,5.0,10.0
"""

import argparse
import copy
import json
import os

import torch
import yaml

from src.data import get_dataloaders
from src.models import MLPClassifier
from src.train import set_seed, train_task2, JsonlLogger, build_baseline_artifacts


def resolve_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/split_mnist_2task.yaml")
    parser.add_argument("--lambdas", default="0.0,0.5,1.0,2.0,5.0,10.0",
                         help="Comma-separated list of tmp.lambda_ values to sweep.")
    args = parser.parse_args()

    lambda_values = [float(x) for x in args.lambdas.split(",")]

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    set_seed(base_cfg["seed"])
    device = resolve_device(base_cfg)
    print(f"Using device: {device}")

    loaders = get_dataloaders(base_cfg)

    ckpt_path = os.path.join(base_cfg["output_dir"], "models", "task1_base_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No cached Task-1 model found at {ckpt_path}. "
            "Run scripts/run_finetune.py (or any other run_*.py script) at least once first."
        )

    base_model = MLPClassifier(
        input_dim=base_cfg["input_dim"],
        hidden1_dim=base_cfg["hidden1_dim"],
        hidden2_dim=base_cfg["hidden2_dim"],
        num_classes=base_cfg["num_classes"],
    ).to(device)
    base_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded cached Task-1 baseline model from {ckpt_path}")

    # Build D_base / base_point_cloud ONCE -- shared across every lambda
    # run in the sweep, so lambda is the only variable that changes.
    print("Building shared D_base (baseline persistence diagram) for the sweep...")
    base_point_cloud, diagram_base = build_baseline_artifacts(base_cfg, base_model, loaders, device)
    print(f"D_base ready (point cloud shape={base_point_cloud.shape})")

    results = []
    log_dir = os.path.join(base_cfg["output_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)

    for lambda_val in lambda_values:
        print(f"\n{'=' * 70}\nSweep run: tmp.lambda_ = {lambda_val}\n{'=' * 70}")

        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["tmp"]["lambda_"] = lambda_val

        model = copy.deepcopy(base_model)
        log_path = os.path.join(log_dir, f"sweep_tmp_lambda_{lambda_val}.jsonl")
        logger = JsonlLogger(log_path)

        _, history = train_task2(
            run_cfg, model, loaders, device, logger, method="tmp",
            base_point_cloud=base_point_cloud, diagram_base=diagram_base,
        )

        final = history[-1]
        results.append({
            "lambda": lambda_val,
            "final_retention_accuracy": final["retention_accuracy"],
            "final_retention_accuracy_task_incremental": final["retention_accuracy_task_incremental"],
            "final_learning_accuracy": final["learning_accuracy"],
            "final_drift_w_inf": final["feature_space_drift_w_inf"],
            "mean_drift_w_inf": sum(
                r["feature_space_drift_w_inf"] for r in history
                if r["feature_space_drift_w_inf"] is not None
            ) / len(history),
        })

    # ---- Summary table ----
    print(f"\n{'=' * 90}")
    print(f"{'lambda':>8} | {'retention(class-inc)':>20} | {'retention(task-inc)':>20} | "
          f"{'learning_acc':>12} | {'final_drift':>12} | {'mean_drift':>12}")
    print("-" * 90)
    for r in results:
        print(f"{r['lambda']:>8.2f} | {r['final_retention_accuracy']:>20.4f} | "
              f"{r['final_retention_accuracy_task_incremental']:>20.4f} | "
              f"{r['final_learning_accuracy']:>12.4f} | "
              f"{r['final_drift_w_inf']:>12.4f} | {r['mean_drift_w_inf']:>12.4f}")
    print("=" * 90)

    results_path = os.path.join(log_dir, "tmp_lambda_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to: {results_path}")

    # ---- Plot: retention & drift vs. lambda ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lambdas = [r["lambda"] for r in results]
        retentions = [r["final_retention_accuracy"] for r in results]
        drifts = [r["final_drift_w_inf"] for r in results]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "tab:blue"
        ax1.set_xlabel("tmp.lambda_")
        ax1.set_ylabel("Final Retention Accuracy (class-incremental)", color=color1)
        ax1.plot(lambdas, retentions, marker="o", color=color1, label="Retention Accuracy")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:red"
        ax2.set_ylabel("Final Feature Space Drift (W_inf)", color=color2)
        ax2.plot(lambdas, drifts, marker="s", color=color2, linestyle="--", label="Drift")
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("TMP: Retention Accuracy & Drift vs. Lambda")
        fig.tight_layout()

        plot_dir = os.path.join(base_cfg["output_dir"], "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "tmp_lambda_sweep.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Sweep plot written to: {plot_path}")
    except Exception as e:
        print(f"(Plot generation skipped due to error: {e})")


if __name__ == "__main__":
    main()
