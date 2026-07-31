"""
sweep_tmp_lambda_multitask.py
--------------------------------
Recommendation 4: adapts sweep_tmp_lambda.py (built for the 2-task
pipeline) to the N-task multi-task pipeline (train_general.py).

Runs TMP across several `tmp.lambda_` values (including a 0.0
ablation), reusing the SAME cached Task-0 baseline model for every run
so lambda is the only thing that varies.

Usage:
    python sweep_tmp_lambda_multitask.py --config configs/config_split_mnist_5task.yaml
    python sweep_tmp_lambda_multitask.py --config configs/config_split_mnist_5task.yaml \
        --lambdas 0.0,2.5,5.0,7.5,10.0,15.0
"""

import argparse
import copy
import json
import os

import torch
import yaml

from src.datasets_extended import build_tasks
from src.models import MLPClassifier
from src.train_general import pretrain_first_task, train_continual_multitask, JsonlLogger


def resolve_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lambdas", default="0.0,2.5,5.0,7.5,10.0,15.0",
                         help="Comma-separated list of tmp.lambda_ values to sweep.")
    args = parser.parse_args()

    lambda_values = [float(x) for x in args.lambdas.split(",")]

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    torch.manual_seed(base_cfg["seed"])
    device = resolve_device(base_cfg)
    print(f"Using device: {device}")

    tasks = build_tasks(base_cfg["dataset"], base_cfg["data_dir"], base_cfg["num_tasks"], seed=base_cfg["seed"])
    print(f"Built {len(tasks)} tasks.")

    os.makedirs(os.path.join(base_cfg["output_dir"], "models"), exist_ok=True)
    os.makedirs(os.path.join(base_cfg["output_dir"], "logs"), exist_ok=True)

    ckpt_path = os.path.join(base_cfg["output_dir"], "models", "task0_base_model.pt")
    base_model = MLPClassifier(
        input_dim=base_cfg["input_dim"], hidden1_dim=base_cfg["hidden1_dim"],
        hidden2_dim=base_cfg["hidden2_dim"], num_classes=base_cfg["num_classes"],
    ).to(device)

    if os.path.exists(ckpt_path):
        print(f"Loading cached Task-0 baseline model from {ckpt_path}")
        base_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        logger0 = JsonlLogger(os.path.join(base_cfg["output_dir"], "logs", "task0_pretrain.jsonl"))
        base_model = pretrain_first_task(
            base_cfg, base_model, tasks, device, logger0,
            target_acc=base_cfg["task1_target_accuracy"],
            max_epochs=base_cfg["task1_max_epochs"],
            lr=base_cfg["task1_lr"],
        )
        torch.save(base_model.state_dict(), ckpt_path)
        print(f"Saved Task-0 baseline model to {ckpt_path}")

    results = []
    log_dir = os.path.join(base_cfg["output_dir"], "logs")

    for lambda_val in lambda_values:
        print(f"\n{'=' * 70}\nSweep run: tmp.lambda_ = {lambda_val}\n{'=' * 70}")

        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["tmp"]["lambda_"] = lambda_val

        model = copy.deepcopy(base_model)
        log_path = os.path.join(log_dir, f"mt_sweep_tmp_lambda_{lambda_val}.jsonl")
        logger = JsonlLogger(log_path)

        _, history = train_continual_multitask(run_cfg, model, tasks, device, logger, method="tmp")

        final = history[-1]
        results.append({
            "lambda": lambda_val,
            "final_avg_retention_accuracy": final["avg_retention_accuracy"],
            "final_learning_accuracy": final["learning_accuracy"],
            "final_drift_w_inf": final["feature_space_drift_w_inf"],
            "mean_avg_retention_accuracy": sum(r["avg_retention_accuracy"] for r in history) / len(history),
            "mean_learning_accuracy": sum(r["learning_accuracy"] for r in history) / len(history),
        })

    print(f"\n{'=' * 100}")
    print(f"{'lambda':>8} | {'final_retention':>16} | {'final_learning':>16} | "
          f"{'mean_retention':>16} | {'mean_learning':>16}")
    print("-" * 100)
    for r in results:
        print(f"{r['lambda']:>8.2f} | {r['final_avg_retention_accuracy']:>16.4f} | "
              f"{r['final_learning_accuracy']:>16.4f} | {r['mean_avg_retention_accuracy']:>16.4f} | "
              f"{r['mean_learning_accuracy']:>16.4f}")
    print("=" * 100)

    results_path = os.path.join(log_dir, "mt_tmp_lambda_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to: {results_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lambdas = [r["lambda"] for r in results]
        retentions = [r["final_avg_retention_accuracy"] for r in results]
        learnings = [r["final_learning_accuracy"] for r in results]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        color1 = "tab:blue"
        ax1.set_xlabel("tmp.lambda_")
        ax1.set_ylabel("Final Avg. Retention Accuracy", color=color1)
        ax1.plot(lambdas, retentions, marker="o", color=color1, label="Retention")
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "tab:orange"
        ax2.set_ylabel("Final Learning Accuracy", color=color2)
        ax2.plot(lambdas, learnings, marker="s", color=color2, linestyle="--", label="Learning")
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("Multi-Task TMP: Retention & Learning vs. Lambda")
        fig.tight_layout()

        plot_dir = os.path.join(base_cfg["output_dir"], "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "mt_tmp_lambda_sweep.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Sweep plot written to: {plot_path}")
    except Exception as e:
        print(f"(Plot generation skipped due to error: {e})")


if __name__ == "__main__":
    main()
