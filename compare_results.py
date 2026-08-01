"""
compare_results.py
-------------------
Final step of the Simulation section: statistical comparison + plot
generation across Finetune, EWC, and TMP.

Run this AFTER scripts/run_finetune.py, scripts/run_ewc.py, and
scripts/run_tmp.py have all completed.

Usage:
    python compare_results.py --config configs/baseline/split_mnist_2task.yaml
"""

import argparse
import os

import yaml

from src.evaluate import run_full_evaluation
from src.visualize import generate_all_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/split_mnist_2task.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log_dir = os.path.join(cfg["output_dir"], "logs")
    plot_dir = os.path.join(cfg["output_dir"], "plots")
    results_path = os.path.join(cfg["output_dir"], "logs", "comparison_results.json")

    df, results = run_full_evaluation(log_dir, results_path)

    from src.evaluate import summarize
    summary_df = summarize(df)
    generate_all_plots(df, summary_df, plot_dir)

    print(f"\nAll comparison plots written to: {plot_dir}")
    print(f"Full statistical results written to: {results_path}")


if __name__ == "__main__":
    main()
