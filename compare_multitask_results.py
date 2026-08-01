"""
compare_multitask_results.py
------------------------------
Comparison and statistical analysis for the N-task (e.g. 5-task
Split-MNIST) continual learning logs, separate from compare_results.py
(which is for the original 2-task format).

Auto-detects whichever of finetune.jsonl / ewc.jsonl / tmp.jsonl are
present in the log directory -- works with just 2 of the 3 methods if
that's all you've run so far.

Usage:
    python compare_multitask_results.py --log-dir outputs/experiments/EXP-002/logs --plot-dir outputs/experiments/EXP-002/plots
"""

import argparse
import os

from src.evaluate_multitask import run_full_multitask_evaluation, summarize, summarize_final_epoch_per_task
from src.visualize_multitask import generate_all_multitask_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True,
                         help="Directory containing finetune.jsonl / ewc.jsonl / tmp.jsonl")
    parser.add_argument("--plot-dir", default=None,
                         help="Directory to write plots to (default: <log-dir>/../plots)")
    parser.add_argument("--results-name", default="mt_comparison_results.json",
                         help="Filename for the JSON results summary, written into --log-dir")
    args = parser.parse_args()

    plot_dir = args.plot_dir or os.path.join(os.path.dirname(args.log_dir.rstrip("/")), "plots")
    results_path = os.path.join(args.log_dir, args.results_name)

    df, results = run_full_multitask_evaluation(args.log_dir, results_path)
    final_per_task = summarize_final_epoch_per_task(df)

    generate_all_multitask_plots(df, final_per_task, plot_dir)

    print(f"\nAll comparison plots written to: {plot_dir}")
    print(f"Full statistical results written to: {results_path}")


if __name__ == "__main__":
    main()
