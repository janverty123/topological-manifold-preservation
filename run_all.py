"""
run_all.py
----------
Convenience orchestrator: runs Task-1 pretraining (once, cached),
then Finetune, EWC, and TMP Task-2 runs in sequence, then the final
comparison/plotting step. Equivalent to running the individual scripts
one after another.

Usage:
    python run_all.py --config configs/config.yaml
"""

import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n{'=' * 70}\n>>> {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/run_finetune.py", "--config", args.config])
    run([py, "scripts/run_ewc.py", "--config", args.config])
    run([py, "scripts/run_tmp.py", "--config", args.config])
    run([py, "compare_results.py", "--config", args.config])


if __name__ == "__main__":
    main()
