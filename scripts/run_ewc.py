"""
run_ewc.py
----------
Runs the Elastic Weight Consolidation (EWC) baseline referenced in the
Simulation section's comparative analysis.

Usage:
    python scripts/run_ewc.py --config configs/baseline/split_mnist_2task.yaml
"""

import argparse
import os

import torch

from _common import bootstrap, fresh_copy
from src.train import train_task2, JsonlLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/split_mnist_2task.yaml")
    args = parser.parse_args()

    cfg, device, loaders, base_model = bootstrap(args.config)
    model = fresh_copy(base_model)

    logger = JsonlLogger(os.path.join(cfg["output_dir"], "logs", "ewc.jsonl"))
    model, history = train_task2(cfg, model, loaders, device, logger, method="ewc")

    ckpt_path = os.path.join(cfg["output_dir"], "models", "ewc_final_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"EWC run complete. Model saved to {ckpt_path}")
    print(f"Log written to {os.path.join(cfg['output_dir'], 'logs', 'ewc.jsonl')}")


if __name__ == "__main__":
    main()
