"""
run_finetune.py
----------------
Runs the "baseline Finetune script with no memory protections" referenced
in the Simulation section's comparative analysis.

Usage:
    python scripts/run_finetune.py --config configs/config.yaml
"""

import argparse
import os

from _common import bootstrap, fresh_copy
from src.train import train_task2, JsonlLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg, device, loaders, base_model = bootstrap(args.config)
    model = fresh_copy(base_model)

    logger = JsonlLogger(os.path.join(cfg["output_dir"], "logs", "finetune.jsonl"))
    model, history = train_task2(cfg, model, loaders, device, logger, method="finetune")

    ckpt_path = os.path.join(cfg["output_dir"], "models", "finetune_final_model.pt")
    import torch
    torch.save(model.state_dict(), ckpt_path)
    print(f"Finetune run complete. Model saved to {ckpt_path}")
    print(f"Log written to {os.path.join(cfg['output_dir'], 'logs', 'finetune.jsonl')}")


if __name__ == "__main__":
    main()
