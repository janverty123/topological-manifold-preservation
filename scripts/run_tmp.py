"""
run_tmp.py
----------
Runs the proposed Topological Manifold Preservation (TMP) algorithm:
extracts D_base after Task-1 mastery, then trains Task 2 under the
TMP loss (Mathematical Formulation, Section II) while monitoring the
true Bottleneck Distance W_inf(D_base, D_current) every epoch.

Usage:
    python scripts/run_tmp.py --config configs/baseline/split_mnist_2task.yaml
"""

import argparse
import json
import os

import numpy as np
import torch

from _common import bootstrap, fresh_copy
from src.train import train_task2, JsonlLogger, build_baseline_artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline/split_mnist_2task.yaml")
    args = parser.parse_args()

    cfg, device, loaders, base_model = bootstrap(args.config)
    model = fresh_copy(base_model)

    diagrams_dir = os.path.join(cfg["output_dir"], "diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)

    print("Building D_base (baseline persistence diagram) from Task-1 mastery state...")
    base_point_cloud, diagram_base = build_baseline_artifacts(cfg, model, loaders, device)
    np.save(os.path.join(diagrams_dir, "base_point_cloud.npy"), base_point_cloud)
    np.save(os.path.join(diagrams_dir, "diagram_base.npy"), diagram_base)
    print(f"D_base saved to {diagrams_dir} (point cloud shape={base_point_cloud.shape}, "
          f"diagram shape={diagram_base.shape})")

    logger = JsonlLogger(os.path.join(cfg["output_dir"], "logs", "tmp.jsonl"))
    model, history = train_task2(
        cfg, model, loaders, device, logger, method="tmp",
        base_point_cloud=base_point_cloud, diagram_base=diagram_base,
    )

    ckpt_path = os.path.join(cfg["output_dir"], "models", "tmp_final_model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"TMP run complete. Model saved to {ckpt_path}")
    print(f"Log written to {os.path.join(cfg['output_dir'], 'logs', 'tmp.jsonl')}")


if __name__ == "__main__":
    main()
