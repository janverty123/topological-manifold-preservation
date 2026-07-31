"""
run_multitask.py
------------------
Runs N-task sequential continual learning (5-task Split-MNIST) under
Finetune, EWC, or TMP.

Usage:
    python run_multitask.py --config configs/config_split_mnist_5task.yaml --method finetune
    python run_multitask.py --config configs/config_split_mnist_5task.yaml --method ewc
    python run_multitask.py --config configs/config_split_mnist_5task.yaml --method tmp
"""

import argparse
import os

import torch
import yaml

from src.datasets_extended import build_tasks
from src.models import MLPClassifier
from src.train_general import (
    pretrain_first_task,
    train_continual_multitask,
    JsonlLogger,
)


def resolve_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg):
    return MLPClassifier(
        input_dim=cfg["input_dim"],
        hidden1_dim=cfg["hidden1_dim"],
        hidden2_dim=cfg["hidden2_dim"],
        num_classes=cfg["num_classes"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=["finetune", "ewc", "tmp"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = resolve_device(cfg)
    print(f"Using device: {device}")
    print(f"Dataset: {cfg['dataset']} | num_tasks: {cfg['num_tasks']} | architecture: {cfg['architecture']}")

    tasks = build_tasks(cfg["dataset"], cfg["data_dir"], cfg["num_tasks"], seed=cfg["seed"])
    print(f"Built {len(tasks)} tasks.")

    os.makedirs(os.path.join(cfg["output_dir"], "models"), exist_ok=True)
    os.makedirs(os.path.join(cfg["output_dir"], "logs"), exist_ok=True)

    ckpt_path = os.path.join(cfg["output_dir"], "models", "task0_base_model.pt")
    model = build_model(cfg).to(device)

    if os.path.exists(ckpt_path):
        print(f"Loading cached Task-0 baseline model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        logger0 = JsonlLogger(os.path.join(cfg["output_dir"], "logs", "task0_pretrain.jsonl"))
        model = pretrain_first_task(
            cfg, model, tasks, device, logger0,
            target_acc=cfg["task1_target_accuracy"],
            max_epochs=cfg["task1_max_epochs"],
            lr=cfg["task1_lr"],
        )
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved Task-0 baseline model to {ckpt_path}")

    import copy
    model = copy.deepcopy(model)  # fresh copy so repeated --method runs share the same Task-0 checkpoint

    logger = JsonlLogger(os.path.join(cfg["output_dir"], "logs", f"{args.method}.jsonl"))
    model, history = train_continual_multitask(cfg, model, tasks, device, logger, method=args.method)

    final_ckpt = os.path.join(cfg["output_dir"], "models", f"{args.method}_final_model.pt")
    torch.save(model.state_dict(), final_ckpt)

    print(f"\n{args.method} run complete.")
    print(f"Final model saved to: {final_ckpt}")
    print(f"Log written to: {os.path.join(cfg['output_dir'], 'logs', args.method + '.jsonl')}")
    if history:
        print(f"Final avg_retention_accuracy across all {len(tasks) - 1} continual tasks: "
              f"{history[-1]['avg_retention_accuracy']:.4f}")


if __name__ == "__main__":
    main()
