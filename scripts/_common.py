"""
_common.py
----------
Shared setup logic for scripts/run_finetune.py, run_ewc.py, run_tmp.py.
Not meant to be run directly.
"""

import copy
import os
import sys

import torch
import yaml

# allow `python scripts/run_xxx.py` to import the `src` package from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_dataloaders
from src.models import MLPClassifier
from src.train import set_seed, pretrain_task1, JsonlLogger


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def bootstrap(cfg_path="configs/config.yaml"):
    """
    Loads config, sets seeds, builds dataloaders, and pretrains (or loads
    a cached) Task-1 model + baseline artifacts. Returns everything the
    three run_*.py scripts need.
    """
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg)
    print(f"Using device: {device}")

    loaders = get_dataloaders(cfg)

    ckpt_path = os.path.join(cfg["output_dir"], "models", "task1_base_model.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    model = MLPClassifier(
        input_dim=cfg["input_dim"],
        hidden1_dim=cfg["hidden1_dim"],
        hidden2_dim=cfg["hidden2_dim"],
        num_classes=cfg["num_classes"],
    ).to(device)

    logger = JsonlLogger(os.path.join(cfg["output_dir"], "logs", "task1_pretrain.jsonl"))

    if os.path.exists(ckpt_path):
        print(f"Loading cached Task-1 baseline model from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        model = pretrain_task1(cfg, model, loaders, device, logger)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved Task-1 baseline model to {ckpt_path}")

    return cfg, device, loaders, model


def fresh_copy(model):
    """Returns a deep copy of the pretrained Task-1 model so each method
    (finetune/ewc/tmp) starts Task 2 from an identical baseline state."""
    return copy.deepcopy(model)
