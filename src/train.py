"""
train.py
--------
Implements the "Simulation" section of the methodology end to end:

  1. Train on digits 0-4 until >= 95% accuracy (baseline mastery
     threshold) -> save D_base (persistence diagram of hidden2
     activations).
  2. Continue training on digits 5-9 for 10 epochs under one of three
     regimes: {finetune, ewc, tmp}.
  3. At the end of every Task-2 epoch: extract a live activation point
     cloud, build D_current, compute true W_inf(D_base, D_current), and
     log Retention Accuracy / Learning Rate Efficiency / Computational
     Overhead (Simulation section, four primary metrics).
"""

import copy
import json
import os
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import MLPClassifier, apply_class_mask
from src.tda_utils import build_activation_point_cloud, compute_persistence_diagram, bottleneck_distance
from src.losses import masked_cross_entropy, tmp_total_loss
from src.ewc import EWC


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate_accuracy(model, loader, allowed_classes, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            logits = apply_class_mask(logits, allowed_classes, device)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
    return correct / max(total, 1)


def pretrain_task1(cfg, model, loaders, device, logger):
    """
    Trains the model on Task 1 (digits 0-4) until the baseline mastery
    threshold is reached, matching:
    'training the PyTorch network on digits 0 through 4 until
    classification accuracy reaches a baseline mastery threshold of at
    least 95%.'
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["task1_lr"])
    target_acc = cfg["task1_target_accuracy"]
    max_epochs = cfg["task1_max_epochs"]
    allowed = cfg["task1_classes"]

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in loaders["task1_train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_cross_entropy(logits, y, allowed, device)
            loss.backward()
            optimizer.step()

        acc = evaluate_accuracy(model, loaders["task1_test"], allowed, device)
        logger.log({"phase": "task1_pretrain", "epoch": epoch, "task1_test_acc": acc})
        print(f"[Task1 Pretrain] epoch={epoch} task1_test_acc={acc:.4f}")
        if acc >= target_acc:
            print(f"[Task1 Pretrain] Reached baseline mastery threshold ({target_acc}). Stopping.")
            break

    return model


def build_baseline_artifacts(cfg, model, loaders, device):
    """
    Captures D_base: extracts the hidden2 activation cloud on Task-1
    data at peak mastery, Maxmin-samples it to `point_cloud_size`
    points, and computes the baseline Persistence Diagram.

    Returns:
        base_point_cloud (np.ndarray), diagram_base (gtda diagram array)
    """
    n_points = cfg["tda"]["point_cloud_size"]
    base_point_cloud = build_activation_point_cloud(
        model, loaders["task1_train"], device, n_points=n_points, seed=cfg["seed"]
    )
    diagram_base = compute_persistence_diagram(
        base_point_cloud,
        homology_dims=cfg["tda"]["homology_dims"],
        max_edge_length=cfg["tda"]["max_edge_length"],
    )
    return base_point_cloud, diagram_base


def train_task2(cfg, model, loaders, device, logger, method: str,
                 base_point_cloud=None, diagram_base=None):
    """
    Runs the Task-2 continual-learning phase under the requested method.

    method in {"finetune", "ewc", "tmp"}
    """
    assert method in ("finetune", "ewc", "tmp")

    allowed_task2 = cfg["task2_classes"]
    allowed_task1 = cfg["task1_classes"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["task2_lr"])

    ewc_reg = None
    if method == "ewc":
        ewc_reg = EWC(model, loaders["task1_train"], device, allowed_task1,
                      sample_size=cfg["ewc"]["fisher_sample_size"])

    baseline_ref_tensor = None
    lambda_current = cfg["tmp"]["lambda_"]
    if method == "tmp":
        baseline_ref_tensor = torch.tensor(base_point_cloud, dtype=torch.float32, device=device)

    history = []
    global_step = 0

    for epoch in range(1, cfg["task2_epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running_ce = 0.0
        running_extra = 0.0
        n_batches = 0

        for x, y in loaders["task2_train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)

            if method == "finetune":
                loss = masked_cross_entropy(logits, y, allowed_task2, device)
                ce_val, extra_val = loss.detach().item(), 0.0

            elif method == "ewc":
                ce_loss = masked_cross_entropy(logits, y, allowed_task2, device)
                penalty = ewc_reg.penalty(model)
                loss = ce_loss + cfg["ewc"]["lambda_"] * penalty
                ce_val, extra_val = ce_loss.detach().item(), penalty.detach().item()

            else:  # tmp
                hidden_act = model.get_last_hidden_activation()
                loss, ce_val, extra_val = tmp_total_loss(
                    logits, y, allowed_task2, device,
                    hidden_act, baseline_ref_tensor,
                    lambda_current,
                )

            loss.backward()
            optimizer.step()

            running_ce += ce_val
            running_extra += extra_val
            n_batches += 1
            global_step += 1

            if global_step % cfg["log_every_n_steps"] == 0:
                logger.log({
                    "phase": f"task2_{method}",
                    "epoch": epoch,
                    "step": global_step,
                    "ce_loss": ce_val,
                    "extra_term": extra_val,
                })

        epoch_time = time.time() - epoch_start
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

        # ---- Four primary metrics (Simulation section) ----
        retention_acc = evaluate_accuracy(model, loaders["task1_test"], allowed_task1, device)
        learning_acc = evaluate_accuracy(model, loaders["task2_test"], allowed_task2, device)

        drift_w_inf = None
        if method == "tmp" and diagram_base is not None:
            current_cloud = build_activation_point_cloud(
                model, loaders["task1_train"], device,
                n_points=cfg["tda"]["point_cloud_size"], seed=cfg["seed"] + epoch,
            )
            diagram_current = compute_persistence_diagram(
                current_cloud,
                homology_dims=cfg["tda"]["homology_dims"],
                max_edge_length=cfg["tda"]["max_edge_length"],
            )
            drift_w_inf = bottleneck_distance(diagram_base, diagram_current,
                                               delta=cfg["tda"]["bottleneck_delta"])
            # Adaptive rescaling of lambda for the NEXT epoch using the
            # true measured drift (see losses.py docstring).
            lambda_current = cfg["tmp"]["lambda_"] * (1.0 + drift_w_inf)

        record = {
            "phase": f"task2_{method}",
            "epoch": epoch,
            "retention_accuracy": retention_acc,
            "learning_accuracy": learning_acc,
            "avg_ce_loss": running_ce / max(n_batches, 1),
            "avg_extra_term": running_extra / max(n_batches, 1),
            "epoch_time_sec": epoch_time,
            "memory_mb": mem_mb,
            "feature_space_drift_w_inf": drift_w_inf,
        }
        history.append(record)
        logger.log(record)
        print(f"[Task2 {method}] epoch={epoch} retention_acc={retention_acc:.4f} "
              f"learning_acc={learning_acc:.4f} drift={drift_w_inf} time={epoch_time:.2f}s")

    return model, history


class JsonlLogger:
    """Simple append-only JSON-Lines logger, one record per line."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        # truncate at the start of each run
        open(self.path, "w").close()

    def log(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
