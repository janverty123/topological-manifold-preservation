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
    """
    Task-incremental evaluation: the model is only allowed to choose
    among `allowed_classes` at test time (logits for all other classes
    are masked to -inf). This tests whether the model still ranks the
    Task-1 digits correctly RELATIVE TO EACH OTHER, but does NOT test
    whether it has confused Task-1 digits with Task-2 digits -- so it
    substantially UNDERSTATES catastrophic forgetting. Kept for
    comparison against `evaluate_accuracy_unmasked` below.
    """
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


def evaluate_accuracy_unmasked(model, loader, device):
    """
    Class-incremental evaluation: the model must choose the correct
    digit from ALL 10 classes, with no knowledge of which task the
    sample came from. This is the evaluation protocol that actually
    matches the research plan's Rationale ("the network ... ends up
    overwriting the important things it learned before") -- a model
    that has forgotten Task 1 will here start predicting Task-2 digits
    for Task-1 images, which `evaluate_accuracy`'s masking cannot detect.
    Use THIS metric as the primary Retention Accuracy figure for
    reporting catastrophic forgetting.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
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


def build_fixed_task1_reference(cfg, model, loaders, device):
    """
    Selects a FIXED subset of `tmp.surrogate_subsample` Task-1 images
    (the SAME images used at every training step) and captures their
    hidden2 activations under the just-mastered Task-1 model.

    WHY THIS MATTERS: the differentiable surrogate loss needs to compare
    "how does the current network represent THESE SPECIFIC images" vs.
    "how did the baseline network represent THESE SAME images". Earlier
    versions of this code instead compared pairwise-distance geometry
    between two DIFFERENT random samples of Task-1 images each step
    (the live mini-batch vs. a randomly-resampled slice of the baseline
    cloud) -- but two different random samples from the same digit
    distribution naturally have somewhat different pairwise-distance
    geometry from sampling noise alone, even with zero real drift. That
    noise dominated the signal: empirically, increasing lambda 20x
    barely changed the true measured W_inf but substantially hurt
    retention accuracy, indicating the gradient was fighting sampling
    noise rather than genuine representational drift. Using a FIXED,
    paired image set removes that confound entirely.
    """
    n_samples = cfg["tmp"]["surrogate_subsample"]
    subset = loaders["task1_train"].dataset
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(subset), size=min(n_samples, len(subset)), replace=False)
    images = torch.stack([subset[i][0] for i in idx]).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(images)
        baseline_activations = model.get_last_hidden_activation().clone()
    model.train(was_training)

    return images, baseline_activations


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
        mean_fisher = sum(f.sum().item() for f in ewc_reg.fisher.values()) / \
                      sum(f.numel() for f in ewc_reg.fisher.values())
        print(f"[EWC config] lambda_={cfg['ewc']['lambda_']} "
              f"fisher_sample_size={cfg['ewc']['fisher_sample_size']} "
              f"mean_fisher_value={mean_fisher:.3e}")

    fixed_task1_images = None
    fixed_baseline_activations = None
    lambda_current = cfg["tmp"]["lambda_"]
    if method == "tmp":
        fixed_task1_images, fixed_baseline_activations = build_fixed_task1_reference(
            cfg, model, loaders, device
        )
        print(f"[TMP config] lambda_={cfg['tmp']['lambda_']} "
              f"surrogate_subsample={fixed_task1_images.shape[0]} "
              f"point_cloud_size={cfg['tda']['point_cloud_size']} "
              f"base_point_cloud_shape={base_point_cloud.shape}")

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
                # Re-forward the FIXED Task-1 reference set (same images
                # every step) through the CURRENT network state to
                # measure genuine, paired representation drift (see
                # build_fixed_task1_reference docstring). This is a
                # second, independent forward pass; `logits` (from the
                # Task-2 batch `x`) is unaffected.
                _ = model(fixed_task1_images)
                hidden_act_task1_current = model.get_last_hidden_activation()

                loss, ce_val, extra_val = tmp_total_loss(
                    logits, y, allowed_task2, device,
                    hidden_act_task1_current, fixed_baseline_activations,
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
        # `retention_accuracy` is now the CLASS-INCREMENTAL (unmasked)
        # figure -- this is what actually demonstrates catastrophic
        # forgetting as described in the Rationale ("the network ...
        # ends up overwriting the important things it learned before").
        # `retention_accuracy_task_incremental` is kept alongside it for
        # reference/ablation, but should NOT be reported as the primary
        # forgetting metric since it structurally cannot detect
        # cross-task confusion (see evaluate_accuracy's docstring).
        retention_acc = evaluate_accuracy_unmasked(model, loaders["task1_test"], device)
        retention_acc_task_incremental = evaluate_accuracy(model, loaders["task1_test"], allowed_task1, device)
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
            # true measured drift (see losses.py docstring). Capped to
            # [1x, 3x] the base lambda: uncapped rescaling can create a
            # runaway feedback loop if a too-large penalty is itself
            # part of what's driving drift up (larger drift -> larger
            # lambda -> more drift next epoch -> even larger lambda...).
            # The cap keeps the adaptive term a mild nudge rather than a
            # potentially destabilizing multiplier.
            rescale_factor = min(1.0 + drift_w_inf, 3.0)
            lambda_current = cfg["tmp"]["lambda_"] * rescale_factor

        record = {
            "phase": f"task2_{method}",
            "epoch": epoch,
            "retention_accuracy": retention_acc,
            "retention_accuracy_task_incremental": retention_acc_task_incremental,
            "learning_accuracy": learning_acc,
            "avg_ce_loss": running_ce / max(n_batches, 1),
            "avg_extra_term": running_extra / max(n_batches, 1),
            "epoch_time_sec": epoch_time,
            "memory_mb": mem_mb,
            "feature_space_drift_w_inf": drift_w_inf,
        }
        history.append(record)
        logger.log(record)
        print(f"[Task2 {method}] epoch={epoch} retention_acc(class-inc)={retention_acc:.4f} "
              f"retention_acc(task-inc)={retention_acc_task_incremental:.4f} "
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
