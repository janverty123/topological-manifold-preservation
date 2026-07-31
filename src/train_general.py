"""
train_general.py
-----------------
Generalizes the two-task (Task 1 / Task 2) Finetune / EWC / TMP logic
in train.py to an arbitrary number of SEQUENTIAL tasks, used here for
5-task Split-MNIST (src/datasets_extended.py).

Key differences from the two-task version:
  - EWC: accumulates a SEPARATE (Fisher, anchor-params) snapshot after
    every completed task, and penalizes drift from ALL of them
    (standard multi-task EWC; the two-task version in train.py is the
    N=2 special case of this).
  - TMP: maintains a GROWING fixed reference pool -- after each
    completed task, a fixed subsample of that task's images (plus
    their activations under the model at that moment) is added to the
    pool. The differentiable surrogate and the true W_inf metric are
    both computed against this whole accumulated pool, so "retention"
    means "retention across everything learned so far", not just the
    first task.
  - Retention is reported both as a per-previously-seen-task breakdown
    and as an average across all of them.

Improvements added on top of the initial working version, based on
diagnosing weak per-task protection (see README_MULTITASK.md):
  - AdaptiveLambdaScheduler: EMA-smoothed lambda instead of reacting to
    a single noisy epoch's drift measurement.
  - MultiTaskTMPReference.weighted_surrogate_loss: per-task-in-pool
    weighting, giving tasks with worse recently-measured retention a
    proportionally larger share of the protective gradient signal,
    instead of treating the whole pool as one flat average.
"""

import copy
import json
import os
import time

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from src.losses import masked_cross_entropy, topological_surrogate_loss
from src.tda_utils import maxmin_sample, normalize_point_cloud, compute_persistence_diagram, bottleneck_distance


# ---------------------------------------------------------------------
# Shared evaluation helpers
# ---------------------------------------------------------------------

def evaluate_accuracy_unmasked(model, loader, device):
    """Class-incremental (unmasked, full-head) accuracy -- see train.py's
    identically-named function for the full rationale."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.shape[0]
    return correct / max(total, 1)


def make_loader(dataset, batch_size, shuffle, num_workers=2):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False,
    )


# ---------------------------------------------------------------------
# Multi-task EWC: one (Fisher, anchor-params) snapshot per completed task
# ---------------------------------------------------------------------

class MultiTaskEWC:
    def __init__(self):
        self.snapshots = []  # list of {"fisher": {...}, "params": {...}}

    def add_snapshot(self, model, task_loader, device, allowed_classes, sample_size):
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        model.eval()

        mask = torch.full((model.fc3.out_features,), float("-inf"), device=device)
        mask[allowed_classes] = 0.0

        seen = 0
        for x, y in task_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            logits = model(x) + mask.unsqueeze(0)
            log_probs = F.log_softmax(logits, dim=1)
            sampled = torch.multinomial(log_probs.exp(), 1).squeeze(1)
            loss = F.nll_loss(log_probs, sampled)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * x.shape[0]
            seen += x.shape[0]
            if seen >= sample_size:
                break
        for n in fisher:
            fisher[n] /= max(seen, 1)

        self.snapshots.append({"fisher": fisher, "params": params})

    def penalty(self, model):
        if not self.snapshots:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        total = 0.0
        for snap in self.snapshots:
            for n, p in model.named_parameters():
                if n in snap["fisher"]:
                    total = total + (snap["fisher"][n] * (p - snap["params"][n]) ** 2).sum()
        return total


class AdaptiveLambdaScheduler:
    """
    Recommendation 1: EMA-smoothed adaptive lambda schedule.

    Replaces the raw, noisy per-epoch rescaling
    (`lambda_t = base * min(1 + drift_t, cap)`) with an Exponential
    Moving Average of drift, so a single noisy epoch's drift spike
    doesn't yank lambda around -- the schedule reacts to the TREND in
    drift, not the latest single measurement.

        d_bar_t = alpha * d_t + (1 - alpha) * d_bar_{t-1}
        lambda_t = base_lambda * (1 + clamp(d_bar_t, 0, max_multiplier - 1))
    """

    def __init__(self, base_lambda: float, alpha: float = 0.2, max_multiplier: float = 3.0):
        self.base_lambda = base_lambda
        self.alpha = alpha
        self.max_multiplier = max_multiplier
        self.ema_drift = None

    def update_and_get_lambda(self, current_drift: float) -> float:
        if self.ema_drift is None:
            self.ema_drift = current_drift
        else:
            self.ema_drift = self.alpha * current_drift + (1 - self.alpha) * self.ema_drift

        multiplier = min(1.0 + max(self.ema_drift, 0.0), self.max_multiplier)
        return self.base_lambda * multiplier


# ---------------------------------------------------------------------
# Multi-task TMP: growing fixed reference pool across completed tasks
# ---------------------------------------------------------------------

class MultiTaskTMPReference:
    """
    Growing fixed reference pool across completed tasks, with
    per-task-segment tracking (Recommendation 3) so the surrogate loss
    can be computed and weighted PER TASK rather than as one flat pool
    average -- letting tasks with poor retention receive a stronger
    protective pull than tasks that are already well-preserved.
    """

    def __init__(self, per_task_samples):
        self.per_task_samples = per_task_samples
        self.images = None       # (total_N, ...) fixed images from all completed tasks
        self.baseline_acts = None  # (total_N, hidden_dim) their activations at the time each task finished
        self.segments = []       # list of (task_id, start_idx, end_idx), in add order
        self.latest_accuracy = {}  # task_id -> most recently measured retention accuracy

    def add_task(self, model, task_dataset, device, seed, task_id):
        rng = np.random.default_rng(seed)
        n = min(self.per_task_samples, len(task_dataset))
        idx = rng.choice(len(task_dataset), size=n, replace=False)
        new_images = torch.stack([task_dataset[i][0] for i in idx]).to(device)

        was_training = model.training
        model.eval()
        with torch.no_grad():
            _ = model(new_images)
            new_acts = model.get_last_hidden_activation().clone()
        model.train(was_training)

        start_idx = 0 if self.images is None else self.images.shape[0]
        self.images = new_images if self.images is None else torch.cat([self.images, new_images], dim=0)
        self.baseline_acts = new_acts if self.baseline_acts is None else torch.cat([self.baseline_acts, new_acts], dim=0)
        self.segments.append((task_id, start_idx, start_idx + n))
        # A task is assumed perfectly retained the moment it's added
        # (it was just mastered) -- updated with real measurements as
        # training on subsequent tasks proceeds.
        self.latest_accuracy[task_id] = 1.0

    def update_accuracy(self, task_id, accuracy):
        self.latest_accuracy[task_id] = accuracy

    def current_activations(self, model):
        _ = model(self.images)
        return model.get_last_hidden_activation()

    @staticmethod
    def _cap_and_redistribute(weights: dict, cap: float, iters: int = 5) -> dict:
        """
        Caps every weight at `cap` and redistributes the excess
        proportionally among the tasks still under the cap, iterating
        in case redistribution itself pushes another task over the cap
        (only relevant with very few tasks in the pool). Weights always
        sum to ~1.0 after this.
        """
        w = dict(weights)
        for _ in range(iters):
            over = {k: v for k, v in w.items() if v > cap}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            for k in over:
                w[k] = cap
            under = {k: v for k, v in w.items() if v < cap}
            under_total = sum(under.values()) + 1e-8
            for k in under:
                w[k] += excess * (under[k] / under_total)
        return w

    def weighted_surrogate_loss(self, model, max_weight_cap=0.5):
        """
        Recommendation 3: task-prioritized weighted surrogate loss.

            L_surrogate_total = sum_k  w_k * L_surrogate(task k)
            w_k = (1 - acc_k) / sum_j (1 - acc_j), capped and
            redistributed so no single task can exceed `max_weight_cap`.

        Tasks with lower recently-measured retention accuracy get a
        proportionally larger share of the protective gradient signal.
        A task at 100% retention contributes near-zero weight; a task
        that's collapsed toward 0% would otherwise dominate the penalty.

        WITHOUT the cap, one severely-underperforming task can consume
        >90% of the ENTIRE protective gradient every single step --
        verified directly: task1 at 40% accuracy alongside two tasks at
        95%+ pulled 90.9% of all weight with the original uncapped
        formula. That creates a persistent, heavily concentrated pull
        on the shared hidden layers every step, consistent with the
        observed symptom of new-task learning_accuracy getting
        completely stuck near 0 regardless of lambda's magnitude,
        rather than a smooth strength-based tradeoff. A simple additive
        floor was tried first and found too weak to matter when the
        accuracy gap is large (barely moved 90.9% -> 85.7%); a hard cap
        with proportional redistribution among the remaining tasks
        actually bounds the concentration (verified: 90.9% -> 50.0% for
        the same example).
        """
        current_acts = self.current_activations(model)

        unstable_weights = {}
        for task_id, _, _ in self.segments:
            acc = self.latest_accuracy.get(task_id, 1.0)
            unstable_weights[task_id] = max(1.0 - acc, 0.0)
        total_weight = sum(unstable_weights.values()) + 1e-8
        weights = {tid: w / total_weight for tid, w in unstable_weights.items()}
        weights = self._cap_and_redistribute(weights, max_weight_cap)

        total_surrogate = 0.0
        per_task_terms = {}
        for task_id, start, end in self.segments:
            weight = weights[task_id]
            seg_loss = topological_surrogate_loss(current_acts[start:end], self.baseline_acts[start:end])
            total_surrogate = total_surrogate + weight * seg_loss
            per_task_terms[task_id] = seg_loss.detach().item()

        return total_surrogate, per_task_terms


def build_true_drift_diagram(model, reference: MultiTaskTMPReference, device, point_cloud_size, homology_dims):
    """
    Builds a persistence diagram from the CURRENT model's activations
    on the (fixed) reference pool, Maxmin-sampled and scale-normalized
    (see tda_utils.normalize_point_cloud), for computing the true
    W_inf drift metric.
    """
    with torch.no_grad():
        acts = reference.current_activations(model).cpu().numpy()
    sampled = maxmin_sample(acts, min(point_cloud_size, acts.shape[0]))
    normalized = normalize_point_cloud(sampled)
    return compute_persistence_diagram(normalized, homology_dims=homology_dims)


# ---------------------------------------------------------------------
# Main sequential N-task training loop
# ---------------------------------------------------------------------

def pretrain_first_task(cfg, model, tasks, device, logger, target_acc=0.95, max_epochs=30, lr=0.001):
    task0 = tasks[0]
    loader = make_loader(task0["train"], cfg["batch_size"], shuffle=True)
    test_loader = make_loader(task0["test"], cfg["eval_batch_size"], shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_cross_entropy(logits, y, task0["classes"], device)
            loss.backward()
            optimizer.step()
        acc = evaluate_accuracy_unmasked(model, test_loader, device)
        logger.log({"phase": "task0_pretrain", "epoch": epoch, "task0_test_acc": acc})
        print(f"[Task0 Pretrain] epoch={epoch} task0_test_acc={acc:.4f}")
        if acc >= target_acc:
            print(f"[Task0 Pretrain] Reached mastery threshold ({target_acc}). Stopping.")
            break
    return model


def train_continual_multitask(cfg, model, tasks, device, logger, method: str):
    """
    Sequentially trains on tasks[1:], evaluating retention across ALL
    previously-seen tasks (tasks[0..i-1]) after every epoch of task i.
    `tasks[0]` must already be mastered by the model BEFORE calling this
    (see pretrain_first_task).

    method in {"finetune", "ewc", "tmp"}
    """
    assert method in ("finetune", "ewc", "tmp")

    ewc = MultiTaskEWC() if method == "ewc" else None
    tmp_ref = MultiTaskTMPReference(cfg["tmp"]["surrogate_subsample"]) if method == "tmp" else None
    lambda_scheduler = AdaptiveLambdaScheduler(
        base_lambda=cfg["tmp"]["lambda_"],
        alpha=cfg["tmp"].get("ema_alpha", 0.2),
        max_multiplier=cfg["tmp"].get("lambda_max_multiplier", 3.0),
    ) if method == "tmp" else None
    lambda_current = cfg["tmp"]["lambda_"] if method == "tmp" else None

    # Give the regularizer its first "protected" snapshot from task 0
    # (the just-pretrained model), before any Task-2+ training happens.
    if method == "ewc":
        loader0 = make_loader(tasks[0]["train"], cfg["batch_size"], shuffle=True)
        ewc.add_snapshot(model, loader0, device, tasks[0]["classes"], cfg["ewc"]["fisher_sample_size"])
    if method == "tmp":
        tmp_ref.add_task(model, tasks[0]["train"], device, seed=cfg["seed"], task_id=0)

    diagram_prev = None
    if method == "tmp":
        diagram_prev = build_true_drift_diagram(
            model, tmp_ref, device, cfg["tda"]["point_cloud_size"], cfg["tda"]["homology_dims"]
        )

    history = []
    global_step = 0  # cumulative across ALL tasks -- used for intermittent surrogate application

    for task_idx in range(1, len(tasks)):
        task = tasks[task_idx]
        train_loader = make_loader(task["train"], cfg["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["task2_lr"])

        print(f"\n{'#' * 70}\nTraining on task {task_idx} ({len(task['classes'])} classes) "
              f"under method='{method}'\n{'#' * 70}")

        for epoch in range(1, cfg["task2_epochs"] + 1):
            model.train()
            epoch_start = time.time()
            running_ce, running_extra, n_batches = 0.0, 0.0, 0
            running_grad_norm, max_grad_norm_seen = 0.0, 0.0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)

                if method == "finetune":
                    loss = masked_cross_entropy(logits, y, task["classes"], device)
                    ce_val, extra_val = loss.detach().item(), 0.0

                elif method == "ewc":
                    ce_loss = masked_cross_entropy(logits, y, task["classes"], device)
                    penalty = ewc.penalty(model)
                    loss = ce_loss + cfg["ewc"]["lambda_"] * penalty
                    ce_val, extra_val = ce_loss.detach().item(), penalty.detach().item()

                else:  # tmp
                    ce_loss = masked_cross_entropy(logits, y, task["classes"], device)
                    apply_every = cfg["tmp"].get("apply_every_n_steps", 1)
                    if global_step % apply_every == 0:
                        topo_loss, _ = tmp_ref.weighted_surrogate_loss(model)
                        loss = ce_loss + lambda_current * topo_loss
                        extra_val = topo_loss.detach().item()
                    else:
                        # Skip the surrogate term this step. This directly
                        # reduces CUMULATIVE protective dosage across the
                        # whole run -- important because task0 (present in
                        # the pool from the very first step) otherwise
                        # accumulates far more total reinforcement than
                        # later tasks regardless of the per-step weight cap,
                        # which was diagnosed (via analyze_output_layer_bias)
                        # as the cause of predictions systematically
                        # collapsing toward task0's class even for
                        # completely unrelated inputs.
                        loss = ce_loss
                        extra_val = 0.0
                    ce_val = ce_loss.detach().item()
                    global_step += 1

                loss.backward()

                # Diagnostic + safety net: log the pre-clip gradient norm
                # (catches silent gradient-explosion bugs like the one
                # diagnosed via the lambda sweep -- a small forward loss
                # value says nothing about backward gradient magnitude),
                # then clip so a single bad batch can't corrupt the whole
                # run even if something upstream misbehaves.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                grad_norm = float(grad_norm)
                running_grad_norm += grad_norm
                max_grad_norm_seen = max(max_grad_norm_seen, grad_norm)

                optimizer.step()
                running_ce += ce_val
                running_extra += extra_val
                n_batches += 1

            epoch_time = time.time() - epoch_start
            mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

            # ---- Retention across ALL previously-seen tasks ----
            per_task_retention = {}
            for prev_idx in range(task_idx):
                prev_test_loader = make_loader(tasks[prev_idx]["test"], cfg["eval_batch_size"], shuffle=False)
                per_task_retention[f"task{prev_idx}"] = evaluate_accuracy_unmasked(model, prev_test_loader, device)
            avg_retention = sum(per_task_retention.values()) / len(per_task_retention)

            # Feed this epoch's measurements back into the reference
            # pool's per-task weighting (Recommendation 3) for next
            # epoch's weighted surrogate loss.
            if method == "tmp":
                for prev_idx in range(task_idx):
                    tmp_ref.update_accuracy(prev_idx, per_task_retention[f"task{prev_idx}"])

            test_loader_current = make_loader(task["test"], cfg["eval_batch_size"], shuffle=False)
            learning_acc = evaluate_accuracy_unmasked(model, test_loader_current, device)

            drift_w_inf = None
            if method == "tmp":
                diagram_current = build_true_drift_diagram(
                    model, tmp_ref, device, cfg["tda"]["point_cloud_size"], cfg["tda"]["homology_dims"]
                )
                drift_w_inf = bottleneck_distance(
                    diagram_prev, diagram_current,
                    delta=cfg["tda"]["bottleneck_delta"],
                )
                # Recommendation 1: EMA-smoothed adaptive lambda, instead
                # of reacting to a single noisy epoch's raw drift value.
                lambda_current = lambda_scheduler.update_and_get_lambda(drift_w_inf)

            record = {
                "phase": f"task{task_idx}_{method}",
                "task_idx": task_idx,
                "epoch": epoch,
                "avg_retention_accuracy": avg_retention,
                "per_task_retention": per_task_retention,
                "learning_accuracy": learning_acc,
                "avg_ce_loss": running_ce / max(n_batches, 1),
                "avg_extra_term": running_extra / max(n_batches, 1),
                "epoch_time_sec": epoch_time,
                "memory_mb": mem_mb,
                "feature_space_drift_w_inf": drift_w_inf,
                "lambda_current": lambda_current if method == "tmp" else None,
                "avg_grad_norm": running_grad_norm / max(n_batches, 1),
                "max_grad_norm": max_grad_norm_seen,
            }
            history.append(record)
            logger.log(record)
            print(f"[Task {task_idx} {method}] epoch={epoch} avg_retention={avg_retention:.4f} "
                  f"learning_acc={learning_acc:.4f} drift={drift_w_inf} "
                  f"lambda={lambda_current if method == 'tmp' else '-'} "
                  f"grad_norm(avg/max)={running_grad_norm / max(n_batches, 1):.3f}/{max_grad_norm_seen:.3f} "
                  f"time={epoch_time:.2f}s")

        # ---- Task completed: give the regularizer a new snapshot ----
        if method == "ewc":
            ewc.add_snapshot(model, train_loader, device, task["classes"], cfg["ewc"]["fisher_sample_size"])
        if method == "tmp":
            tmp_ref.add_task(model, task["train"], device, seed=cfg["seed"] + task_idx, task_id=task_idx)
            diagram_prev = build_true_drift_diagram(
                model, tmp_ref, device, cfg["tda"]["point_cloud_size"], cfg["tda"]["homology_dims"]
            )

    return model, history


class JsonlLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        open(self.path, "w").close()

    def log(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
