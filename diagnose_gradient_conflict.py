"""
diagnose_gradient_conflict.py
--------------------------------
Directly tests whether the cross-entropy gradient and the weighted
topological surrogate gradient are CONFLICTING (pointing in opposing
directions) on the shared hidden-layer parameters, as opposed to
merely differing in magnitude.

This is a standalone diagnostic (not run every training step, since
computing two separate backward passes per step would meaningfully
slow down real training) -- run it after a `run_multitask.py --method
tmp` run to inspect what's actually happening at a specific point in
training.

Interpretation:
    cosine similarity near -1  -> the two gradients actively oppose
                                   each other (a "tug of war"); more
                                   protective pressure directly fights
                                   new-task learning
    cosine similarity near  0  -> the two objectives are largely
                                   orthogonal / independent
    cosine similarity near +1  -> the two objectives are well-aligned
                                   (rare, but would mean the surrogate
                                   isn't actually the bottleneck)

Usage:
    python diagnose_gradient_conflict.py --config configs/baseline/split_mnist_5task.yaml \
        --task-idx 2
"""

import argparse

import torch
import torch.nn.functional as F
import yaml

from src.datasets_extended import build_tasks
from src.models import MLPClassifier
from src.losses import masked_cross_entropy
from src.train_general import (
    pretrain_first_task,
    make_loader,
    MultiTaskEWC,
    MultiTaskTMPReference,
    JsonlLogger,
    evaluate_accuracy_unmasked,
)


def resolve_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def flatten_grad(model):
    """
    Flattens all parameter gradients into one vector, using a ZERO
    vector for any parameter with no gradient (e.g. fc3/the output
    layer never receives a gradient from the topo loss alone, since it
    only depends on the hidden2 activation, which is upstream of fc3 --
    not skipping these keeps both gradient vectors the same length and
    correctly aligned for cosine similarity.
    """
    parts = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-idx", type=int, required=True,
                         help="Which task_idx to probe (e.g. 2 for the 3rd sequential task)")
    parser.add_argument("--n-batches", type=int, default=5,
                         help="Number of batches to average the cosine similarity over")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg)
    tasks = build_tasks(cfg["dataset"], cfg["data_dir"], cfg["num_tasks"], seed=cfg["seed"])

    model = MLPClassifier(
        input_dim=cfg["input_dim"], hidden1_dim=cfg["hidden1_dim"],
        hidden2_dim=cfg["hidden2_dim"], num_classes=cfg["num_classes"],
    ).to(device)

    logger0 = JsonlLogger("/tmp/diag_task0.jsonl")
    model = pretrain_first_task(
        cfg, model, tasks, device, logger0,
        target_acc=cfg["task1_target_accuracy"], max_epochs=cfg["task1_max_epochs"], lr=cfg["task1_lr"],
    )

    # Build the TMP reference pool up through the task BEFORE the probed one,
    # matching what it would look like at the start of that task's training.
    tmp_ref = MultiTaskTMPReference(cfg["tmp"]["surrogate_subsample"])
    tmp_ref.add_task(model, tasks[0]["train"], device, seed=cfg["seed"], task_id=0)
    for t in range(1, args.task_idx):
        # Fine-tune briefly on each intermediate task so the pool's
        # baseline activations and accuracy estimates are realistic,
        # not just the pristine task-0 state.
        loader = make_loader(tasks[t]["train"], cfg["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["task2_lr"])
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = masked_cross_entropy(model(x), y, tasks[t]["classes"], device)
            loss.backward()
            optimizer.step()
        tmp_ref.add_task(model, tasks[t]["train"], device, seed=cfg["seed"] + t, task_id=t)
        for prev in range(t + 1):
            test_loader = make_loader(tasks[prev]["test"], cfg["eval_batch_size"], shuffle=False)
            tmp_ref.update_accuracy(prev, evaluate_accuracy_unmasked(model, test_loader, device))

    print(f"\nReference pool built through task{args.task_idx - 1}. "
          f"Current per-task accuracy estimates: {tmp_ref.latest_accuracy}")

    unstable = {tid: max(1.0 - acc, 0.0) for tid, acc in tmp_ref.latest_accuracy.items()}
    total = sum(unstable.values()) + 1e-8
    raw_weights = {tid: w / total for tid, w in unstable.items()}
    capped_weights = tmp_ref._cap_and_redistribute(raw_weights, 0.5)
    print(f"Raw (uncapped) weights:    {{{', '.join(f'{k}: {v:.3f}' for k, v in raw_weights.items())}}}")
    print(f"Capped weights (in use):   {{{', '.join(f'{k}: {v:.3f}' for k, v in capped_weights.items())}}}")

    # Now probe gradient conflict on the actual task being trained.
    train_loader = make_loader(tasks[args.task_idx]["train"], cfg["batch_size"], shuffle=True)
    cos_sims = []

    for i, (x, y) in enumerate(train_loader):
        if i >= args.n_batches:
            break
        x, y = x.to(device), y.to(device)

        # CE gradient alone
        model.zero_grad()
        logits = model(x)
        ce_loss = masked_cross_entropy(logits, y, tasks[args.task_idx]["classes"], device)
        ce_loss.backward()
        ce_grad = flatten_grad(model).clone()

        # Topo gradient alone
        model.zero_grad()
        topo_loss, _ = tmp_ref.weighted_surrogate_loss(model)
        topo_loss.backward()
        topo_grad = flatten_grad(model).clone()

        cos_sim = F.cosine_similarity(ce_grad.unsqueeze(0), topo_grad.unsqueeze(0)).item()
        cos_sims.append(cos_sim)
        print(f"Batch {i}: CE_grad_norm={ce_grad.norm().item():.3f}  "
              f"topo_grad_norm={topo_grad.norm().item():.3f}  cosine_similarity={cos_sim:.4f}")

    avg_cos = sum(cos_sims) / len(cos_sims)
    print(f"\nMean cosine similarity over {len(cos_sims)} batches: {avg_cos:.4f}")
    if avg_cos < -0.3:
        print("-> STRONG CONFLICT: the two gradients are substantially opposing each other.")
    elif avg_cos < 0.0:
        print("-> Mild conflict: gradients are somewhat opposing.")
    else:
        print("-> No significant conflict detected -- the bottleneck is likely elsewhere "
              "(e.g. raw magnitude dominance rather than directional opposition).")


if __name__ == "__main__":
    main()
