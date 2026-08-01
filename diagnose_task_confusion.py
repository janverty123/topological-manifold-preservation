"""
diagnose_task_confusion.py
----------------------------
Recommendation 2: diagnosing why a specific task's retention collapses
more than others (e.g. Task 1 retaining much worse than Task 0 despite
similar protection).

Loads a saved final model checkpoint from a completed run_multitask.py
run, and for a chosen "victim" task:
  1. Prints a confusion matrix + classification report on that task's
     test set, so you can see exactly which classes it's being
     misclassified AS (is it consistently confused with one particular
     later task's classes, or just generally degraded?).
  2. Computes the mean cosine similarity between the victim task's
     hidden-layer activations and every other task's activations, to
     check whether representational/domain overlap explains the extra
     interference.

Usage:
    python diagnose_task_confusion.py --config configs/baseline/split_mnist_5task.yaml \
        --method tmp --victim-task 1
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.datasets_extended import build_tasks
from src.models import MLPClassifier
from src.train_general import make_loader


def resolve_device(cfg):
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_final_model(cfg, method, device):
    import os
    ckpt_path = os.path.join(cfg["output_dir"], "models", f"{method}_final_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No final model found at {ckpt_path}. Run "
            f"`python run_multitask.py --config <config> --method {method}` first."
        )
    model = MLPClassifier(
        input_dim=cfg["input_dim"], hidden1_dim=cfg["hidden1_dim"],
        hidden2_dim=cfg["hidden2_dim"], num_classes=cfg["num_classes"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def confusion_report(model, task, device, class_names=None):
    loader = make_loader(task["test"], batch_size=256, shuffle=False)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().tolist())

    all_labels = sorted(set(all_targets) | set(all_preds))
    print("\nConfusion matrix (rows=true, cols=predicted), labels =", all_labels)
    print(confusion_matrix(all_targets, all_preds, labels=all_labels))
    print("\nClassification report:")
    print(classification_report(all_targets, all_preds, labels=all_labels, zero_division=0))

    # What are the misclassifications actually predicted AS?
    wrong_preds = [p for p, t in zip(all_preds, all_targets) if p != t]
    if wrong_preds:
        vals, counts = np.unique(wrong_preds, return_counts=True)
        print("Misclassified samples were predicted as (class: count):")
        for v, c in sorted(zip(vals, counts), key=lambda x: -x[1]):
            in_task_classes = "own task class" if v in task["classes"] else "OTHER task class"
            print(f"  class {v}: {c} times  <- {in_task_classes}")
    else:
        print("No misclassifications -- perfect retention on this task's test set.")


def cross_task_activation_similarity(model, tasks, victim_idx, device, n_samples=200):
    """
    Mean cosine similarity between the victim task's hidden2 activations
    and every other task's, as a proxy for representational/domain
    overlap that could explain extra interference.
    """
    def get_activations(task, n):
        loader = make_loader(task["test"], batch_size=min(n, 256), shuffle=True)
        return model.extract_activations(loader, device, max_samples=n)

    victim_acts = get_activations(tasks[victim_idx], n_samples)
    victim_mean = F.normalize(victim_acts.mean(dim=0, keepdim=True), dim=1)

    print(f"\nCross-task activation similarity (victim = task{victim_idx}):")
    for i, task in enumerate(tasks):
        if i == victim_idx:
            continue
        other_acts = get_activations(task, n_samples)
        other_mean = F.normalize(other_acts.mean(dim=0, keepdim=True), dim=1)
        sim = F.cosine_similarity(victim_mean, other_mean).item()
        print(f"  task{victim_idx} vs task{i}: mean cosine similarity = {sim:.4f}")


def analyze_output_layer_bias(model, tasks, device, n_samples_per_task=200):
    """
    Directly checks whether one class's output weights/logits are
    disproportionately dominant -- the mechanism that would explain a
    confusion matrix where errors overwhelmingly collapse to ONE
    specific class (e.g. class 0) rather than spreading across
    "plausible" confusions. Two independent checks:

      1. ||fc3.weight[c]|| and fc3.bias[c] per class -- a class whose
         output weight vector has grown much larger than others will
         tend to win argmax more often almost regardless of the input,
         since its raw logit contribution dominates.
      2. Mean logit value per class over a MIXED sample of images from
         ALL tasks -- if one class's average logit is far higher than
         the rest even across totally unrelated inputs, that's direct
         evidence of a systematic output bias, not input-dependent
         confusion.
    """
    print("\n--- Output layer bias analysis ---")
    weight_norms = model.fc3.weight.detach().norm(dim=1).cpu().numpy()
    biases = model.fc3.bias.detach().cpu().numpy()
    print("Per-class output weight norm and bias:")
    for c in range(len(weight_norms)):
        print(f"  class {c}: ||weight||={weight_norms[c]:.4f}  bias={biases[c]:.4f}")

    all_images = []
    for task in tasks:
        loader = make_loader(task["test"], batch_size=min(n_samples_per_task, 256), shuffle=True)
        x, _ = next(iter(loader))
        all_images.append(x[:n_samples_per_task])
    mixed_batch = torch.cat(all_images, dim=0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(mixed_batch)
    mean_logits = logits.mean(dim=0).cpu().numpy()

    print(f"\nMean logit per class over a MIXED batch of {mixed_batch.shape[0]} images "
          f"from ALL {len(tasks)} tasks (should be roughly balanced if unbiased):")
    for c in range(len(mean_logits)):
        print(f"  class {c}: mean_logit={mean_logits[c]:.4f}")

    top_class = int(mean_logits.argmax())
    print(f"\nHighest mean logit: class {top_class} ({mean_logits[top_class]:.4f}) -- "
          f"if this is far above the rest, the network is systematically biased toward "
          f"predicting this class regardless of input.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", required=True, choices=["finetune", "ewc", "tmp"])
    parser.add_argument("--victim-task", type=int, required=True,
                         help="task_idx (0-based) to diagnose, e.g. 1 for the task that retains poorly")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = resolve_device(cfg)
    tasks = build_tasks(cfg["dataset"], cfg["data_dir"], cfg["num_tasks"], seed=cfg["seed"])
    model = load_final_model(cfg, args.method, device)

    print(f"Diagnosing task{args.victim_task} (classes={tasks[args.victim_task]['classes']}) "
          f"under the FINAL {args.method} model (after all {len(tasks)} tasks).")

    confusion_report(model, tasks[args.victim_task], device)
    cross_task_activation_similarity(model, tasks, args.victim_task, device)
    analyze_output_layer_bias(model, tasks, device)


if __name__ == "__main__":
    main()
