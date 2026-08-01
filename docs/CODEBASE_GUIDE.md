# Codebase Guide

Derived directly from the current source files (verified, not guessed).
If this ever looks stale, trust the code over this document and update
this document to match.

## Two parallel pipelines, by design

This repo has **two separate, independently-working training
pipelines** that intentionally do NOT share code, so that fixing/
experimenting with one can never silently break the other:

| | 2-task pipeline (EXP-001) | Multi-task pipeline (EXP-002) |
|---|---|---|
| Config | `configs/baseline/split_mnist_2task.yaml` | `configs/baseline/split_mnist_5task.yaml` |
| Core trainer | `src/train.py` (342 lines) | `src/train_general.py` (496 lines) |
| Dataset builder | `src/data.py` | `src/datasets_extended.py` |
| Entry points | `scripts/run_finetune.py`, `scripts/run_ewc.py`, `scripts/run_tmp.py` | `run_multitask.py --method <finetune\|ewc\|tmp>` |
| Comparison | `compare_results.py` | `compare_multitask_results.py` |
| Lambda sweep | `sweep_tmp_lambda.py` | `sweep_tmp_lambda_multitask.py` |
| Evaluation stats | `src/evaluate.py` | `src/evaluate_multitask.py` |
| Plotting | `src/visualize.py` | `src/visualize_multitask.py` |
| Tasks | exactly 2 (hardcoded) | N sequential tasks (generalized) |
| EWC | `src/ewc.py` (`EWC` class, single Fisher snapshot) | `MultiTaskEWC` class inside `train_general.py` (one Fisher snapshot per completed task, summed) |
| TMP reference | single fixed Task-1 image set | `MultiTaskTMPReference` class inside `train_general.py`: GROWING pool, one segment added per completed task |

**The 2-task pipeline is the N=2 special case of the multi-task
pipeline's design**, but they are literally different code, not a
shared abstraction -- this was a deliberate choice (see
DESIGN_PRINCIPLES.md) to keep the validated EXP-001 pipeline stable
while EXP-002 was actively being debugged.

## Shared, dataset/pipeline-agnostic modules

- **`src/models.py`** -- `MLPClassifier` (784->256->128->10, used by
  BOTH pipelines) and `apply_class_mask` (masks logits to a task's
  allowed classes for class-incremental training/eval). A `CNNClassifier`
  for Split-CIFAR-100 existed in an earlier version of this repo but was
  REMOVED when the project scope was narrowed to Split-MNIST-only (see
  OPEN_QUESTIONS.md and PROJECT_STATUS.md) -- do not re-add it without
  checking current scope first.
- **`src/losses.py`** -- `masked_cross_entropy` (task-incremental CE)
  and `topological_surrogate_loss` (the differentiable TMP surrogate,
  used by BOTH pipelines). See "Where TMP is implemented" below for
  the important numerical-stability details in this file.
- **`src/tda_utils.py`** -- Maxmin point-cloud sampling, persistence
  diagram construction (giotto-tda `VietorisRipsPersistence`), true
  Bottleneck Distance computation (`W_inf`), and
  `normalize_point_cloud` (removes raw-activation-scale confound from
  the true drift metric).

## Where things actually happen

### Dataset loading
- 2-task: `src/data.py` -- downloads MNIST via torchvision, splits into
  digits `[0-4]` / `[5-9]` using a picklable `_FlattenTransform` class
  (NOT a lambda -- lambdas broke `DataLoader(num_workers>0)` on
  Windows; this was a real bug found and fixed).
- Multi-task: `src/datasets_extended.py` -- `build_split_mnist_tasks`
  generalizes the same MNIST split to N sequential tasks (default 5:
  `[0,1]->[2,3]->[4,5]->[6,7]->[8,9]`). Reuses `_FlattenTransform`
  from `src/data.py` rather than duplicating it.

### Where the TMP loss is implemented
- **Differentiable surrogate**: `src/losses.py::topological_surrogate_loss`.
  Compares pairwise-distance matrices of a FIXED reference image set's
  activations (current model state) against that same set's baseline
  activations (captured once, frozen). Both matrices are normalized by
  their own mean before comparing -- **the denominator is DETACHED and
  clamped to a floor** (not a live, non-detached mean) specifically to
  prevent a gradient-explosion bug found via `sweep_tmp_lambda_multitask.py`
  (see EXPERIMENT_LOG.md EXP-002-05 for the full diagnosis).
- **True (non-differentiable) drift metric**: `src/tda_utils.py::bottleneck_distance`,
  computed via giotto-tda on a Maxmin-sampled, `normalize_point_cloud`'d
  activation cloud -- used for logging/monitoring `feature_space_drift_w_inf`
  and for the adaptive lambda schedule, NOT backpropagated through directly.
- **2-task TMP training loop**: `src/train.py::train_task2` (method="tmp"
  branch). Uses a single fixed Task-1 reference set built once via
  `build_fixed_task1_reference`.
- **Multi-task TMP training loop**: `src/train_general.py::train_continual_multitask`
  (method="tmp" branch) + the `MultiTaskTMPReference` class (same file).
  The reference pool GROWS by `tmp.surrogate_subsample` images every
  time a task completes. `weighted_surrogate_loss` weights each
  pooled task's contribution by `(1 - its_latest_measured_accuracy)`,
  capped at `max_weight_cap` (default 0.5) with proportional
  redistribution -- see EXPERIMENT_LOG.md EXP-002-06 for why the cap
  exists (an earlier uncapped version let one struggling task consume
  >90% of all protective gradient). `AdaptiveLambdaScheduler` (same
  file) EMA-smooths lambda based on measured drift instead of reacting
  to a single noisy epoch.
- **`tmp.apply_every_n_steps`** (multi-task config only, default 2):
  skips the surrogate loss on alternating steps to reduce CUMULATIVE
  protective dosage -- added after diagnosing that task0 (protected
  since step 1) structurally accumulates far more total reinforcement
  than later tasks regardless of per-step weight caps (see
  EXPERIMENT_LOG.md EXP-002-07).

### Where EWC is implemented
- 2-task: `src/ewc.py::EWC` -- single Fisher/anchor-parameter snapshot
  computed once from Task-1 data before Task-2 training starts.
- Multi-task: `MultiTaskEWC` class inside `src/train_general.py` --
  accumulates a NEW Fisher/anchor snapshot after every completed task,
  penalty is the sum across all snapshots.

### Where evaluation/retention is computed
- `evaluate_accuracy` (task-incremental, masked -- classifier only
  chooses among the current task's classes) vs.
  `evaluate_accuracy_unmasked` (class-incremental, full head -- the
  classifier must pick the right class from ALL classes). **The
  unmasked version is the primary reported "retention accuracy"**
  metric in both pipelines -- it's the one that actually demonstrates
  catastrophic forgetting; the masked version substantially understates
  it (an earlier bug: results looked artificially good because only
  the masked metric was being reported). Both live in `src/train.py`
  (2-task) and `src/train_general.py` (multi-task).

### Diagnostic tools (built during EXP-002 debugging, standalone scripts)
- **`diagnose_task_confusion.py`** -- confusion matrix, misclassification
  breakdown, cross-task activation cosine similarity, AND
  `analyze_output_layer_bias` (per-class output weight norm / mean
  logit -- catches systematic prediction collapse toward one class).
- **`diagnose_gradient_conflict.py`** -- directly measures cosine
  similarity between the CE gradient and the TMP surrogate gradient on
  shared parameters, to distinguish "the two objectives are actively
  opposing each other" from "one just has larger raw magnitude."

## Execution flow (multi-task pipeline, the actively-developed one)

```
run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method tmp
  |
  v
src/datasets_extended.py::build_tasks()          # builds list of 5 task dicts
  |
  v
src/train_general.py::pretrain_first_task()      # trains + caches task0 model
  |
  v
src/train_general.py::train_continual_multitask()  # sequential loop over tasks 1..4
  |     for method="tmp": MultiTaskTMPReference grows each task boundary,
  |     AdaptiveLambdaScheduler updates lambda each epoch,
  |     weighted_surrogate_loss computes the capped, weighted TMP loss
  v
outputs/experiments/EXP-002/logs/tmp.jsonl        # one JSON record per epoch
outputs/experiments/EXP-002/models/tmp_final_model.pt
  |
  v
compare_multitask_results.py                      # reads all method .jsonl logs
  |
  v
src/evaluate_multitask.py + src/visualize_multitask.py
  |
  v
outputs/experiments/EXP-002/plots/*.png + mt_comparison_results.json
```

## A note on top-level scripts vs. `scripts/`

`scripts/` only contains the ORIGINAL 2-task pipeline's entry points
(`run_finetune.py`, `run_ewc.py`, `run_tmp.py`, `_common.py`), which
need a `sys.path` hack (`scripts/_common.py`) to import `src.*` since
they don't live at the repo root. Every script added AFTER that
(`run_multitask.py`, `sweep_tmp_lambda*.py`, `compare*.py`,
`diagnose_*.py`) lives at the repo root instead and imports `src.*`
directly with no path hack needed, since Python auto-adds a
directly-invoked script's own directory to `sys.path`, and that
directory IS the repo root for these. This is intentional, not an
inconsistency to "fix" -- moving them into `scripts/` would require
adding the same path hack to every one of them for no behavioral
benefit (see DESIGN_PRINCIPLES.md: avoid unnecessary abstraction).
