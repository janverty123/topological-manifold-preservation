# Topological Manifold Preservation (TMP)

Implementation and experimental evidence for a proposed continual-learning
regularization technique: using Bottleneck Distance (persistent
homology) to mitigate catastrophic forgetting in deep neural networks.
Based on the approved research plan by Chua, Oasan, Ople, and Sales
(Adviser: Mr. Jose M. Manga Jr.).

## If you are Claude, start here

Read **`CLAUDE_PROJECT.md`** first. It gives the required reading
order and collaboration rules for working on this repository. Do not
skip it -- the short version is: read `CURRENT_CONTEXT.md`, this file,
`PROJECT_STATUS.md`, `DESIGN_PRINCIPLES.md`, `OPEN_QUESTIONS.md`, and
`EXPERIMENT_LOG.md`, in that order, before touching source code.

## If you are a human wanting to actually run this

Go straight to **`docs/SETUP_GUIDE.md`** -- full environment setup
(Python version, CUDA, giotto-tda), project structure, and a
step-by-step execution guide for the 2-task pipeline. For the 5-task
extension, see **`docs/MULTITASK.md`** after that.

## What this repository contains

Two experiments, both within Split-MNIST (per the approved research
plan's scope):

- **EXP-001** -- the original 2-task setup (digits 0-4, then 5-9).
  **Validated.** TMP significantly outperforms Finetune and EWC on
  retention with no significant learning-speed cost.
- **EXP-002** -- a harder 5-task stress test (`[0,1]->[2,3]->[4,5]->[6,7]->[8,9]`),
  testing whether protection holds up over a longer sequence.
  **Under active debugging** -- see `CURRENT_CONTEXT.md` for exactly
  where things stand.

## Documentation overview

| File | Purpose | Changes... |
|---|---|---|
| `CURRENT_CONTEXT.md` | What's happening right now, immediate next task | frequently -- read every session |
| `PROJECT_STATUS.md` | Long-term phase, milestones | slowly |
| `DESIGN_PRINCIPLES.md` | Stable rules for how this codebase is built | rarely |
| `OPEN_QUESTIONS.md` | Every investigated question, resolved or not | as questions resolve |
| `EXPERIMENT_LOG.md` | Chronological record of every experiment run | every experiment |
| `EXPERIMENT_WORKFLOW.md` | The process experiments follow | rarely |
| `CLAUDE_PROJECT.md` | Reading order + collaboration rules for Claude | rarely |
| `docs/CODEBASE_GUIDE.md` | Architecture, maps objectives to actual code locations | when code structure changes |
| `docs/SETUP_GUIDE.md` | Environment setup + 2-task execution guide | rarely |
| `docs/MULTITASK.md` | 5-task extension: setup, bugs found/fixed, usage | as EXP-002 progresses |

## Folder overview

```
configs/
  baseline/           # active configs (split_mnist_2task.yaml, split_mnist_5task.yaml)
  archive/             # superseded configs, kept for reference
docs/                 # see table above
src/                  # library code -- imported, never run directly
  train.py             # EXP-001 (2-task) trainer
  train_general.py      # EXP-002 (multi-task) trainer
  losses.py, tda_utils.py, models.py, ewc.py   # shared building blocks
  data.py, datasets_extended.py                 # dataset construction
  evaluate*.py, visualize*.py                   # stats + plots
scripts/              # EXP-001 entry points (run_finetune.py, run_ewc.py, run_tmp.py)
run_multitask.py       # EXP-002 entry point
compare_results.py / compare_multitask_results.py
sweep_tmp_lambda.py / sweep_tmp_lambda_multitask.py
diagnose_task_confusion.py / diagnose_gradient_conflict.py
outputs/
  experiments/EXP-001/, EXP-002/    # per-experiment logs, models, plots
  archive/
templates/
  experiment/          # copy into a new outputs/experiments/EXP-XXX/ for a new experiment
```

See `docs/CODEBASE_GUIDE.md` for why there are two separate trainers
instead of one shared one, and exactly where every piece of the TMP
algorithm is implemented.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# EXP-001 (2-task, validated)
python scripts/run_finetune.py --config configs/baseline/split_mnist_2task.yaml
python scripts/run_ewc.py --config configs/baseline/split_mnist_2task.yaml
python scripts/run_tmp.py --config configs/baseline/split_mnist_2task.yaml
python compare_results.py --config configs/baseline/split_mnist_2task.yaml

# EXP-002 (5-task, active debugging)
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method finetune
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method ewc
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method tmp
python compare_multitask_results.py --log-dir outputs/experiments/EXP-002/logs --plot-dir outputs/experiments/EXP-002/plots
```

Full details, troubleshooting, and CUDA setup: `docs/SETUP_GUIDE.md`.
