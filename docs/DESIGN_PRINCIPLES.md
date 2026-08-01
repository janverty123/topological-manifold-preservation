# Design Principles

Stable rules this project follows. These should rarely change; if you
find yourself wanting to violate one, that's a signal to update this
document deliberately, not silently work around it.

## 1. Two pipelines, not one shared abstraction

The 2-task (`src/train.py`) and multi-task (`src/train_general.py`)
pipelines are deliberately separate code, even though the multi-task
version is a strict generalization of the 2-task version (N=2 special
case). This is intentional: EXP-001 (2-task) is validated and stable;
EXP-002 (multi-task) is under active debugging. Merging them into one
shared abstraction would risk regressing EXP-001 every time EXP-002
needs a fix. Only unify them if EXP-002 reaches the same validated
status as EXP-001 AND a concrete maintenance cost justifies the merge.

## 2. Every regularization fix must be diagnosed before being applied

Do not tune a hyperparameter blind and hope. This project's actual
history: raising `ewc.lambda_` from 400 to 20000 was only done after
measuring `mean_fisher_value` and finding it 1000x smaller than
assumed; the TMP gradient-explosion bug was only fixed after directly
measuring gradient norms under a controlled near-collapse scenario,
not just observing "results look bad." Build a small, targeted
diagnostic script/test before changing a number. See
EXPERIMENT_WORKFLOW.md.

## 3. A small, well-behaved forward-pass loss value does NOT imply a
   well-behaved gradient

This bit the project once already (the self-normalizing surrogate
loss's denominator blew up on backward pass while looking completely
fine on forward pass). When debugging "training collapsed but the
loss value looks normal," check gradient magnitudes directly, not just
loss values.

## 4. Prefer a fixed, paired reference over resampled/random comparisons

Any drift-measuring or structure-preserving loss in this codebase
compares THE SAME underlying data at two points in time (e.g. "how did
the network represent these specific images before vs. now"), never
two independently-sampled batches. Comparing different random samples
introduces sampling noise that can dominate the actual signal -- this
was a real, diagnosed bug early in the project.

## 5. Class-incremental (unmasked) evaluation is the default "truth"

`retention_accuracy` always means the classifier had to choose from
ALL classes, not just the current task's classes. Task-incremental
(masked) evaluation is kept alongside for reference but is NOT the
headline metric -- it substantially understates catastrophic
forgetting and made an early result look artificially good before this
was caught.

## 6. Avoid unnecessary abstraction

Top-level utility/diagnostic scripts (`run_multitask.py`,
`sweep_tmp_lambda*.py`, `diagnose_*.py`, `compare*.py`) live at the
repo root and import `src.*` directly with no path hacks, because
that's simpler and already works. `scripts/` only holds the original
2-task pipeline's entry points, which need a `sys.path` hack for
historical reasons. Don't move things around just to make the
directory listing look more uniform if it adds no real benefit and
some risk.

## 7. Config files carry documentation metadata, but code never depends on it

Every config's `experiment:` and `output:` blocks (id, name, version,
status, description) are for humans/Claude reading the file, not
consumed by any script. The actual behavior-controlling keys
(`output_dir`, `tmp.lambda_`, etc.) are unchanged in meaning and
location. Never make code depend on the metadata block -- keep the
metadata purely additive so config files stay safe to reorganize
without touching training logic.

## 8. Research parameters are not touched during organizational/refactor work

Hyperparameters, dataset splits, model architecture, and training
logic are changed ONLY as part of an explicitly diagnosed experiment
(see EXPERIMENT_LOG.md), never as a side effect of reorganizing files,
renaming things, or improving documentation.
