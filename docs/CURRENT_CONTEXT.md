# Current Context

**Read this file first, every time.** It is the single source of truth
for "where things stand right now." Update it whenever the current
objective, priority, or known-issue list changes -- it should never go
stale for more than one work session.

Last updated: 2026-08-02 (documentation/repository synchronization pass)

---

## Current objective

Get EXP-002 (5-task Split-MNIST) to a genuinely validated state --
same bar EXP-001 already cleared -- by confirming TMP's improvement
over Finetune/EWC holds up AFTER the three bugs found and fixed during
lambda-sweep testing (see "Fixed project decisions" below).

## Current implementation goal

Re-run the full 3-method comparison
(`run_multitask.py --method finetune/ewc/tmp`, then
`compare_multitask_results.py`) with all three EXP-002 fixes active,
and determine whether:
1. Learning accuracy is no longer stuck at 0.0000 for any task.
2. The class-0 prediction collapse (diagnosed via
   `analyze_output_layer_bias`) is gone or substantially reduced.
3. The previously-validated 75.3% / 64.4% / 60.2% retention numbers
   (TMP / EWC / Finetune) still hold, improve, or need lambda re-tuning.

## Current priorities (in order)

1. Re-run EXP-002's full comparison with the current code (all 3 fixes
   active) -- this is the single next action, see "Immediate next
   task" below.
2. If still broken: use `diagnose_gradient_conflict.py` on whichever
   task is still stuck, to distinguish direct gradient opposition from
   magnitude dominance (two different fixes).
3. If working: re-validate `tmp.lambda_` / `ewc.lambda_` specifically
   for 5-task via the sweep scripts (currently inherited, unconfirmed
   starting points from EXP-001).
4. Only after EXP-002 is validated: consider whether Permuted MNIST /
   Split-CIFAR-100 are worth reviving (currently explicitly
   out-of-scope -- see "Fixed project decisions").
5. **(Documentation, non-blocking)** Resolve the EXP-001 config
   discrepancy noted below -- requires a decision from the project
   owner, not further investigation. See "Current known issues."

## Fixed project decisions (do not re-litigate without new evidence)

- **Scope is Split-MNIST only** (2-task EXP-001 + 5-task EXP-002).
  Permuted MNIST and Split-CIFAR-100 were explicitly descoped by the
  research team to stay within the approved research plan and to
  finish debugging one dataset properly before adding more. Code for
  CIFAR-100 (`CNNClassifier`, `build_split_cifar100_tasks`) was
  REMOVED from this repo copy, not just unused -- don't assume it
  exists.
- **`retention_accuracy` means class-incremental (unmasked, full
  10-way head)**, not task-incremental (masked). This was a real bug
  fixed early in EXP-001 -- the masked metric substantially understates
  forgetting and made all three methods look artificially similar.
- **The 2-task and multi-task pipelines are separate code**, not a
  shared abstraction (see DESIGN_PRINCIPLES.md). Do not refactor them
  together without strong justification.
- **TMP's differentiable surrogate compares a FIXED reference image
  set against itself over time**, never two different random samples.
  An earlier version compared different images and produced a
  meaningless, noisy training signal -- this was a real, diagnosed bug.

## Current known issues

See OPEN_QUESTIONS.md for the full, actively-tracked list. Headline
items right now:
- EXP-002's validated-looking 75.3/64.4/60.2 retention numbers predate
  three bugfixes and have NOT been re-confirmed.
- `tmp.lambda_=5.0` and `ewc.lambda_=20000.0` in the 5-task config are
  inherited from EXP-001, not independently re-validated for 5 tasks.
- The `apply_every_n_steps` fix for cumulative-dosage asymmetry is a
  blunt, uniform mitigation, not a structural fix -- it may not fully
  resolve task-specific collapse (see EXPERIMENT_LOG.md EXP-002-07).
- **NEW (found during 2026-08-02 doc sync, unresolved, needs project-owner decision):**
  `configs/baseline/split_mnist_2task.yaml` (the EXP-001 config) still
  holds the PRE-VALIDATION starting values `tmp.lambda_=0.5` /
  `ewc.lambda_=400.0`, not the validated `5.0` / `20000.0` reported in
  `outputs/experiments/EXP-001/metadata.yaml` and
  `outputs/experiments/EXP-001/README.md`. `EXP-001/README.md`'s "How
  to reproduce" section currently does not actually reproduce the
  validated numbers if followed literally against the checked-in
  config. This is a documentation/config inconsistency, not a code
  bug -- see the `SYNC-FLAG` comment block directly above `tmp:` and
  `ewc:` in that config file for the two possible resolutions. Not
  resolved automatically; treat as a research-parameter decision
  requiring explicit sign-off, per DESIGN_PRINCIPLES.md #8.

## Immediate next task

```bash
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method finetune
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method ewc
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method tmp
python compare_multitask_results.py --log-dir outputs/experiments/EXP-002/logs --plot-dir outputs/experiments/EXP-002/plots
```
Then compare the new `mt_comparison_results.json` against the
provisional numbers in `outputs/experiments/EXP-002/metadata.yaml`,
and log the outcome as a new entry in `EXPERIMENT_LOG.md` (EXP-002-08).

(Separately, and non-blocking: get a decision on the EXP-001 config
discrepancy above and resolve it in its own small documentation/config
commit -- it does not affect EXP-002 work.)
