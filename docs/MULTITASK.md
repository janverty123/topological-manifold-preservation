# 5-Task Split-MNIST: A Harder Stress Test, Same Dataset

This extends the original 2-task Split-MNIST implementation (see main
`README.md`) to a 5-task sequence, **without introducing any new
dataset** — still MNIST, still within your approved research plan's
scope, just a harder sequential stress test.

## What changed vs. the original 2-task setup

| | Original | This extension |
|---|---|---|
| Tasks | 2 | 5 |
| Split | `[0-4]` → `[5-9]` | `[0,1]` → `[2,3]` → `[4,5]` → `[6,7]` → `[8,9]` |
| Sequential drift events | 1 | 4 |
| Dataset | MNIST | MNIST (identical) |
| Architecture | `MLPClassifier` | `MLPClassifier` (identical) |

Same dataset, same model — just more, smaller sequential tasks. This
directly tests whether TMP's protection holds up over a longer
continual-learning sequence, or decays as more tasks accumulate, which
the original 2-task setup can't show you on its own.

## Files added (none of your original, validated 2-task code was touched)

- `src/datasets_extended.py` — `build_split_mnist_tasks()`, generalizing the class split to N tasks
- `src/train_general.py` — generalized N-task Finetune/EWC/TMP trainer (your validated 2-task version is the N=2 special case of this)
- `configs/baseline/split_mnist_5task.yaml` — same architecture and starting hyperparameters as your validated 2-task config
- `run_multitask.py` — top-level runner

## Improvements added after diagnosing weak Task 1 protection

After the initial 5-task results showed Task 1 retaining much worse
than Task 0 (see conversation history / earlier results), five
targeted improvements were implemented:

1. **EMA-smoothed adaptive lambda** (`AdaptiveLambdaScheduler` in
   `src/train_general.py`) — replaces the old raw, capped
   per-epoch rescale with an exponential moving average of drift, so
   a single noisy epoch's spike doesn't yank lambda around. Configurable
   via `tmp.ema_alpha` (default 0.2) and `tmp.lambda_max_multiplier`
   (default 3.0) in the config file. The current lambda is now logged
   every epoch as `lambda_current` in the JSONL output.

2. **Confusion-matrix diagnostic** (`diagnose_task_confusion.py`) —
   run after a `run_multitask.py` run to see exactly which classes a
   poorly-retained task's mistakes are being predicted as, plus
   cross-task activation similarity (checking for representational
   overlap as a source of extra interference):
   ```bash
   python diagnose_task_confusion.py --config configs/baseline/split_mnist_5task.yaml \
       --method tmp --victim-task 1
   ```

3. **Task-prioritized weighted reference pool**
   (`MultiTaskTMPReference.weighted_surrogate_loss` in
   `src/train_general.py`) — the surrogate loss is no longer one flat
   average over the whole reference pool. Each task-in-pool's
   contribution is weighted by `(1 - its_latest_retention_accuracy)`,
   normalized across all pooled tasks, so a task that's collapsing
   gets proportionally more protective gradient signal than a task
   that's already solid.

4. **Multi-task lambda sweep** (`sweep_tmp_lambda_multitask.py`) —
   adapts `sweep_tmp_lambda.py` (built for the 2-task pipeline) to
   the N-task pipeline, sharing the same cached Task-0 model across
   every lambda value tested:
   ```bash
   python sweep_tmp_lambda_multitask.py --config configs/baseline/split_mnist_5task.yaml
   python sweep_tmp_lambda_multitask.py --config configs/baseline/split_mnist_5task.yaml \
       --lambdas 0.0,2.5,5.0,7.5,10.0,15.0
   ```

5. **Per-task retention trajectory plots** — `compare_multitask_results.py`
   now automatically generates `mt_retention_trajectories_<method>.png`
   for every method present, tracing each individual task's retention
   accuracy across every subsequent task boundary (not just its final
   value) — shows WHEN a task's protection collapses, not just whether
   it did.

**Recommended order to actually run these**, if you're picking up from
here: (1) re-run all three methods with the updated TMP code, (2) look
at the new trajectory plots to see exactly when Task 1 (or whichever
task is weakest) starts declining, (3) if it's still weak, run the
lambda sweep, (4) if it's STILL weak after finding a better lambda,
use the confusion diagnostic to check whether it's genuine
representational interference from a specific other task, or something
else.

## Files added/changed in this round

- `src/train_general.py` — `AdaptiveLambdaScheduler` class added;
  `MultiTaskTMPReference` extended with per-task segment tracking and
  `weighted_surrogate_loss`; main loop updated to use both.
- `src/evaluate_multitask.py` — `build_retention_trajectory_matrix` added.
- `src/visualize_multitask.py` — `plot_task_retention_trajectories` added,
  wired into `generate_all_multitask_plots`.
- `diagnose_task_confusion.py` — new top-level script.
- `sweep_tmp_lambda_multitask.py` — new top-level script.
- `configs/baseline/split_mnist_5task.yaml` — added `tmp.ema_alpha` and
  `tmp.lambda_max_multiplier`.

None of the original 2-task pipeline (`train.py`, `run_finetune.py`,
`run_ewc.py`, `run_tmp.py`, `sweep_tmp_lambda.py`, `compare_results.py`)
was touched by any of this.

## How to run it (updated commands)

```bash
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method finetune
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method ewc
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method tmp

python compare_multitask_results.py --log-dir outputs/experiments/EXP-002/logs --plot-dir outputs/experiments/EXP-002/plots
```

Each `--method` run reuses the same cached Task-0 (`[0,1]`) baseline
model (`outputs/experiments/EXP-002/models/task0_base_model.pt`), so only
the first run pays the pretraining cost.

## Important — don't assume your 2-task hyperparameters transfer unchanged

`configs/baseline/split_mnist_5task.yaml` starts `tmp.lambda_` and
`ewc.lambda_` at the values you validated on the 2-task setup (5.0 and
20000.0), but a 5-task sequence accumulates drift differently — more
sequential adaptation steps, more total representational change by the
end. Re-run the same diagnostic sequence you already used successfully:

1. Check the printed `mean_fisher_value` for EWC — retune λ if it's
   far off from what worked at 2 tasks.
2. Run a `lambda_=0.0` TMP ablation and confirm it collapses toward
   Finetune-level retention — this is what proved the regularizer
   (not just the code structure) was responsible for your 2-task
   result, and it's the same check that matters here.
3. If needed, sweep a few λ values and check the trend is sensible.

## What to look at in the results

`outputs/experiments/EXP-002/logs/<method>.jsonl` gives you, per epoch:
- `avg_retention_accuracy` — averaged across all previously-seen tasks
- `per_task_retention` — a breakdown, e.g.
  `{"task0": 0.91, "task1": 0.87, "task2": 0.60, "task3": 0.55}`

**The `per_task_retention` breakdown is the most valuable new thing
this setup gives you.** Does TMP protect *task0* (`[0,1]`, learned
longest ago) as well as *task3* (`[6,7]`, learned most recently)? A
known failure mode in continual learning is retention decaying with
distance — the oldest tasks get forgotten fastest even under a
"protective" method. If that pattern shows up (or doesn't), it's worth
a dedicated plot and a paragraph in your Discussion section — it's a
more nuanced, more defensible claim than a single before/after number
from the 2-task setup.

## Bug found via the lambda sweep: gradient explosion in the surrogate loss

Running `sweep_tmp_lambda_multitask.py` surfaced a real bug: **every**
nonzero lambda (2.5 through 15.0) collapsed learning accuracy to
near-zero, while retention didn't improve either -- not the gradual
tradeoff curve you'd expect from real regularization. Diagnostic
logging (`lambda_current`, `avg_extra_term`) ruled out both "lambda
pinned at its cap" and "the loss value itself is exploding" -- the
forward-pass loss value stayed small and well-behaved throughout.

**Root cause:** `topological_surrogate_loss` (in `src/losses.py`)
normalized pairwise-distance matrices by their own LIVE, non-detached
mean. A small, well-behaved forward loss value says nothing about the
BACKWARD gradient magnitude -- the gradient of `x / mean(x)` contains a
`1 / mean(x)` term that blows up as the mean shrinks toward zero (which
can happen if activations partially collapse under training pressure).
Verified directly: as mean pairwise distance dropped from ~1e-2 to
~1e-9, the old formula's gradient norm exploded from ~11 to ~5.8
million, while a detached-and-clamped denominator stayed bounded around
~2,500 across the same range.

**Fix applied** (both now in place):
1. `src/losses.py` -- the normalization denominator is now detached
   and clamped to a minimum floor (the "stop-gradient on statistics"
   technique used in BYOL/SimSiam), keeping gradients numerically
   bounded regardless of how much the activations shrink.
2. `src/train_general.py` -- gradient norm is now clipped
   (`max_norm=10.0`) every step as a safety net, and both the pre-clip
   average and max gradient norm per epoch are logged
   (`avg_grad_norm`, `max_grad_norm` in the JSONL output) so a future
   silent explosion like this one would be immediately visible instead
   of only showing up as mysteriously collapsed accuracy.

**If you already ran the lambda sweep with the old code, re-run it** --
those results (the ones showing a cliff at any nonzero lambda) were
produced under the bug and should not be used or reported.

## Second bug found: weight concentration in the task-prioritized pool

Even after the gradient-explosion fix above, re-running the sweep
showed a SECOND, distinct problem: new-task learning accuracy got
stuck near exactly 0.0000 for entire tasks, at every nonzero lambda
tested (2.5 through 15), non-monotonically -- not the smooth
strength-based tradeoff real regularization should show.

**Root cause:** the weighting formula from Recommendation 3
(`w_k = (1 - acc_k) / sum_j(1 - acc_j)`) has no limit on how
concentrated it can get. Verified directly with realistic accuracy
values matching the actual runs (task1 struggling at 40% while other
tasks sit at 95%+): **task1 alone consumed 90.9% of the entire
protective gradient budget**, every single step. That's a persistent,
heavily concentrated pull on the shared hidden layers, consistent with
new-task learning getting completely blocked rather than merely slowed.

**Fix applied:** `MultiTaskTMPReference.weighted_surrogate_loss` now
caps any single task's weight at `max_weight_cap` (default 0.5) and
redistributes the excess proportionally among the remaining tasks
(`_cap_and_redistribute`). A simple additive floor was tried first and
found too weak to matter when the accuracy gap is large (barely moved
90.9% -> 85.7%); the hard cap actually bounds it (90.9% -> 50.0% for
the same example, verified directly).

## New diagnostic: gradient conflict test

`diagnose_gradient_conflict.py` directly measures whether the CE
gradient and the topo surrogate gradient are pointing in OPPOSING
directions (a "tug of war") on the shared parameters, or merely
differ in magnitude -- two different problems needing different fixes.

```bash
python diagnose_gradient_conflict.py --config configs/baseline/split_mnist_5task.yaml --task-idx 2
```

Prints per-batch cosine similarity between the two gradients, plus
their raw magnitudes -- a strongly negative cosine similarity confirms
direct conflict; a similarity near 0 with a much LARGER topo gradient
magnitude points to magnitude dominance instead (drowning out the CE
signal without directly opposing it). Both are real, distinct failure
modes worth telling apart before deciding on a fix.

**Recommended next step:** re-run `sweep_tmp_lambda_multitask.py` with
both fixes now in place, and if learning is STILL stuck at any nonzero
lambda, run the gradient conflict diagnostic to see which of the two
remaining mechanisms (direct opposition vs. magnitude dominance) is
responsible, since that determines whether the fix is capping
`max_weight_cap` further, scaling down `lambda_`, or something more
structural like reducing how often the surrogate loss is applied.

## Third bug found: cumulative protective dosage asymmetry (class-0 collapse)

Re-running the sweep with the weight-cap fix showed a MIXED result:
task3 learned properly (smooth growth to ~26%), but task2 and task4
stayed completely stuck at 0.0000 for their entire training. Running
`diagnose_task_confusion.py` on both victim tasks revealed the real
mechanism: **~75-80% of ALL misclassifications collapsed to class 0
specifically**, not spread across "plausible" confusions. Cross-task
activation cosine similarity was also uniformly high (0.96-0.99)
across every task pair for both victims -- ruling out "task2/task3
specifically resemble some other task" and pointing instead to a
general representational collapse.

**Root cause:** task0 has been in the protected reference pool since
the very first step and gets reinforced on EVERY training step across
the ENTIRE 40-epoch run. Task1 only gets protected starting from
task2's training; task3 only from task4's. Even with the per-step
weight cap (fix #2, still correct and still needed), **task0
accumulates vastly more TOTAL protective pressure over the full run
than any later task** -- the cap limits concentration per step, but
does nothing about cumulative exposure over time. This was CONFIRMED
directly with `analyze_output_layer_bias` (new function in
`diagnose_task_confusion.py`): on an artificially-biased test model,
it correctly showed the dominant class's output weight norm and mean
logit standing out sharply from every other class -- exactly the kind
of signature to check for on your own runs.

**Fix applied:** `src/train_general.py` now applies the surrogate loss
only every `tmp.apply_every_n_steps` steps (default 2, config in
`configs/baseline/split_mnist_5task.yaml`) instead of every single step.
This directly reduces cumulative dosage across the board, proportionally
affecting task0 (which accumulates the most exposure) the most in
absolute terms, while still applying real protection regularly.

**To check whether your run shows this specific failure mode**, run:
```bash
python diagnose_task_confusion.py --config configs/baseline/split_mnist_5task.yaml --method tmp --victim-task <N>
```
and look at the new "Output layer bias analysis" section in the
output -- if one class's weight norm / mean logit is far above the
rest, you're looking at this same collapse mechanism.

**This is a genuinely deep design tension worth being upfront about in
your paper**, not just a bug to quietly patch: any method that keeps a
GROWING, unequally-aged reference pool (as this TMP implementation
does) will structurally tend to over-protect early tasks unless
something explicitly corrects for unequal cumulative exposure.
`apply_every_n_steps` is a blunt, uniform mitigation; a more
principled fix (e.g. explicitly tracking and equalizing each task's
TOTAL cumulative protection dosage, not just its per-step share) is a
reasonable "Future Work" item if you want to extend this further.



