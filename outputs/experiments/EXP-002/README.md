# EXP-002: Split-MNIST 5-Task Stress Test

**Status:** active_debugging (NOT yet validated -- see below)
**Config:** `configs/baseline/split_mnist_5task.yaml`
**Related:** `EXPERIMENT_LOG.md#EXP-002`

## Objective

Same dataset as EXP-001, harder sequence: 5 tasks of 2 digits each
(`[0,1] -> [2,3] -> [4,5] -> [6,7] -> [8,9]`) instead of 2 tasks of 5
digits each. Tests whether TMP's protection holds up over more
sequential drift events, and whether it decays with task age
(oldest-learned task vs. most-recently-learned task).

## How to reproduce

```bash
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method finetune
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method ewc
python run_multitask.py --config configs/baseline/split_mnist_5task.yaml --method tmp
python compare_multitask_results.py --log-dir outputs/experiments/EXP-002/logs --plot-dir outputs/experiments/EXP-002/plots
```

Diagnostics used during debugging:
```bash
python diagnose_task_confusion.py --config configs/baseline/split_mnist_5task.yaml --method tmp --victim-task <N>
python diagnose_gradient_conflict.py --config configs/baseline/split_mnist_5task.yaml --task-idx <N>
python sweep_tmp_lambda_multitask.py --config configs/baseline/split_mnist_5task.yaml
```

## Results summary

**One complete, statistically-tested comparison exists**, but it
PREDATES three bugs discovered afterward -- treat these numbers as
provisional, not final, until re-run with all fixes applied:

| Method | Mean retention (all epochs) | Mean learning accuracy |
|---|---|---|
| finetune | 60.2% | ~50% |
| ewc (lambda=20000) | 64.4% | ~47% |
| tmp (lambda=5.0) | 75.3% | ~41% |

Wilcoxon: TMP vs Finetune retention p<0.0001 (significant); TMP vs EWC
retention p=0.0001 (significant); TMP vs EWC learning accuracy p=0.755
(not significant -- no meaningful learning-speed cost).

## Conclusion (provisional)

TMP appears to outperform both baselines on this 5-task setup too, but
**this needs re-confirmation** after three bugs found via
`sweep_tmp_lambda_multitask.py` were fixed:

1. **Gradient explosion** in the surrogate loss's self-normalization
   (fixed: detached + clamped denominator, plus grad-norm clipping).
2. **Weight concentration**: the task-prioritized pool weighting could
   put >90% of all protective gradient on one struggling task (fixed:
   hard cap at 50% + redistribution).
3. **Cumulative dosage asymmetry**: task0 (protected since step 1)
   accumulates far more total protective pressure than later tasks
   regardless of per-step caps, diagnosed via a class-0 prediction
   collapse (fixed: `apply_every_n_steps` to reduce total dosage
   uniformly -- a blunt mitigation, not a complete structural fix; see
   OPEN_QUESTIONS.md).

Full detail on each bug, how it was diagnosed, and what evidence
confirmed the fix: `EXPERIMENT_LOG.md#EXP-002`.

## Next steps

1. Re-run the full finetune/ewc/tmp comparison with all three fixes
   active; confirm whether retention numbers hold, improve, or need
   further tuning.
2. If task2/task4 learning is still stuck near 0, use
   `diagnose_gradient_conflict.py` to test direct gradient opposition
   vs. magnitude dominance as the remaining mechanism.
3. Re-validate `tmp.lambda_` and `ewc.lambda_` specifically for the
   5-task setting via the sweep scripts -- current values are
   inherited starting points from EXP-001, not independently confirmed
   here.
