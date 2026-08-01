# Experiment Log

Structured record of every experiment. Every entry follows the same
fields (see EXPERIMENT_WORKFLOW.md for the process this log tracks).
Entries are numbered `<EXP-ID>-<sequence>` and never edited after the
fact -- if something turns out wrong, add a NEW entry that supersedes
it and note that in the "Next Action" of the old one.

---

## EXP-001: Split-MNIST 2-Task Baseline

### EXP-001-01
- **Objective:** Build a working 2-task Split-MNIST pipeline (Finetune, EWC, TMP).
- **Motivation:** Implement the approved research plan's Methodology section.
- **Files Modified:** `src/data.py`, `src/models.py`, `src/train.py`, `src/losses.py`, `src/ewc.py`, `src/tda_utils.py`, `scripts/*.py`.
- **Configuration:** `configs/baseline/split_mnist_2task.yaml` (initial values).
- **Metrics:** N/A (initial build).
- **Results:** Pipeline runs end to end; Windows `DataLoader(num_workers>0)` crash found and fixed (lambda transform not picklable -> replaced with `_FlattenTransform` class).
- **Conclusion:** Working baseline established.
- **Next Action:** Run Finetune and inspect retention metric.

### EXP-001-02
- **Objective:** Sanity-check Finetune's retention behavior.
- **Motivation:** First real run showed retention staying suspiciously high (82-92%) for an "unprotected" baseline.
- **Files Modified:** `src/train.py` (added `retention_accuracy_task_incremental` alongside unmasked `retention_accuracy`).
- **Configuration:** unchanged.
- **Metrics:** Masked retention ~85-92%; unmasked (corrected) retention dropped to 24% by epoch 10.
- **Results:** Confirmed the masked (task-incremental) evaluation was hiding real forgetting.
- **Conclusion:** Unmasked, class-incremental accuracy is the correct primary retention metric. See OPEN_QUESTIONS.md (folded into general project decisions, not a standalone Q).
- **Next Action:** Run EWC with corrected metric.

### EXP-001-03
- **Objective:** Get EWC to show a meaningful retention improvement over Finetune.
- **Motivation:** Initial `ewc.lambda_=400` produced almost no improvement (27% vs. Finetune's 24%).
- **Files Modified:** `src/train.py` (added `mean_fisher_value` diagnostic print), `configs/baseline/split_mnist_2task.yaml`.
- **Configuration:** `ewc.lambda_`: 400 -> 20000.
- **Metrics:** `mean_fisher_value` measured at 1.528e-06 (far smaller than assumed). At lambda=20000: retention 68.87% by epoch 10.
- **Results:** Confirmed via direct measurement, not guesswork -- see Q-002.
- **Conclusion:** EWC lambda must be tuned to the actual Fisher magnitude scale, no universal default works.
- **Next Action:** Run TMP and compare.

### EXP-001-04
- **Objective:** Diagnose why TMP performed WORSE than Finetune (11-19% retention).
- **Motivation:** First TMP run underperformed the unprotected baseline.
- **Files Modified:** `src/losses.py` (surrogate comparing wrong activations -- fixed to use FIXED Task-1 reference set), `src/train.py`.
- **Configuration:** unchanged (`tmp.lambda_=0.5`).
- **Metrics:** Retention still weak after fix (drift climbing every epoch, 8-11 range).
- **Results:** Root cause was TWO bugs, found sequentially: (1) surrogate compared different images (see Q-001), (2) surrogate loss unnormalized + adaptive lambda uncapped, causing runaway feedback (see Q-003). Also found: true `W_inf` metric itself was confounded by raw activation-scale growth (see Q-004).
- **Conclusion:** Three compounding bugs, each independently necessary to fix.
- **Next Action:** Re-run after all three fixes.

### EXP-001-05
- **Objective:** Validate `tmp.lambda_` empirically instead of guessing.
- **Motivation:** All three EXP-001-04 fixes applied; needed a real sweep, not a single value.
- **Files Modified:** `sweep_tmp_lambda.py` (new).
- **Configuration:** Swept `tmp.lambda_` in {0.0, 0.5, 1.0, 2.0, 5.0, 10.0}.
- **Metrics:** retention @ lambda=0.0: 24.13% (matches Finetune, confirms ablation correctness). @ lambda=5.0: 96.28%, learning_acc 98.81%. @ lambda=10.0: 96.03% (past the peak).
- **Results:** Clean, sensible dose-response curve -- huge jump 0->0.5, diminishing returns after, slight dip at 10.0.
- **Conclusion:** `tmp.lambda_=5.0` is the validated sweet spot for the 2-task setup.
- **Next Action:** Full 3-method comparison with validated hyperparameters -> EXP-001 considered validated.

**EXP-001 final status: validated.** See `outputs/experiments/EXP-001/README.md` for the results table.

---

## EXP-002: Split-MNIST 5-Task Stress Test

### EXP-002-01
- **Objective:** Generalize the 2-task pipeline to N sequential tasks.
- **Motivation:** Test whether TMP's protection holds up over more, smaller sequential tasks within the same dataset (per research-team scoping decision, staying within Split-MNIST rather than adding new datasets).
- **Files Modified:** `src/train_general.py`, `src/datasets_extended.py` (new), `run_multitask.py` (new), `configs/baseline/split_mnist_5task.yaml` (new).
- **Configuration:** 5 tasks of 2 digits each, inherited `tmp.lambda_=5.0` / `ewc.lambda_=20000.0` from EXP-001 as starting points.
- **Metrics:** N/A (initial build).
- **Results:** Pipeline runs end to end.
- **Conclusion:** Generalization mechanically works; hyperparameters not yet re-validated for this harder setting.
- **Next Action:** Run all three methods and compare.

### EXP-002-02
- **Objective:** First full 3-method comparison on 5-task Split-MNIST.
- **Motivation:** Establish a baseline result to iterate on.
- **Files Modified:** `src/evaluate_multitask.py`, `src/visualize_multitask.py`, `compare_multitask_results.py` (all new).
- **Configuration:** inherited hyperparameters, unchanged.
- **Metrics:** TMP retention 75.3%, EWC 64.4%, Finetune 60.2% (mean across sequence). Wilcoxon: TMP vs Finetune p<0.0001, TMP vs EWC p=0.0001, TMP vs EWC learning-speed difference p=0.755 (not significant).
- **Results:** Statistically significant retention advantage for TMP, no significant learning-speed cost.
- **Conclusion:** Promising initial result -- **but see EXP-002-05 through 07: this run predates three later-discovered bugs and needs re-confirmation.**
- **Next Action:** Recommendations for further improvement requested and implemented (see EXP-002-03/04).

### EXP-002-03
- **Objective:** Implement 5 improvement recommendations (EMA lambda schedule, confusion diagnostics, weighted reference pool, lambda sweep for multi-task, per-task retention trajectories).
- **Motivation:** External review of EXP-002-02's results suggested concrete improvements, particularly around Task 1's comparatively weak retention.
- **Files Modified:** `src/train_general.py` (`AdaptiveLambdaScheduler`, `MultiTaskTMPReference.weighted_surrogate_loss`), `src/evaluate_multitask.py` (`build_retention_trajectory_matrix`), `src/visualize_multitask.py` (`plot_task_retention_trajectories`), `diagnose_task_confusion.py` (new), `sweep_tmp_lambda_multitask.py` (new).
- **Configuration:** added `tmp.ema_alpha`, `tmp.lambda_max_multiplier`.
- **Metrics:** N/A (feature build).
- **Results:** All 5 pieces implemented and smoke-tested with synthetic data.
- **Conclusion:** Ready for real-data testing.
- **Next Action:** Run the multi-task lambda sweep.

### EXP-002-04
- **Objective:** Confirm the lambda sweep behaves sensibly after EXP-002-03's changes.
- **Motivation:** Direct test of whether lambda now has a predictable effect.
- **Files Modified:** none (testing existing code).
- **Configuration:** Swept `tmp.lambda_` in {0.0, 2.5, 5.0, 7.5, 10.0, 15.0}.
- **Metrics:** EVERY nonzero lambda collapsed learning accuracy to near-zero (0.0000-0.0066), non-monotonically -- e.g. lambda=10.0 outperformed lambda=2.5/5.0/7.5.
- **Results:** NOT healthy regularization behavior -- a real bug, not "TMP doesn't work here."
- **Conclusion:** Something is structurally broken, not just a tuning issue.
- **Next Action:** Diagnose the gradient behavior directly.

### EXP-002-05
- **Objective:** Diagnose why every nonzero lambda collapses learning regardless of magnitude.
- **Motivation:** EXP-002-04's non-monotonic collapse pattern.
- **Files Modified:** `src/losses.py` (detached + clamped normalization denominator), `src/train_general.py` (grad-norm clipping + logging).
- **Configuration:** unchanged.
- **Metrics:** Directly measured: gradient norm under near-collapse went from ~11 (mean pairwise dist ~1e-2) to ~5.8 million (mean pairwise dist ~1e-9) under the OLD formula; stayed bounded ~2,500 under the NEW (detached/clamped) formula across the same range.
- **Results:** Confirmed root cause: live (non-detached) normalization denominator's gradient blows up as activations partially collapse. A small, well-behaved forward loss value said nothing about backward gradient magnitude.
- **Conclusion:** Fixed (see Q-007). Standard "stop-gradient on statistics" technique applied (as used in BYOL/SimSiam).
- **Next Action:** Re-run the sweep.

### EXP-002-06
- **Objective:** Confirm EXP-002-05's fix resolved the sweep pathology.
- **Motivation:** Re-run after the gradient-explosion fix.
- **Files Modified:** none (testing).
- **Configuration:** Same sweep as EXP-002-04.
- **Metrics:** Grad norms now bounded (max ~87, and this is the PRE-clip value clip_grad_norm_ reports). But learning accuracy STILL fully stuck at 0.0000 for tasks 2 and 4 at every nonzero lambda tested.
- **Results:** Partial fix -- numerical explosion resolved, but a SECOND distinct bug remained.
- **Conclusion:** Directly tested the weighting formula with realistic accuracy values (task1@40%, others@95%+): confirmed 90.9% of ALL protective gradient concentrated on task1 alone.
- **Next Action:** Fix the weight concentration. Tried additive floor first (weak: only reduced to 85.7%), then hard cap + redistribution (effective: reduced to 50.0%). Applied the hard cap (`_cap_and_redistribute`, default `max_weight_cap=0.5`).

### EXP-002-07
- **Objective:** Confirm the weight-cap fix resolved remaining collapse; diagnose any residual issue.
- **Motivation:** Re-run after EXP-002-06's fix.
- **Files Modified:** `diagnose_task_confusion.py` (added `analyze_output_layer_bias`), `diagnose_gradient_conflict.py` (new), `src/train_general.py` (`tmp.apply_every_n_steps`).
- **Configuration:** Same sweep. Added `apply_every_n_steps: 2` to `configs/baseline/split_mnist_5task.yaml`.
- **Metrics:** Task3 now learns properly (smooth growth to 26%). Task2 and Task4 STILL fully stuck at 0.0000. Confusion matrix: ~75-80% of all misclassifications collapsed to class 0. Cross-task cosine similarity uniformly high (0.96-0.99) across every task pair -- ruled out task-specific interference.
- **Results:** Mixed outcome disproved the "task-specific interference" hypothesis (similarity data was uniform, not differential). Pointed instead to cumulative protective-dosage asymmetry: task0 protected since step 1, accumulating far more total pressure than later tasks regardless of the per-step cap.
- **Conclusion:** Applied `apply_every_n_steps` as a uniform dosage-reduction mitigation. NOT yet re-confirmed whether this fully resolves task2/task4 (see Q-009, Q-011).
- **Next Action:** Re-run the full sweep AND the full 3-method comparison with all three fixes active. **This is EXP-002's current blocking task -- see CURRENT_CONTEXT.md.**

**EXP-002 current status: active_debugging.** Three bugs found and fixed
in sequence; final confirmation still pending. Do not treat
EXP-002-02's headline numbers as final until a fresh run under the
current code is logged as EXP-002-08 or later.
