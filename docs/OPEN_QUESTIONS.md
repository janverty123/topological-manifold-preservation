# Open Questions

Tracks unresolved (and resolved) implementation questions. Questions
move between sections as they're investigated -- **never delete a
question**, move it to Resolved or Rejected with the outcome recorded
instead, so the reasoning trail stays intact.

---

## Active

### Q-010: Does EXP-002's TMP improvement hold after all three bugfixes?
The 75.3%/64.4%/60.2% (TMP/EWC/Finetune) retention comparison was
measured BEFORE the gradient-explosion, weight-concentration, and
cumulative-dosage fixes. Needs a fresh full run to confirm.
- **Status:** blocking CURRENT_CONTEXT.md's immediate next task.

### Q-011: Is `apply_every_n_steps` a sufficient fix for cumulative dosage asymmetry, or just a mitigation?
The fix reduces total protective dosage uniformly across all tasks,
but doesn't explicitly equalize task0's much-longer total exposure
against later tasks. If task2/task4 are still stuck after re-running
with this fix, a more structural fix (explicit per-task cumulative
protection budget) may be needed.
- **Status:** cannot resolve until Q-010's re-run completes.

### Q-012: Are `tmp.lambda_=5.0` / `ewc.lambda_=20000.0` actually correct for 5 tasks, or just inherited from the 2-task result?
Never independently re-validated via `sweep_tmp_lambda_multitask.py`
under the CURRENT (fixed) code. The one sweep run that was attempted
was under buggy code and shouldn't be trusted.
- **Status:** waiting on Q-010.

---

## Waiting for Experiments

### Q-005: Does TMP's retention advantage decay with task age (protection weaker for earlier-learned tasks)?
`per_task_retention` / `mt_retention_trajectories_<method>.png` were
built specifically to answer this, but haven't been analyzed on a
POST-bugfix run yet -- prior trajectory data predates the fixes.
- **Status:** re-run needed (same run as Q-010 will produce this data).

### Q-006: If Q-011's mitigation isn't enough, would an explicit per-task cumulative-protection budget work better?
Design sketch: track total protective gradient magnitude each task has
received over the whole run, and additionally weight down tasks that
have already received "enough" cumulative protection, not just tasks
currently doing well.
- **Status:** only worth building if Q-011 comes back "no."

---

## Resolved

### Q-001: Why is TMP's retention lower than expected on early runs? (RESOLVED)
The differentiable surrogate was comparing the CURRENT Task-2 batch's
activations against the Task-1 BASELINE cloud -- different images
entirely, a meaningless comparison. Fixed by using a FIXED Task-1
reference set re-forwarded through the current model every step.
- **Resolution:** src/losses.py + src/train.py, EXP-001.

### Q-002: Why did increasing EWC's lambda from 400 to 20000 seem necessary? (RESOLVED)
Measured `mean_fisher_value` directly and found it ~1.5e-6 -- far
smaller than assumed, so `lambda_=400` produced a negligible penalty
regardless of Fisher-weighted theory being correct.
- **Resolution:** confirmed via direct measurement, not guesswork.
  EXP-001.

### Q-003: Why did TMP sometimes perform WORSE than unprotected Finetune? (RESOLVED)
Two compounding causes: (1) the differentiable surrogate loss was
unnormalized, so its raw gradient magnitude could dominate/destabilize
training regardless of how small lambda was set; (2) the adaptive
lambda rescaling had no cap, creating a runaway feedback loop (more
drift -> higher lambda -> more drift). Fixed by normalizing the
surrogate loss and capping the rescale multiplier.
- **Resolution:** src/losses.py, src/train.py. EXP-001.

### Q-004: Why did the TRUE (giotto-tda) bottleneck distance keep climbing every epoch even after the surrogate normalization fix? (RESOLVED)
The persistence diagrams were built from RAW, un-normalized activation
magnitudes, which naturally grow during training regardless of actual
forgetting -- a confound unrelated to real structural drift. Fixed by
adding `tda_utils.normalize_point_cloud` (center + scale by mean
pairwise distance) before building any persistence diagram.
- **Resolution:** src/tda_utils.py. EXP-001.

### Q-007: Why did EVERY nonzero lambda collapse multi-task learning to near-zero regardless of magnitude? (RESOLVED)
The surrogate loss's normalization denominator was a LIVE (non-detached)
mean, whose gradient contains a 1/mean(x) term that explodes as the
mean shrinks -- invisible from the forward-pass loss value alone.
Verified directly: gradient norm went from ~11 to ~5.8 million as mean
pairwise distance dropped from 1e-2 to 1e-9. Fixed by detaching and
floor-clamping the denominator (the "stop-gradient on statistics"
technique from BYOL/SimSiam), plus added grad-norm clipping as a
permanent safety net.
- **Resolution:** src/losses.py, src/train_general.py. EXP-002-05.

### Q-008: After fixing Q-007, why was learning STILL stuck near zero at any nonzero lambda? (RESOLVED)
The task-prioritized weighting formula `w_k = (1-acc_k)/sum(1-acc_j)`
had no concentration limit -- verified directly that a struggling task
(40% accuracy) could consume 90.9% of the ENTIRE protective gradient
budget every step. An additive floor was tried first and found too
weak (only reduced concentration to 85.7%). Fixed with a hard cap
(default 50%) plus proportional redistribution among remaining tasks.
- **Resolution:** src/train_general.py
  (`MultiTaskTMPReference._cap_and_redistribute`). EXP-002-06.

### Q-009: After fixing Q-008, why were task2/task4 STILL fully stuck at 0.0000 learning accuracy while task3 recovered normally?
Confusion matrix analysis showed ~75-80% of ALL misclassifications
collapsing to class 0 specifically, not spread across plausible
confusions. Cross-task activation cosine similarity was uniformly high
(0.96-0.99) across every task pair, ruling out task-specific
representational overlap. Root cause: task0 is protected on every
single step across the full run, so it accumulates far more TOTAL
protective pressure than later tasks even with the per-step cap from
Q-008. Confirmed via a new diagnostic (`analyze_output_layer_bias`)
that flags disproportionate per-class output weight norms / mean
logits. Mitigated (not fully structurally fixed -- see Q-011) via
`tmp.apply_every_n_steps`.
- **Resolution:** src/train_general.py, diagnose_task_confusion.py.
  EXP-002-07.

---

## Rejected

### Q-R1: Should the multi-task pipeline reuse `src/train.py` directly instead of a separate `train_general.py`?
Considered and rejected -- see DESIGN_PRINCIPLES.md #1. The risk of
regressing the validated EXP-001 pipeline while EXP-002 is under
active, iterative debugging outweighs the code-duplication cost.

### Q-R2: Should we add Permuted MNIST / Split-CIFAR-100 back in now?
Considered, implemented once, then explicitly descoped to focus on
fully validating Split-MNIST (2-task and 5-task) first, matching the
approved research plan's actual scope. CNNClassifier and the CIFAR/
Permuted MNIST dataset builders were removed from this repo copy.
Revisit only after PROJECT_STATUS.md's M5 milestone is complete, and
prefer Permuted MNIST over Split-CIFAR-100 if revisited (faster to
train, stays within MLP architecture, lower risk).
