# Project Status

Long-term status only. For "what's happening right now," see
CURRENT_CONTEXT.md instead -- this file changes slowly, that one
changes often.

## What this project is

An implementation of Topological Manifold Preservation (TMP), a
proposed regularization technique using Bottleneck Distance (persistent
homology) to mitigate catastrophic forgetting in continual learning,
per the approved research plan "Topological Manifold Preservation
(TMP): Utilizing Bottleneck Distance for Mitigating Catastrophic
Forgetting in Deep Neural Networks" (Chua, Oasan, Ople, Sales --
Adviser: Mr. Jose M. Manga Jr.).

## Current phase

**Empirical validation and debugging**, specifically: confirming TMP's
retention advantage over Finetune/EWC baselines holds up across
increasingly demanding test conditions within the approved scope
(Split-MNIST), and fixing real implementation bugs as they're
discovered through systematic hyperparameter sweeps and diagnostics.

Not yet started: final write-up / paper polishing (out of scope for
this repository -- this repo is the implementation + experimental
evidence, not the manuscript).

## Milestones

- [x] **M1** -- Working 2-task Split-MNIST pipeline (Finetune, EWC, TMP)
  with correct class-incremental evaluation. (EXP-001)
- [x] **M2** -- TMP's differentiable surrogate corrected to compare a
  fixed reference set against itself over time (not two different
  random samples). (EXP-001)
- [x] **M3** -- `tmp.lambda_` empirically validated via sweep on
  2-task setup; TMP shown to significantly outperform Finetune and EWC
  on retention with no significant learning-speed cost. (EXP-001,
  validated)
- [x] **M4** -- Generalized pipeline built for N sequential tasks
  (multi-task EWC, growing TMP reference pool), applied to 5-task
  Split-MNIST. (EXP-002)
- [ ] **M5** -- EXP-002 fully validated: all three known bugs
  (gradient explosion, weight concentration, cumulative dosage
  asymmetry) fixed AND the full 3-method comparison re-confirmed under
  the fixed code. **Currently in progress -- see CURRENT_CONTEXT.md.**
- [ ] **M6** -- (Deferred, not currently prioritized) Extend beyond
  Split-MNIST if time permits after M5 -- Permuted MNIST preferred
  over Split-CIFAR-100 if revisited, per prior scoping discussion.

## Repository philosophy

- **Two independent, non-shared pipelines** (2-task vs. multi-task) so
  that active debugging of one can never silently regress the other.
  See DESIGN_PRINCIPLES.md.
- **Every fix is diagnosed with evidence before being applied.** This
  project's history is full of "the numbers look wrong -> build a
  targeted diagnostic -> confirm the mechanism -> fix -> re-verify"
  cycles, not blind hyperparameter guessing. See EXPERIMENT_WORKFLOW.md
  and EXPERIMENT_LOG.md for the actual record of this.
- **Correctness over speed.** Several "obvious" fixes were tried,
  measured, and found insufficient or actively wrong before the real
  fix was identified (e.g. an additive weight floor was tried and
  rejected in favor of a hard cap once measured to be too weak). This
  is normal and expected for this kind of research code, not a sign of
  instability.
- **Documentation must never contradict the implementation.** If code
  changes and a doc file wasn't updated, the doc is wrong -- fix it,
  don't work around it.

## Overall objectives (from the approved research plan)

1. Design a TMP algorithm utilizing Bottleneck Distance to preserve
   structural features during sequential learning.
2. Implement and simulate the designed TMP algorithm in Python.
3. Analyze the graphs and numerical data from the simulated TMP
   algorithm.
4. Analyze the difference in terms of retention accuracy, learning
   rate, and computational overhead with Elastic Weight Consolidation
   (EWC) and Finetune.

See `docs/CODEBASE_GUIDE.md` for how each objective maps to actual code.
