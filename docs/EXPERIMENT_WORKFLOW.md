# Experiment Workflow

```
Question
   |
   v
Hypothesis
   |
   v
Implementation
   |
   v
Run Experiment
   |
   v
Collect Metrics
   |
   v
Analyze Results
   |
   v
Update Documentation
   |
   v
Next Experiment
```

## What each step means in this repo, concretely

- **Question** -> add to `OPEN_QUESTIONS.md` under Active.
- **Hypothesis** -> state a specific, falsifiable mechanism (e.g. "the
  weighting formula concentrates too much gradient on one task") --
  not just "TMP isn't working."
- **Implementation** -> build the smallest thing that tests the
  hypothesis first. Prefer a standalone diagnostic (see
  `diagnose_task_confusion.py`, `diagnose_gradient_conflict.py`) over
  changing training code blind.
- **Run Experiment** -> actually execute it. A hypothesis is not
  confirmed until measured.
- **Collect Metrics** -> use the existing JSONL logging + comparison
  scripts (`compare_results.py` / `compare_multitask_results.py`)
  rather than eyeballing terminal output.
- **Analyze Results** -> does the evidence support the hypothesis? Be
  willing to reject your own hypothesis if the data says so (this has
  happened multiple times in this project's real history -- see
  OPEN_QUESTIONS.md Q-009, where a plausible task-interference
  hypothesis was directly disproven by uniform cosine-similarity data).
- **Update Documentation** -> move the question to Resolved/Rejected in
  `OPEN_QUESTIONS.md`, add an entry to `EXPERIMENT_LOG.md`, and update
  `CURRENT_CONTEXT.md` if the immediate next task changed.
- **Next Experiment** -> update `CURRENT_CONTEXT.md`'s "Immediate next
  task" before ending the session.

## Rules

- Never skip straight from Question to Implementation without stating
  a hypothesis -- undirected changes are how bugs get "fixed" by
  accident and re-appear later.
- Never mark a fix as done in `OPEN_QUESTIONS.md` without a Run
  Experiment + Collect Metrics step confirming it. "This should fix
  it" is not a resolution.
- If a fix turns out insufficient (common in this project -- see the
  EXP-002 log), don't delete the old attempt's record. Log it as its
  own entry and let the log show the real sequence of attempts.
