# EXPERIMENT WORKFLOW

> **Purpose**
>
> Define the standard workflow for all TMP experiments.
>
> This workflow exists to keep experiments reproducible and reduce repeated prompting.

---

# Standard Workflow

```
Question

↓

Hypothesis

↓

Implementation

↓

Run Experiment

↓

Collect Metrics

↓

Analyze Results

↓

Update Documentation

↓

Next Experiment
```

---

# Before Running

Verify:

* Objective is clearly defined.
* Configuration is saved.
* Baseline is known.
* Expected outcome is documented.

---

# During Execution

Save:

* Training logs
* Configuration
* Metrics
* Important warnings
* Generated figures (if useful)

Do not save unnecessary temporary files.

---

# After Completion

Always update:

* EXPERIMENT_LOG.md

Update only if necessary:

* CURRENT_CONTEXT.md
* PROJECT_STATUS.md
* OPEN_QUESTIONS.md

---

# Required Outputs

Every important experiment should preserve:

* Configuration
* Metrics
* Logs
* Final results

If a figure significantly improves understanding, save it.

---

# Temporary Outputs

Do NOT commit:

* Cache
* Temporary checkpoints
* Intermediate tensors
* Temporary plots
* Debug files

---

# Git Policy

Commit:

* Source code
* Documentation
* Configurations
* Final metrics
* Important figures

Avoid committing generated files that can be recreated.

---

# Goal

Every experiment should be reproducible with minimal additional explanation.

Claude should be able to determine:

* what changed,
* why it changed,
* and what happened.
