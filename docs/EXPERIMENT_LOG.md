# EXPERIMENT LOG

> **Purpose**
>
> This document records all meaningful experiments performed during the project.
>
> It is intended to prevent repeating previous work and to provide Claude with a searchable history of implementation changes.
>
> Every experiment should receive a unique Experiment ID.

---

# Experiment Template

---

## Experiment ID

EXP-XXX

---

### Date

YYYY-MM-DD

---

### Objective

What was the purpose of this experiment?

---

### Motivation

Why was this experiment performed?

What problem was it trying to solve?

---

### Files Modified

List every important file.

Example:

* src/losses/tmp_loss.py
* src/train.py
* configs/tmp.yaml

---

### Configuration

Document important settings.

Example:

* Learning Rate
* Lambda
* Epochs
* Batch Size
* Dataset
* Random Seed

---

### Metrics

Always record:

* Final Accuracy
* Average Accuracy
* Forgetting Measure
* Training Time

Additional metrics may be added when necessary.

---

### Results

Summarize the outcome objectively.

Do not explain why.

Simply state what happened.

---

### Conclusion

Was the experiment successful?

Choose one:

* Improved
* No Change
* Worse
* Inconclusive

---

### Next Action

Describe the next experiment that should follow.

---

# Search Index

Maintain this section for quick navigation.

| ID      | Objective | Outcome |
| ------- | --------- | ------- |
| EXP-001 |           |         |
| EXP-002 |           |         |
| EXP-003 |           |         |

---

# Rules

Every meaningful experiment should be recorded.

Small debugging changes do not require an experiment entry.

Never delete experiments.

If an experiment becomes obsolete, mark it as superseded.
