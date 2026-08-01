# EXPERIMENT WORKFLOW

This document defines the standard workflow for conducting research in the Topological Manifold Preservation (TMP) project.

Its purpose is to ensure that every experiment is planned, executed, documented, and interpreted consistently.

---

# Research Lifecycle

Every research iteration should follow this sequence:

1. Identify an open research question.
2. Formulate a hypothesis.
3. Design the experiment.
4. Execute the experiment.
5. Analyze the evidence.
6. Record the findings.
7. Decide the next research direction.

No experiment should skip the documentation stage.

---

# Before Running an Experiment

Before starting, clearly define:

* Research question
* Hypothesis
* Configuration
* Dataset
* Evaluation metrics
* Expected outcome

The objective should be specific enough that the experiment can determine whether the hypothesis is supported.

---

# During the Experiment

Record:

* Configuration used
* Runtime information
* Training logs
* Important warnings
* Metrics
* Relevant plots

Large temporary files do not need to be preserved.

---

# After the Experiment

Every completed experiment should produce:

* Final metrics
* Relevant figures
* Training logs
* Configuration used
* Brief written conclusion

Then update:

* `EXPERIMENT_LOG.md`
* `RESEARCH_JOURNAL.md` (only if a new insight was obtained)
* `PROJECT_STATUS.md` (only if the research direction changed)
* `OPEN_QUESTIONS.md` (if a question was answered or a new one emerged)

---

# Experiment Records

Each experiment should include:

* Experiment ID
* Objective
* Hypothesis
* Configuration
* Metrics
* Results
* Interpretation
* Next action

The experiment itself is the unit of organization.

---

# Evidence Policy

Research conclusions should only be made when supported by experimental evidence.

Ideas without evidence remain hypotheses.

Negative results are valuable and should be documented rather than discarded.

---

# Repository Policy

Commit:

* Source code
* Documentation
* Configurations
* Small summary metrics
* Important figures

Do not commit:

* Large temporary outputs
* Cache files
* Intermediate checkpoints unless necessary
* Generated files that can easily be reproduced

---

# Continuous Improvement

This workflow may evolve as the project matures, but every modification should improve consistency, traceability, and scientific rigor.
