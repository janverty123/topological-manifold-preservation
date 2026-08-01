# Topological Manifold Preservation

> **Research Repository**
>
> This repository contains the ongoing research, experiments, documentation, and implementation for the **Topological Manifold Preservation (TMP)** framework. The repository is designed to serve as the single source of truth for both human collaborators and AI research assistants throughout the lifetime of the project.

---

# Research Overview

Topological Manifold Preservation (TMP) is a research project investigating methods for preserving learned knowledge while maintaining the underlying topological structure of learned representations.

Rather than optimizing solely for benchmark performance, TMP emphasizes scientific understanding, interpretability, and the preservation of meaningful geometric and topological relationships during continual learning.

This repository documents not only the implementation, but also the reasoning, experimental evidence, and research decisions that guide the project.

---

# Research Objectives

The project aims to:

* Develop a continual learning framework based on topological manifold preservation.
* Investigate catastrophic forgetting through the lens of topology and representation geometry.
* Compare TMP against existing continual learning approaches.
* Produce reproducible experimental evidence supporting the proposed methodology.
* Maintain a transparent record of research decisions and experimental outcomes.

---

# Current Research Status

The current state of the project is maintained in:

* `docs/PROJECT_STATUS.md`

This document should always reflect the latest confirmed understanding of the research.

---

# Repository Structure

```text
configs/        Experiment configurations
docs/           Research documentation
scripts/        Executable scripts
src/            Source code
tests/          Tests
outputs/        Generated experiment outputs
```

---

# Documentation Guide

The documentation is intentionally divided into focused documents.

| Document               | Purpose                                                   |
| ---------------------- | --------------------------------------------------------- |
| `PROJECT_STATUS.md`    | Current state of the research                             |
| `DESIGN_PRINCIPLES.md` | Scientific and engineering principles guiding the project |
| `OPEN_QUESTIONS.md`    | Active research questions and unresolved problems         |
| `EXPERIMENT_LOG.md`    | Record of completed experiments                           |
| `RESEARCH_JOURNAL.md`  | Chronological record of important discoveries             |
| `CLAUDE_PROJECT.md`    | Instructions for AI collaborators                         |

Each document has a single responsibility to reduce duplication and improve long-term maintainability.

---

# Research Workflow

The intended research workflow is:

1. Identify an open research question.
2. Design an experiment.
3. Execute the experiment.
4. Record the results.
5. Interpret the evidence.
6. Update the documentation if new knowledge is obtained.

Research conclusions should always be supported by experimental evidence.

---

# AI Collaboration

This repository is designed to work alongside a Claude Project.

AI assistants should:

* Read the documentation before suggesting changes.
* Distinguish confirmed findings from hypotheses.
* Explain reasoning before recommending modifications.
* Preserve research history rather than rewriting it.
* Prioritize scientific correctness over benchmark improvements.

The complete collaboration protocol is documented in:

`docs/CLAUDE_PROJECT.md`

---

# Getting Started

1. Clone the repository.
2. Create the project environment using `environment.yml`.
3. Review the documentation in the following order:

   1. `README.md`
   2. `docs/PROJECT_STATUS.md`
   3. `docs/DESIGN_PRINCIPLES.md`
   4. `docs/OPEN_QUESTIONS.md`
   5. `docs/CLAUDE_PROJECT.md`

Only after understanding the current research state should new experiments or modifications be proposed.

---

# Future Publication

This repository supports ongoing academic research.

Publication details, preprints, and citation information will be added when available.

---

# License

(Add your preferred license here.)
