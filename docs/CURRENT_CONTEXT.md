# Topological Manifold Preservation (TMP)

> **AI-Assisted Research Repository**

This repository contains the implementation, experiments, and documentation for **Topological Manifold Preservation (TMP)**, a continual learning framework designed to mitigate catastrophic forgetting using topological constraints derived from Bottleneck Distance.

The repository is primarily optimized for **AI-assisted development**, specifically Claude Projects, to minimize repeated prompting and maintain consistent project context.

---

# Repository Purpose

The purpose of this repository is to:

* Develop and evaluate the TMP framework.
* Compare TMP against Finetune and Elastic Weight Consolidation (EWC).
* Maintain reproducible experiments.
* Provide Claude with enough context to continue development with minimal user prompting.

---

# Current Goal

The project is currently focused on improving the existing TMP implementation until it performs competitively with EWC.

Success is defined as:

* outperforming EWC,
* matching EWC,
* or achieving performance close enough that TMP's theoretical advantages justify its use.

---

# Repository Structure

```text
configs/        Experiment configurations

docs/           Project documentation

scripts/        Training / evaluation entry points

src/            Source code

tests/          Testing

outputs/        Generated experiment outputs
```

---

# Documentation

| File                   | Purpose                                    |
| ---------------------- | ------------------------------------------ |
| CURRENT_CONTEXT.md     | Current project state for Claude           |
| CLAUDE_PROJECT.md      | Claude collaboration instructions          |
| PROJECT_STATUS.md      | Overall implementation progress            |
| DESIGN_PRINCIPLES.md   | Rules that should not change               |
| OPEN_QUESTIONS.md      | Current unresolved implementation problems |
| EXPERIMENT_LOG.md      | History of experiments                     |
| EXPERIMENT_WORKFLOW.md | Standard experiment workflow               |

---

# Development Philosophy

This repository prioritizes:

1. Correct implementation
2. Reproducibility
3. Clear documentation
4. Efficient AI collaboration

---

# First Steps for Claude

Before inspecting code, Claude should read:

1. docs/CURRENT_CONTEXT.md
2. docs/CLAUDE_PROJECT.md
3. docs/PROJECT_STATUS.md

Only then should source files be inspected.

---

# Notes

This repository is under active development.

Documentation is optimized for rapid AI-assisted iteration rather than publication.

# KNOWN CONTEXT

> This file exists to eliminate repeated prompting.
>
> Claude should read this before proposing solutions.

---

# Project

Topological Manifold Preservation (TMP)

---

# Current Goal

Improve TMP until it performs competitively with EWC.

---

# Success Criteria

* Better than EWC
* Equal to EWC
* Slightly lower than EWC if justified by theoretical advantages

---

# Fixed Decisions

Do not repeatedly suggest changing these:

* Bottleneck Distance remains the topological metric.
* Continual Learning remains the problem domain.
* EWC is the primary comparison.
* Finetune is the secondary comparison.

---

# Already Investigated

Maintain this list.

Example:

* Gradient normalization
* Surrogate weighting
* Loss scaling

---

# Do Not Repeat

Add ideas here after they have been discussed extensively.

Example:

* Redesign the entire framework
* Replace Bottleneck Distance
* Change the research objective

---

# Current Files of Interest

Update when necessary.

Example:

* src/losses/
* src/training/
* configs/

---

# Next Immediate Task

Only one item should appear here.

This is the task Claude should focus on first.

Example:

Investigate why new-task learning degrades despite stable retention.
