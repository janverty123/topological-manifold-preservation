# DESIGN PRINCIPLES

> **Purpose**
>
> This document defines the principles that should remain stable throughout the project.
>
> These are not temporary implementation choices.
>
> They are the rules that guide future decisions.

---

# Development Principles

Claude should preserve the existing project direction unless the user explicitly requests a redesign.

Incremental improvements are preferred over large architectural changes.

Recommendations should be supported by reasoning before implementation.

---

# Repository Principles

The repository should remain:

* organized
* modular
* reproducible
* easy to understand

Documentation should reduce repeated prompting rather than increase it.

---

# Implementation Principles

Prefer:

* reusable modules
* readable code
* consistent naming
* documented functions

Avoid:

* unnecessary complexity
* duplicated logic
* speculative optimization
* premature abstraction

---

# Experiment Principles

Every meaningful experiment should:

* have a clear objective
* produce reproducible results
* be documented
* support a specific implementation decision

Negative results should be documented rather than discarded.

---

# Claude Collaboration Principles

Claude should:

* understand existing code before suggesting changes
* avoid repeating previous suggestions
* use repository documentation as context
* recommend the smallest effective change first
* explain trade-offs when multiple solutions exist

---

# Documentation Principles

Documentation exists to answer questions quickly.

Each document has a single responsibility.

Avoid duplicating information across multiple files.

Temporary information belongs in:

* CURRENT_CONTEXT.md

Long-term information belongs in:

* PROJECT_STATUS.md
