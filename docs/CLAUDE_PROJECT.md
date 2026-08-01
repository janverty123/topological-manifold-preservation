# Claude Project Instructions

This file exists to reduce repeated prompting. Read the files below IN
THIS ORDER before inspecting source code or answering questions about
this project's status, results, or next steps.

## Reading order

1. **`CURRENT_CONTEXT.md`** -- what's happening right now, the
   immediate next task, current known issues. Changes frequently;
   always read this first, even if you've read it before in an
   earlier session.
2. **`README.md`** -- repository overview, what this project is, where
   everything lives.
3. **`PROJECT_STATUS.md`** -- long-term phase, milestones, philosophy.
   Changes slowly.
4. **`DESIGN_PRINCIPLES.md`** -- stable rules for how this codebase is
   built and organized. Don't propose changes that violate these
   without flagging the conflict explicitly first.
5. **`OPEN_QUESTIONS.md`** -- full history of what's been investigated,
   resolved, and rejected. Check here BEFORE re-investigating something
   that looks like a new problem -- it may already have a documented
   answer or a documented reason it was rejected.
6. **`EXPERIMENT_LOG.md`** -- the actual chronological record of every
   experiment run, including ones that didn't work. Read this to
   understand HOW the project got to its current state, not just what
   the current state is.

Only after reading all six should you inspect source code directly.
When you do, start with `docs/CODEBASE_GUIDE.md` rather than opening
files at random -- it maps objectives to actual file/function
locations, verified against the real code.

## Collaboration rules

- **Don't ask the user to re-explain project context that's already in
  these files.** If something is genuinely missing or stale, say so
  and update the relevant file rather than asking the same question
  every session.
- **Don't re-propose fixes already in `OPEN_QUESTIONS.md`'s Rejected
  section** without new evidence that changes the situation.
- **Don't assume a result is final just because a number looks good.**
  This project's real history (see `EXPERIMENT_LOG.md`) is full of
  results that looked validated and later turned out to predate a real
  bug. Check `CURRENT_CONTEXT.md`'s "known issues" before treating any
  number as settled.
- **When you fix something, update the documentation in the same
  session**: move the question in `OPEN_QUESTIONS.md`, add the
  `EXPERIMENT_LOG.md` entry, update `CURRENT_CONTEXT.md`'s next task.
  Don't leave this for "later" -- later means the next session starts
  from stale context again.
- **Preserve the two-pipeline separation** (`src/train.py` vs.
  `src/train_general.py`) unless `DESIGN_PRINCIPLES.md` #1's stated
  conditions for merging them are actually met.
- **Never silently change a research parameter** (hyperparameter,
  dataset split, architecture) as a side effect of an organizational
  or documentation task. If a parameter needs to change, that's an
  experiment -- log it as one.
- **When editing YAML configs**, only touch the `experiment:` /
  `output:` metadata blocks freely; treat every other key as a
  research parameter requiring the same care as a code change.
- **If the person's request conflicts with something in
  `DESIGN_PRINCIPLES.md` or a `Rejected` entry in `OPEN_QUESTIONS.md`**,
  say so explicitly and ask whether they want to proceed anyway, rather
  than silently overriding prior reasoning.
