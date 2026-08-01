# Tests

No automated test suite exists yet. Validation for this project has
so far been done via:
- Direct functional smoke tests during development (synthetic data,
  since MNIST download isn't always available in every environment)
- The diagnostic scripts (`diagnose_task_confusion.py`,
  `diagnose_gradient_conflict.py`) acting as targeted correctness
  checks for specific mechanisms
- The lambda sweep scripts acting as an end-to-end sanity check that
  the regularizer responds sensibly to its main hyperparameter

If a proper test suite is added later (e.g. pytest), it belongs here.
Until then, this folder exists to match the target repository
structure but is intentionally empty rather than populated with
placeholder tests that don't test anything real.
