# Topological Manifold Preservation (TMP) — Technical Execution Guide

Reference implementation for the research plan *"Topological Manifold
Preservation (TMP): Utilizing Bottleneck Distance for Mitigating
Catastrophic Forgetting in Deep Neural Networks"* (Chua, Oasan, Ople,
Sales — Adviser: Mr. Jose M. Manga Jr.).

This guide takes you from a clean machine to fully reproduced results
(logs, saved models, persistence diagrams, comparative plots) that
directly answer Research Questions 1–3 and test Hypotheses H0/H1.

---

## 1. System & Environment Setup

### 1.1 Hardware / software prerequisites

| Requirement | Minimum | Recommended | Maps to |
|---|---|---|---|
| OS | Ubuntu 20.04+ / macOS 12+ / Windows 10+ (WSL2) | Ubuntu 22.04 | Simulation environment |
| Python | 3.9+ | 3.12 | Methodology I |
| RAM | 8 GB | 16 GB | Computational Overhead metric |
| GPU | Optional (CPU works, MLP is small) | NVIDIA GPU + CUDA 11.8/12.1 | Faster Task-2 training |
| Disk | 2 GB free (MNIST + logs + plots) | 5 GB | Data Collection |

Core library versions (pinned in `requirements.txt` / `environment.yml`):

- `torch >= 2.1.0`, `torchvision >= 0.16.0` — PyTorch network + Split-MNIST
  data (Methodology I, "leveraging PyTorch to build and execute the
  neural architectures")
- `giotto-tda >= 0.6.0` — persistent homology, Vietoris-Rips filtration,
  Bottleneck Distance (Methodology I, "giotto-tda to extract and
  structure the topological properties")
- `scikit-learn`, `numpy<2.0`, `scipy` — Maxmin sampling, pairwise
  distances, Wilcoxon significance test
- `matplotlib`, `seaborn` — comparative line graphs and heatmaps
- `pandas`, `pyyaml`, `tqdm`, `psutil` — data wrangling, config, memory
  profiling for the Computational Overhead metric

> **Note:** `giotto-tda` requires `numpy<2.0`. If you have other
> packages that force `numpy>=2`, use a dedicated virtual environment
> (see below) — do not install this project into a shared/global
> environment.

### 1.2 Installation — Option A: `venv` + `pip`

```bash
# 1. Clone / unzip the project, then move into it
cd tmp

# 2. Create an isolated virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the install
python -c "import torch, gtda, sklearn, matplotlib; print('OK:', torch.__version__, gtda.__version__)"
```

### 1.3 Installation — Option B: Conda

```bash
conda env create -f environment.yml
conda activate tmp
python -c "import torch, gtda; print('OK')"
```

### 1.4 GPU check (optional but recommended)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**If this prints `False` and you have an NVIDIA GPU:** `pip install torch` from
plain PyPI installs the **CPU-only** build on Windows and macOS by default —
the CUDA-enabled builds live on PyTorch's own package index, not PyPI. Fix it
with:

```bash
# 1. Remove the CPU-only build
pip uninstall torch torchvision -y

# 2. Check your driver's max supported CUDA version
nvidia-smi   # look for "CUDA Version: XX.X" in the top-right

# 3. Install the CUDA build matching (or older than) that version -- a
#    wheel built for a NEWER CUDA runtime than your driver supports can
#    fail to initialize, so don't just grab the newest one.
#    e.g. "CUDA Version: 12.4" -> use cu124:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    Other common options: cu118, cu121, cu126, cu128 (RTX 50-series)

# 4. Verify
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

If `False`, everything still runs on CPU — the MLP architecture used
here (784→256→128→10) is small enough that CPU training on Split-MNIST
completes in a few minutes.

### 1.5 Troubleshooting — Setup

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: numpy.core.multiarray failed to import` | numpy 2.x installed alongside giotto-tda | `pip install "numpy<2.0.0" --force-reinstall` |
| `giotto-tda` build fails on Apple Silicon | missing compiler toolchain | `xcode-select --install`, then retry, or use conda-forge build |
| `RuntimeError: CUDA out of memory` | unrelated to this project (model is tiny) — usually another process | Restart Python kernel / check `nvidia-smi` |
| MNIST download hangs / fails | no internet access or corporate proxy | Set `HTTP_PROXY`/`HTTPS_PROXY` env vars, or manually place MNIST raw files under `data/MNIST/raw/` |

---

## 2. Project Architecture & File Directory Structure

```
tmp/
├── README.md                  # This guide
├── requirements.txt           # pip dependencies
├── environment.yml            # conda dependencies
├── run_all.py                 # orchestrator: runs the entire pipeline end-to-end
├── compare_results.py         # Step 5: statistical comparison + plots
├── configs/
│   └── config.yaml            # ALL hyperparameters (single source of truth)
├── data/                      # MNIST raw files auto-downloaded here (gitignored)
├── src/                       # Core library code (imported, never run directly)
│   ├── data.py                 # Methodology I — Split-MNIST construction
│   ├── models.py                # MLP architecture + hidden-layer hook
│   ├── tda_utils.py             # Methodology II — Maxmin sampling, persistence
│   │                             #   diagrams, Bottleneck Distance (W_inf)
│   ├── losses.py                 # TMP Loss Function (L_total) + documented
│   │                             #   differentiable surrogate
│   ├── ewc.py                    # Elastic Weight Consolidation baseline
│   ├── train.py                   # Task-1 pretrain + Task-2 continual loop,
│   │                              #   4 primary metrics, JSONL logger
│   ├── evaluate.py                 # Data Analysis — comparison table +
│   │                               #   Wilcoxon significance test (H0/H1)
│   └── visualize.py                 # matplotlib line graphs + heatmaps
├── scripts/
│   ├── _common.py               # shared bootstrap (config, seed, Task-1 model)
│   ├── run_finetune.py           # Baseline 1: naive finetuning
│   ├── run_ewc.py                # Baseline 2: Elastic Weight Consolidation
│   └── run_tmp.py                # Proposed method: Topological Manifold
│                                  #   Preservation
└── outputs/                    # ALL generated artifacts (gitignored contents)
    ├── logs/                    # JSONL training logs + comparison_results.json
    ├── models/                  # task1_base_model.pt, {method}_final_model.pt
    ├── diagrams/                # base_point_cloud.npy, diagram_base.npy
    └── plots/                   # retention_accuracy.png, summary_heatmap.png, ...
```

### File purpose quick-reference

| File | Research-plan section it implements |
|---|---|
| `src/data.py` | Methodology I — Gathering of Data (Split-MNIST) |
| `src/models.py` | Engineering Goals — network + "specific hidden layer" mapping |
| `src/tda_utils.py` | Mathematical Formulation §II — filtration, persistence diagrams, `W_inf` |
| `src/losses.py` | Mathematical Formulation §II — `L_total` (TMP Loss Function) |
| `src/ewc.py` | Objective 4 — EWC baseline |
| `src/train.py` | Simulation — full training loop, 4 primary metrics |
| `src/evaluate.py` | Data Analysis — comparative statistics, H0/H1 testing |
| `src/visualize.py` | Simulation — "comparative line graphs and heatmaps" |

---

## 3. Complete Code Implementation

All code lives in the files listed above (fully implemented, no
placeholders). Key implementation notes:

- **`src/models.py`** defines `MLPClassifier` (784→256→128→10) and
  registers a forward hook on the 128-d `hidden2` layer — this is the
  "specific hidden layer" whose activations form the geometric point
  cloud referenced throughout the Mathematical Formulation.
- **`src/tda_utils.py`** implements Maxmin (farthest-point) sampling to
  a uniform 500-point cloud, builds a Vietoris-Rips filtration via
  `giotto-tda`'s `VietorisRipsPersistence` (H0 + H1), and computes the
  true Bottleneck Distance `W_inf(D_base, D_current)` via
  `PairwiseDistance(metric="bottleneck")`.
- **`src/losses.py`** implements `L_total = L_CE + λ·W_inf` as written
  in the plan. **Important implementation note:** the true `W_inf` from
  giotto-tda is a combinatorial optimal-matching result and is not
  auto-differentiable in PyTorch. This implementation therefore:
  1. backpropagates a differentiable pairwise-distance-preservation
     surrogate every training step (principled relaxation — Vietoris-Rips
     filtrations are themselves built from pairwise distances, so
     preserving pairwise geometry preserves persistent-homology
     structure), and
  2. computes the **true** `W_inf` once per epoch via giotto-tda (matching
     the plan's own "at the end of every training epoch..." loop) to (a)
     log the authoritative Feature Space Drift metric and (b) adaptively
     rescale λ for the next epoch.

  This is documented in detail in the `losses.py` module docstring —
  read it before modifying the loss.

---

## 4. Sequential Step-by-Step Execution Guide

Run every command from the project root with your environment activated.

### Step 0 — One-shot pipeline (fastest path)

```bash
python run_all.py --config configs/baseline/split_mnist_2task.yaml
```
This runs Steps 2–5 below in order and is equivalent to running each
script individually. Use the individual scripts (Steps 2–5) if you want
to inspect intermediate outputs or modify one method at a time.

---

### Step 1 — Task-1 Baseline Pretraining (implicit, cached automatically)

The first script you run will automatically:
1. Download MNIST into `data/` (Methodology I).
2. Build the Split-MNIST partitions (Task 1 = digits 0–4, Task 2 = digits 5–9).
3. Train the MLP on Task 1 until **≥ 95% test accuracy** (Simulation
   section, "baseline mastery threshold").
4. Save the model to `outputs/models/task1_base_model.pt` and log to
   `outputs/logs/task1_pretrain.jsonl`.

This step is **cached**: subsequent scripts detect the saved checkpoint
and skip retraining, guaranteeing Finetune, EWC, and TMP all start from
an *identical* Task-1 state (fair comparison).

**Troubleshooting:**
- If accuracy plateaus below 95% before `task1_max_epochs` (30) is
  reached, the loop still stops at the epoch cap and proceeds with the
  best model reached — increase `task1_max_epochs` in `configs/baseline/split_mnist_2task.yaml`
  if you need a stricter guarantee.
- Delete `outputs/models/task1_base_model.pt` to force retraining
  (e.g., after changing `hidden1_dim`/`hidden2_dim`).

---

### Step 2 — Baseline 1: Finetune (no memory protection)

```bash
python scripts/run_finetune.py --config configs/baseline/split_mnist_2task.yaml
```
**What happens:** loads the cached Task-1 model, trains 10 epochs on
Task 2 with plain cross-entropy only (no regularization).

**Expected outputs:**
- `outputs/logs/finetune.jsonl` — per-step and per-epoch records
  (`retention_accuracy`, `learning_accuracy`, `epoch_time_sec`, `memory_mb`)
- `outputs/models/finetune_final_model.pt`

**Troubleshooting:** retention accuracy should visibly collapse toward
chance level (~20% within the 5-class task) across epochs — this is the
*expected* catastrophic-forgetting signature this baseline is meant to
demonstrate, not a bug.

---

### Step 3 — Baseline 2: Elastic Weight Consolidation (EWC)

```bash
python scripts/run_ewc.py --config configs/baseline/split_mnist_2task.yaml
```
**What happens:** computes the diagonal Fisher Information Matrix on
500 Task-1 samples, then trains Task 2 with the EWC quadratic penalty
(`λ = 400` by default, in `configs/baseline/split_mnist_2task.yaml`).

**Expected outputs:**
- `outputs/logs/ewc.jsonl`
- `outputs/models/ewc_final_model.pt`

**Troubleshooting:**
- If retention accuracy is *worse* than Finetune, `ewc.lambda_` is
  likely too small — try 1000–5000.
- If Task-2 learning accuracy stalls near chance, `ewc.lambda_` is too
  large (over-constraining the weights) — try 50–200.

---

### Step 4 — Proposed Method: Topological Manifold Preservation (TMP)

```bash
python scripts/run_tmp.py --config configs/baseline/split_mnist_2task.yaml
```
**What happens (maps directly to Mathematical Formulation §II and Simulation):**
1. Extracts hidden2 activations over Task-1 data at peak mastery.
2. Maxmin-downsamples to a 500-point cloud → builds `D_base` via
   Vietoris-Rips filtration (H0, H1).
3. Saves `D_base` artifacts to `outputs/diagrams/`.
4. Trains 10 epochs on Task 2 under `L_total = L_CE + λ·L_topo_surrogate`.
5. At the end of every epoch: rebuilds the activation point cloud,
   computes `D_current`, and calculates the true
   `W_inf(D_base, D_current)` — this is your **Feature Space Drift**
   time series, directly answering Research Question 2.

**Expected outputs:**
- `outputs/diagrams/base_point_cloud.npy`, `outputs/diagrams/diagram_base.npy`
- `outputs/logs/tmp.jsonl` (includes `feature_space_drift_w_inf` per epoch)
- `outputs/models/tmp_final_model.pt`

**Troubleshooting:**
- `ValueError` about mismatched diagram shapes inside
  `PairwiseDistance` → already handled by `tda_utils._stack_diagrams`,
  which pads diagrams **per homology dimension** with trivial
  zero-persistence points; if you modify this function, preserve that
  per-dimension padding behavior.
- Bottleneck computation slow on large point clouds → lower
  `tda.point_cloud_size` in the config (e.g., 200) for faster iteration
  during development, then restore 500 for final results.
- If `feature_space_drift_w_inf` grows without bound, lower
  `tmp.lambda_` (over-aggressive adaptive rescaling) or cap the
  rescaling in `src/train.py`'s `train_task2`.

---

### Step 5 — Comparative Statistical Analysis & Plots (Data Analysis section)

```bash
python compare_results.py --config configs/baseline/split_mnist_2task.yaml
```
**What happens:**
1. Loads all three JSONL logs and builds a tidy long-format table.
2. Computes mean/std per method for the four primary metrics.
3. Runs a paired Wilcoxon signed-rank test on `retention_accuracy`
   between TMP vs. Finetune and TMP vs. EWC — directly tests **H0 vs.
   H1**.
4. Generates six plots into `outputs/plots/`:
   `retention_accuracy.png`, `learning_accuracy.png`, `epoch_time.png`,
   `memory_usage.png`, `feature_space_drift.png`, `summary_heatmap.png`.

**Expected outputs:**
- `outputs/logs/comparison_results.json` (summary stats + verdicts,
  printed to console as well)
- Six `.png` files in `outputs/plots/`

**Troubleshooting:**
- `FileNotFoundError: Missing log for method 'X'` → you skipped Step
  2/3/4 for that method; run it first.
- Wilcoxon test returns `"Insufficient or identical paired samples"` →
  you likely ran with `task2_epochs < 2`; use at least 2 epochs (10 is
  the plan's default) for a meaningful paired test.

---

## 5. Methodology Alignment Summary

| Research Plan Element | Implementation |
|---|---|
| **Rationale / Objective 1** — design TMP using Bottleneck Distance | `src/tda_utils.bottleneck_distance`, `src/losses.tmp_total_loss` |
| **Objective 2** — implement & simulate in Python | Entire `src/` package + `scripts/run_tmp.py` |
| **Objective 3** — analyze graphs/numerical data | `src/evaluate.py`, `src/visualize.py` |
| **Objective 4** — compare vs. EWC and Finetune | `scripts/run_ewc.py`, `scripts/run_finetune.py`, `compare_results.py` |
| **RQ1** — How does TMP relate to catastrophic forgetting? | `retention_accuracy` time series, all three methods, Step 5 plots |
| **RQ2** — significant difference in feature space drift vs. standard regularization | `feature_space_drift_w_inf` (TMP) vs. EWC's Fisher-penalty trajectory; extend `train_task2` to log `W_inf` for EWC/Finetune too if a direct drift-vs-drift test is required |
| **RQ2.1–2.3** — retention accuracy / learning rate / computational overhead | `evaluate_accuracy` (retention & learning), `epoch_time_sec` + `memory_mb` (overhead) |
| **H0 / H1** | `src/evaluate.significance_test` (Wilcoxon signed-rank, α = 0.05) |
| **Methodology I** — Split-MNIST, PyTorch, giotto-tda | `src/data.py`, `src/models.py`, `src/tda_utils.py` |
| **Methodology II** — Vietoris-Rips, H0/H1, `D_base`/`D_current`, `W_inf`, `L_total` | `src/tda_utils.py`, `src/losses.py` |
| **Simulation** — 95% mastery threshold, 500-pt Maxmin cloud, 10 epochs, 4 metrics, comparative graphs/heatmaps | `src/train.py`, `src/visualize.py` |
| **Limitations** — restricted to H0/H1 persistent homology | `tda.homology_dims: [0, 1]` in `configs/baseline/split_mnist_2task.yaml` |

---

## 6. Reproducibility Checklist

- [ ] `pip install -r requirements.txt` (or conda env) completed without errors
- [ ] `configs/baseline/split_mnist_2task.yaml` reviewed (seed = 42 by default for exact reproducibility)
- [ ] Step 1 (implicit) produced `task1_base_model.pt` with logged accuracy ≥ 0.95
- [ ] Steps 2, 3, 4 each produced a `.jsonl` log and a `_final_model.pt`
- [ ] Step 5 produced `comparison_results.json` and 6 plots
- [ ] Wilcoxon verdicts recorded for the final report / paper write-up

---

## 7. Extending This Implementation

- **More tasks within Split-MNIST (5-task):** already implemented — see
  `docs/MULTITASK.md` for a generalized N-task pipeline
  (`run_multitask.py`, `src/train_general.py`, `src/datasets_extended.py`)
  that splits the same MNIST digits into 5 sequential tasks
  (`[0,1] → [2,3] → [4,5] → [6,7] → [8,9]`) instead of 2, to test
  whether TMP's protection holds up over a longer sequence, without
  touching the original validated 2-task code.

- **Different architecture:** swap `MLPClassifier` in `src/models.py`
  for a CNN; keep the `_capture_hidden2` hook pattern on whichever
  layer you want to topologically monitor.
- **Exact (non-adaptive) λ:** set `tmp.lambda_` fixed and remove the
  rescaling line in `src/train.py` if your study design calls for a
  constant-λ ablation.
