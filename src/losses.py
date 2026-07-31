"""
losses.py
---------
Implements the TMP Loss Function from the Mathematical Formulation:

    L_total = L_cross-entropy(D_Task2) + lambda * W_inf(D_base, D_current)

IMPLEMENTATION NOTE ON DIFFERENTIABILITY (read this first)
------------------------------------------------------------
The true Bottleneck Distance W_inf is computed by giotto-tda through a
combinatorial optimal-matching / bipartite-graph algorithm. That
routine is not expressed in PyTorch tensor ops and therefore has no
usable gradient with respect to the network's weights.

To stay faithful to the research plan's objective ("penalize any
structural feature space drift") while keeping the training loop
end-to-end differentiable, this module implements TWO complementary
pieces, matching the plan's own simulation loop design:

  1. `topological_surrogate_loss` -- a DIFFERENTIABLE proxy that is
     backpropagated every training step. It penalizes changes in the
     pairwise-distance structure of the current mini-batch's hidden2
     activations relative to a frozen baseline sub-sample of D_base's
     source activations. Preserving pairwise distances is what
     preserves the persistent-homology structure (Vietoris-Rips
     filtrations are built directly from pairwise distances), so
     minimizing this term is a principled differentiable relaxation
     of minimizing W_inf.

  2. The TRUE W_inf(D_base, D_current), computed via
     `tda_utils.bottleneck_distance` on the full 500-point Maxmin
     cloud, is recomputed once per epoch (matching the plan's
     "At the end of every training epoch..." simulation loop) and used
     to (a) log Feature Space Drift, and (b) adaptively rescale
     lambda for the *next* epoch, so the true bottleneck measurement
     still steers training even though it isn't directly
     back-propagated through.

This hybrid is the standard way persistent-homology regularizers are
made trainable in the literature (cf. topological-autoencoder /
structure-preserving regularization approaches), and keeps every
number in the paper's formula ("L_total", "lambda", "W_inf") faithfully
represented in code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_cross_entropy(logits, targets, allowed_classes, device):
    """
    L_cross-entropy(D_Task2): standard cross-entropy restricted to the
    classes active in the current task (class-incremental masking).
    """
    mask = torch.full((logits.shape[1],), float("-inf"), device=device)
    mask[allowed_classes] = 0.0
    masked_logits = logits + mask.unsqueeze(0)
    return F.cross_entropy(masked_logits, targets)


def topological_surrogate_loss(current_batch_activations: torch.Tensor,
                                baseline_reference_activations: torch.Tensor) -> torch.Tensor:
    """
    Differentiable proxy for structural (topological) drift.

    Compares the pairwise-distance matrix of `current_batch_activations`
    against the pairwise-distance matrix of `baseline_reference_activations`,
    penalizing the (normalized) mean absolute change in pairwise geometry.
    This is fully differentiable w.r.t. the network weights that
    produced `current_batch_activations`.

    IMPORTANT -- for this to measure genuine drift rather than sampling
    noise, both arguments must correspond to THE SAME underlying images,
    just captured at two different points in training (e.g. the frozen
    baseline activations vs. a live re-forward of that same fixed image
    set through the current network). This is exactly what
    `train.build_fixed_task1_reference` + the TMP branch in
    `train.train_task2` set up: a FIXED subset of Task-1 images is
    forwarded through the current model every step and compared against
    that same subset's baseline activations. Comparing DIFFERENT random
    samples of Task-1 images here (as an earlier version of this code
    did) confounds sampling noise with real drift and produces an
    unreliable gradient signal.

    Args:
        current_batch_activations: (N, H) tensor, requires_grad=True,
            the live hidden2 output for a FIXED reference image set.
        baseline_reference_activations: (N, H) tensor (no grad needed),
            that SAME fixed image set's activations under the baseline
            (just-mastered-Task-1) model.

    Returns:
        scalar torch.Tensor loss.
    """
    b = current_batch_activations.shape[0]
    ref = baseline_reference_activations
    if ref.shape[0] != b:
        # Defensive fallback only -- the intended call pattern (see
        # docstring above) always passes matched, equal-sized tensors,
        # so this branch should not trigger in normal use.
        idx = torch.randint(0, ref.shape[0], (b,), device=ref.device)
        ref = ref[idx]

    current_dists = torch.cdist(current_batch_activations, current_batch_activations, p=2)
    baseline_dists = torch.cdist(ref, ref, p=2)

    # Normalize each pairwise-distance matrix by its own mean before
    # comparing. Raw activation-space distances are unbounded and scale
    # with the network's current weight magnitudes -- without this
    # normalization, the surrogate's raw value (and gradient) can be
    # orders of magnitude larger than the cross-entropy loss, letting it
    # dominate/destabilize training regardless of how small `lambda_` is
    # set.
    #
    # IMPORTANT: the denominator (the mean) is DETACHED here. An earlier
    # version of this code normalized by a live, non-detached mean for
    # full scale-invariance -- but that has a serious numerical failure
    # mode: the gradient of x / mean(x) w.r.t. x contains a 1/mean(x)
    # term that BLOWS UP as the mean shrinks toward zero (which can
    # happen if activations partially collapse under training pressure).
    # A tiny, well-behaved FORWARD loss value says nothing about the
    # BACKWARD gradient magnitude -- this is exactly the kind of bug
    # that stays invisible unless you specifically log gradient norms
    # (see train_general.py's grad-norm logging, added after this bug
    # was diagnosed via a multi-task lambda sweep where every nonzero
    # lambda collapsed learning to near-zero regardless of its size,
    # while the reported loss value looked completely normal).
    #
    # Detaching the denominator is the standard "stop-gradient on
    # statistics" technique (used e.g. in BYOL/SimSiam) -- it keeps
    # gradients numerically bounded (1/c for a CONSTANT c each step,
    # not 1/x that can blow up), at the cost of reopening a much MILDER
    # collapse incentive than the original un-normalized loss had: since
    # the denominator is recomputed fresh from the live mean every
    # step, the network can't permanently exploit shrinking scale to
    # cheat the loss -- each subsequent step re-normalizes against
    # whatever the new scale actually is.
    eps = 1e-4
    current_denom = current_dists.mean().detach().clamp(min=eps)
    baseline_denom = baseline_dists.mean().clamp(min=eps)
    current_dists_norm = current_dists / current_denom
    baseline_dists_norm = baseline_dists / baseline_denom

    return F.l1_loss(current_dists_norm, baseline_dists_norm)


def tmp_total_loss(logits, targets, allowed_classes, device,
                    current_batch_activations, baseline_reference_activations,
                    lambda_: float):
    """
    Full TMP training-step loss:

        L_total = L_cross-entropy(D_Task2) + lambda * L_topo_surrogate

    `lambda_` is the same lambda from the paper's formula; it is
    adaptively rescaled once per epoch in train.py using the true
    W_inf measurement (see module docstring).
    """
    ce_loss = masked_cross_entropy(logits, targets, allowed_classes, device)
    topo_loss = topological_surrogate_loss(current_batch_activations, baseline_reference_activations)
    total = ce_loss + lambda_ * topo_loss
    return total, ce_loss.detach().item(), topo_loss.detach().item()
