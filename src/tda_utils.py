"""
tda_utils.py
------------
Implements Methodology Section II (Mathematical Formulation) and the
"Simulation" section's TDA pipeline:

    1. Maxmin downsampling of the raw activation cloud to a uniform
       500-point cloud.
    2. Vietoris-Rips filtration -> Persistence Diagram (H0, H1) via
       giotto-tda.
    3. Bottleneck Distance W_inf(D_base, D_current) between two
       persistence diagrams.

giotto-tda's `PairwiseDistance` computes the true combinatorial
Bottleneck / Wasserstein distance but is NOT auto-differentiable
(it wraps a C++/numba optimal-matching routine). This module is
therefore used for:
    - producing the authoritative Persistence Diagrams (D_base, D_current)
    - computing the ground-truth W_inf metric reported every epoch
      (used for evaluation, plots, and the H2/H3 hypothesis testing)

The differentiable *surrogate* term used inside the training loss is
implemented separately in `losses.py` and is explicitly documented
there as an implementation choice necessitated by this non-
differentiability (see README.md, "Implementation Note on
Differentiability").
"""

import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PairwiseDistance


def maxmin_sample(point_cloud: np.ndarray, n_samples: int, seed: int = 0) -> np.ndarray:
    """
    Greedy Maxmin (farthest-point) sampling: iteratively picks the point
    that maximizes the minimum distance to the already-selected set.
    Produces a uniform, structure-preserving downsampling of the
    high-dimensional activation cloud to `n_samples` points, as
    specified in the Simulation section ("downsampled through Maxmin
    sampling to a uniform 500-point cloud").
    """
    rng = np.random.default_rng(seed)
    n_points = point_cloud.shape[0]
    if n_points <= n_samples:
        # pad by resampling with replacement if the cloud is too small
        idx = rng.choice(n_points, size=n_samples, replace=True)
        return point_cloud[idx]

    selected_idx = [int(rng.integers(0, n_points))]
    dist_to_set = pairwise_distances(point_cloud, point_cloud[selected_idx]).flatten()

    for _ in range(n_samples - 1):
        next_idx = int(np.argmax(dist_to_set))
        selected_idx.append(next_idx)
        new_dist = pairwise_distances(point_cloud, point_cloud[[next_idx]]).flatten()
        dist_to_set = np.minimum(dist_to_set, new_dist)

    return point_cloud[selected_idx]


def compute_persistence_diagram(point_cloud: np.ndarray, homology_dims=(0, 1), max_edge_length=None):
    """
    Builds a Vietoris-Rips filtration over `point_cloud` and returns the
    Persistence Diagram D as a giotto-tda formatted array of shape
    (1, n_features, 3) with columns (birth, death, homology_dim).
    """
    max_edge = np.inf if max_edge_length is None else max_edge_length
    vr = VietorisRipsPersistence(
        homology_dimensions=list(homology_dims),
        metric="euclidean",
        max_edge_length=max_edge,
        n_jobs=1,
    )
    # giotto-tda expects shape (n_samples_batch, n_points, n_dims)
    diagrams = vr.fit_transform(point_cloud[None, :, :])
    return diagrams  # shape (1, n_features, 3)


def bottleneck_distance(diagram_base, diagram_current, delta: float = 0.01) -> float:
    """
    Computes W_inf(D_base, D_current), the Bottleneck Distance between
    two persistence diagrams, exactly as specified in the Mathematical
    Formulation:

        W_inf(D_base, D_current) = inf_gamma sup_{x in D_base} ||x - gamma(x)||_inf

    `delta` is the approximation slack accepted by giotto-tda's internal
    bottleneck solver (0 = exact but slower).
    """
    # PairwiseDistance expects a stack of diagrams padded to a common
    # feature count; giotto-tda handles this internally when given both
    # diagrams together.
    stacked = _stack_diagrams(diagram_base, diagram_current)
    pd = PairwiseDistance(metric="bottleneck", metric_params={"delta": delta}, n_jobs=1)
    dist_matrix = pd.fit_transform(stacked)
    # dist_matrix is (2, 2); the base-vs-current distance is at [0, 1]
    return float(dist_matrix[0, 1])


def _stack_diagrams(diagram_a, diagram_b):
    """
    giotto-tda's PairwiseDistance requires every diagram in the input
    collection to contain the SAME NUMBER OF POINTS *within each
    homology dimension* (not just the same total count). This pads
    diagram_a and diagram_b per-homology-dimension with trivial
    zero-persistence points (birth == death) so both diagrams line up,
    then stacks them into the (2, n_features_max, 3) array required by
    PairwiseDistance.fit_transform. Trivial points do not affect the
    bottleneck matching since they have zero persistence.
    """
    a = diagram_a[0]
    b = diagram_b[0]

    dims = np.unique(np.concatenate([a[:, 2], b[:, 2]]))

    padded_a_parts = []
    padded_b_parts = []
    for dim in dims:
        a_dim = a[a[:, 2] == dim]
        b_dim = b[b[:, 2] == dim]
        max_len = max(a_dim.shape[0], b_dim.shape[0], 1)

        def pad(d_dim):
            if d_dim.shape[0] == max_len:
                return d_dim
            pad_rows = max_len - d_dim.shape[0]
            filler = np.zeros((pad_rows, 3), dtype=a.dtype)
            filler[:, 2] = dim
            return np.vstack([d_dim, filler]) if d_dim.shape[0] > 0 else filler

        padded_a_parts.append(pad(a_dim))
        padded_b_parts.append(pad(b_dim))

    padded_a = np.vstack(padded_a_parts)
    padded_b = np.vstack(padded_b_parts)
    return np.stack([padded_a, padded_b], axis=0)


def build_activation_point_cloud(model, dataloader, device, n_points, seed=0):
    """
    End-to-end helper: extracts hidden-layer activations from `model`
    over `dataloader`, then Maxmin-downsamples to `n_points`.
    Returns a numpy array of shape (n_points, hidden_dim).
    """
    activations = model.extract_activations(dataloader, device).numpy()
    return maxmin_sample(activations, n_points, seed=seed)
