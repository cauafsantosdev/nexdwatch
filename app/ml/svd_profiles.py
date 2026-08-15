"""Shared SVD user-profile construction for offline and live retrieval."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

SVDProfile = Literal[
    "svd_mean",
    "svd_positive_mean",
    "svd_rating_centered",
    "svd_user_centered",
    "svd_positive_weighted",
]


def build_svd_profile(
    item_vectors: NDArray[np.floating],
    item_rows: NDArray[np.int64],
    rating_buckets: NDArray[np.int64],
    strategy: SVDProfile,
) -> NDArray[np.float32] | None:
    """Construct one SVD query under explicit, reproducible rating semantics.

    ``svd_positive_weighted`` uses ``max(rating - 3.0, 0)`` and returns ``None``
    when no positive weight exists. It never falls back to mean SVD.

    Args:
        item_vectors: Model item matrix indexed by ``item_rows``.
        item_rows: One-dimensional vector rows for the user's rated films.
        rating_buckets: Matching zero-based Letterboxd half-star buckets.
        strategy: Explicit aggregation semantics used by offline or live retrieval.

    Returns:
        A contiguous float32 query vector, or ``None`` when the selected strategy
        has no usable evidence or produces a non-finite/zero vector.

    Raises:
        ValueError: If input arrays have incompatible shape or ``strategy`` is unknown.
    """
    if item_rows.ndim != 1 or rating_buckets.ndim != 1:
        raise ValueError("profile rows and ratings must be one-dimensional")
    if len(item_rows) != len(rating_buckets):
        raise ValueError("profile rows and ratings must have matching lengths")
    if not len(item_rows):
        return None
    # Convert buckets back to the exact 0.5–5.0 rating scale before applying the
    # selected, explicitly named weighting semantics.
    vectors = item_vectors[item_rows]
    ratings = (rating_buckets.astype(np.float32) + 1.0) / 2.0
    if strategy == "svd_mean":
        query = vectors.mean(axis=0)
    elif strategy == "svd_positive_mean":
        selected = ratings >= 3.5
        if not selected.any():
            return None
        query = vectors[selected].mean(axis=0)
    else:
        if strategy == "svd_rating_centered":
            weights = ratings - 3.0
        elif strategy == "svd_user_centered":
            weights = ratings - ratings.mean()
        elif strategy == "svd_positive_weighted":
            weights = np.maximum(ratings - 3.0, 0.0)
        else:
            raise ValueError(f"unknown SVD profile strategy: {strategy}")
        denominator = float(np.abs(weights).sum())
        if denominator <= 1e-12:
            return None
        query = (weights @ vectors) / denominator
    # FAISS requires a finite contiguous float32 query; degenerate profiles are
    # represented as unavailable rather than silently falling back to another model.
    query = np.ascontiguousarray(query, dtype=np.float32)
    if not np.isfinite(query).all() or np.linalg.norm(query) <= 1e-12:
        return None
    return query
