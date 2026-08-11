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
    """Construct one unnormalized SVD query under explicit rating semantics."""
    if item_rows.ndim != 1 or rating_buckets.ndim != 1:
        raise ValueError("profile rows and ratings must be one-dimensional")
    if len(item_rows) != len(rating_buckets):
        raise ValueError("profile rows and ratings must have matching lengths")
    if not len(item_rows):
        return None
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
    query = np.ascontiguousarray(query, dtype=np.float32)
    if not np.isfinite(query).all() or np.linalg.norm(query) <= 1e-12:
        return None
    return query
