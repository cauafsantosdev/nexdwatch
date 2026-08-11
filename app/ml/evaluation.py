"""Shared leakage-free evaluation data primitives."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from app.ml.historical_interactions import UserSplit


def build_evaluation_svd_training_matrix(
    splits: tuple[UserSplit, ...], item_count: int
) -> csr_matrix:
    """Build SVD input solely from each user's heldout-free context ratings."""
    user_rows: list[NDArray[np.int64]] = []
    item_rows: list[NDArray[np.int64]] = []
    ratings: list[NDArray[np.float32]] = []
    for user_position, user in enumerate(splits):
        count = len(user.context_item_rows)
        if not count:
            continue
        user_rows.append(np.full(count, user_position, dtype=np.int64))
        item_rows.append(user.context_item_rows)
        ratings.append((user.context_rating_buckets + 1).astype(np.float32) / 2.0)
    if not user_rows:
        raise ValueError("evaluation splits contain no SVD training interactions")
    return csr_matrix(
        (
            np.concatenate(ratings),
            (np.concatenate(user_rows), np.concatenate(item_rows)),
        ),
        shape=(len(splits), item_count),
        dtype=np.float32,
    )


def training_positive_counts(
    splits: tuple[UserSplit, ...], item_count: int
) -> NDArray[np.int64]:
    """Count positive interactions retained in the evaluation training split."""
    counts = np.zeros(item_count, dtype=np.int64)
    for user in splits:
        np.add.at(counts, user.training_positive_rows, 1)
    return counts


def popularity_order_rows(
    counts: NDArray[np.int64], film_ids: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Order item rows by descending count and ascending actual film ID."""
    if counts.shape != film_ids.shape:
        raise ValueError("popularity counts and film IDs must have matching shapes")
    return np.ascontiguousarray(np.lexsort((film_ids, -counts)), dtype=np.int64)
