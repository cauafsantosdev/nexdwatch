"""Exact retrieval primitives shared by candidate analysis and application code."""

from collections.abc import Collection

import faiss
import numpy as np
from numpy.typing import NDArray


def retrieve_exact_candidates(
    index: faiss.IndexIDMap2,
    query: NDArray[np.float32],
    *,
    excluded_film_ids: Collection[int],
    depth: int,
) -> tuple[tuple[int, float], ...]:
    """Retrieve exact inner-product neighbors while excluding every watched film.

    The FAISS index stores actual film IDs through ``IndexIDMap2``. Extra neighbors
    are requested only to compensate for exclusions; the returned depth remains
    bounded and deterministic.

    Args:
        index: Exact inner-product index whose labels are persisted ``Film.id`` values.
        query: One model-space user vector; it is reshaped and made float32-contiguous.
        excluded_film_ids: Complete watched identity set to remove before depth is met.
        depth: Maximum number of unwatched neighbors to return.

    Returns:
        tuple[tuple[int, float], ...]: Film ID and exact inner-product score in FAISS
            order, bounded by ``depth`` and index size.

    Raises:
        ValueError: If ``depth`` is negative.
    """
    if depth < 0:
        raise ValueError("candidate depth must be non-negative")
    if depth == 0 or index.ntotal == 0:
        return ()
    # Over-retrieve only enough indexed positions to compensate for every possible
    # exclusion; the index size remains the absolute upper bound.
    excluded = set(excluded_film_ids)
    requested = min(int(index.ntotal), depth + len(excluded))
    scores, labels = index.search(
        np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32),
        requested,
    )
    # FAISS may emit -1 sentinels for unavailable neighbors. Exclude them together
    # with watched films before enforcing the requested final depth.
    candidates: list[tuple[int, float]] = []
    for raw_id, raw_score in zip(labels[0], scores[0], strict=True):
        film_id = int(raw_id)
        if film_id < 0 or film_id in excluded:
            continue
        candidates.append((film_id, float(raw_score)))
        if len(candidates) == depth:
            break
    return tuple(candidates)
