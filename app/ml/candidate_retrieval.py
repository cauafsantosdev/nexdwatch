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
    """Retrieve an exact ordered list while excluding every known film."""
    if depth < 0:
        raise ValueError("candidate depth must be non-negative")
    if depth == 0 or index.ntotal == 0:
        return ()
    excluded = set(excluded_film_ids)
    requested = min(int(index.ntotal), depth + len(excluded))
    scores, labels = index.search(
        np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32),
        requested,
    )
    candidates: list[tuple[int, float]] = []
    for raw_id, raw_score in zip(labels[0], scores[0], strict=True):
        film_id = int(raw_id)
        if film_id < 0 or film_id in excluded:
            continue
        candidates.append((film_id, float(raw_score)))
        if len(candidates) == depth:
            break
    return tuple(candidates)
