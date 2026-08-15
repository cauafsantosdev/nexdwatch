"""Fold-specific train-user-only SVD, popularity, and film aggregates."""

from dataclasses import dataclass

import faiss
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from app.ml.faiss_index import create_faiss_index
from app.ml.ratings import rating_to_bucket
from experiments.ranker.config import (
    NEGATIVE_RATING_THRESHOLD,
    POSITIVE_RATING_THRESHOLD,
)
from experiments.retrieval.candidate_analysis import assign_popularity_strata


@dataclass(frozen=True, slots=True)
class RankerUserContext:
    """One user's interactions visible to a fold-specific artifact builder."""

    user_id: int
    item_rows: NDArray[np.int64]
    rating_buckets: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FoldArtifacts:
    """All global ranking inputs fit exclusively from ranker-training users."""

    film_ids: NDArray[np.int64]
    item_vectors: NDArray[np.float32]
    retrieval_index: faiss.IndexIDMap2
    popularity_counts: NDArray[np.int64]
    popularity_order_rows: NDArray[np.int64]
    popularity_global_ranks: NDArray[np.int64]
    popularity_strata: NDArray[np.str_]
    popularity_percentiles: NDArray[np.float64]
    rating_counts: NDArray[np.int64]
    positive_counts: NDArray[np.int64]
    negative_counts: NDArray[np.int64]
    rating_means: NDArray[np.float64]
    rating_variances: NDArray[np.float64]
    smoothed_ratings: NDArray[np.float64]
    global_training_mean: float
    contributing_user_ids: frozenset[int]
    contributing_interaction_count: int


def build_fold_artifacts(
    contexts: tuple[RankerUserContext, ...],
    film_ids: NDArray[np.int64],
    *,
    seed: int,
    svd_components: int = 32,
) -> FoldArtifacts:
    """Fit all global artifacts from the supplied training contexts only."""
    if not contexts:
        raise ValueError("fold artifacts require training-user contexts")
    item_count = len(film_ids)
    user_rows: list[NDArray[np.int64]] = []
    item_rows: list[NDArray[np.int64]] = []
    ratings: list[NDArray[np.float32]] = []
    for position, context in enumerate(contexts):
        if len(context.item_rows) != len(context.rating_buckets):
            raise ValueError("fold context rows and ratings differ")
        if not len(context.item_rows):
            continue
        user_rows.append(np.full(len(context.item_rows), position, dtype=np.int64))
        item_rows.append(context.item_rows)
        ratings.append((context.rating_buckets + 1).astype(np.float32) / 2.0)
    if not ratings:
        raise ValueError("fold training contexts contain no ratings")
    concatenated_items = np.concatenate(item_rows)
    concatenated_ratings = np.concatenate(ratings)
    matrix = csr_matrix(
        (
            concatenated_ratings,
            (np.concatenate(user_rows), concatenated_items),
        ),
        shape=(len(contexts), item_count),
        dtype=np.float32,
    )
    component_count = min(svd_components, max(1, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=component_count, random_state=seed)
    svd.fit(matrix)
    item_vectors = np.ascontiguousarray(
        normalize(svd.components_.T, axis=1).astype(np.float32), dtype=np.float32
    )
    retrieval_index = create_faiss_index(item_vectors, film_ids)

    rating_counts = np.bincount(concatenated_items, minlength=item_count).astype(
        np.int64
    )
    rating_sums = np.bincount(
        concatenated_items, weights=concatenated_ratings, minlength=item_count
    )
    squared_sums = np.bincount(
        concatenated_items,
        weights=np.square(concatenated_ratings),
        minlength=item_count,
    )
    positive = concatenated_ratings >= POSITIVE_RATING_THRESHOLD
    negative = concatenated_ratings <= NEGATIVE_RATING_THRESHOLD
    positive_counts = np.bincount(
        concatenated_items[positive], minlength=item_count
    ).astype(np.int64)
    negative_counts = np.bincount(
        concatenated_items[negative], minlength=item_count
    ).astype(np.int64)
    rating_means = np.full(item_count, np.nan, dtype=np.float64)
    rating_variances = np.full(item_count, np.nan, dtype=np.float64)
    populated = rating_counts > 0
    rating_means[populated] = rating_sums[populated] / rating_counts[populated]
    rating_variances[populated] = np.maximum(
        squared_sums[populated] / rating_counts[populated]
        - np.square(rating_means[populated]),
        0.0,
    )
    global_mean = float(concatenated_ratings.mean())
    smoothed = (rating_sums + 20.0 * global_mean) / (rating_counts + 20.0)
    popularity_order = np.ascontiguousarray(
        np.lexsort((film_ids, -positive_counts)), dtype=np.int64
    )
    popularity_ranks = np.empty(item_count, dtype=np.int64)
    popularity_ranks[popularity_order] = np.arange(1, item_count + 1)
    strata, percentiles = assign_popularity_strata(positive_counts, film_ids)
    return FoldArtifacts(
        film_ids=np.ascontiguousarray(film_ids, dtype=np.int64),
        item_vectors=item_vectors,
        retrieval_index=retrieval_index,
        popularity_counts=positive_counts,
        popularity_order_rows=popularity_order,
        popularity_global_ranks=popularity_ranks,
        popularity_strata=strata,
        popularity_percentiles=percentiles,
        rating_counts=rating_counts,
        positive_counts=positive_counts,
        negative_counts=negative_counts,
        rating_means=rating_means,
        rating_variances=rating_variances,
        smoothed_ratings=smoothed,
        global_training_mean=global_mean,
        contributing_user_ids=frozenset(context.user_id for context in contexts),
        contributing_interaction_count=len(concatenated_items),
    )


def positive_rating_bucket() -> int:
    """Return the frozen bucket boundary for positive ranker interactions."""
    return rating_to_bucket(POSITIVE_RATING_THRESHOLD)
