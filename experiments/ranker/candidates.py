"""Fold-specific candidate generation with finalized production semantics."""

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from app.domain.candidates import CandidateGenerationResult, RecommendationCandidate
from app.ml.candidate_retrieval import retrieve_exact_candidates
from app.ml.svd_profiles import build_svd_profile
from experiments.ranker.artifacts import FoldArtifacts
from experiments.ranker.config import POPULARITY_DEPTH, SVD_DEPTH


def generate_fold_candidates(
    user_id: int,
    context_item_rows: NDArray[np.int64],
    context_rating_buckets: NDArray[np.int64],
    artifacts: FoldArtifacts,
) -> CandidateGenerationResult:
    """Generate exact fold candidates without target injection or refill."""
    excluded_ids = {int(artifacts.film_ids[row]) for row in context_item_rows}
    query = build_svd_profile(
        artifacts.item_vectors,
        context_item_rows,
        context_rating_buckets,
        "svd_positive_weighted",
    )
    svd_candidates = (
        retrieve_exact_candidates(
            artifacts.retrieval_index,
            query,
            excluded_film_ids=excluded_ids,
            depth=SVD_DEPTH,
        )
        if query is not None
        else ()
    )
    popularity_rows = []
    for raw_row in artifacts.popularity_order_rows:
        row = int(raw_row)
        film_id = int(artifacts.film_ids[row])
        if film_id in excluded_ids:
            continue
        popularity_rows.append(row)
        if len(popularity_rows) == POPULARITY_DEPTH:
            break

    ordered_ids: list[int] = []
    merged: dict[int, RecommendationCandidate] = {}
    for rank, (film_id, score) in enumerate(svd_candidates, start=1):
        ordered_ids.append(film_id)
        merged[film_id] = RecommendationCandidate(
            film_id=film_id,
            svd_score=score,
            svd_rank=rank,
            retrieved_by_svd=True,
        )
    for popularity_rank, row in enumerate(popularity_rows, start=1):
        film_id = int(artifacts.film_ids[row])
        candidate = merged.get(film_id)
        if candidate is None:
            ordered_ids.append(film_id)
            candidate = RecommendationCandidate(film_id=film_id)
        merged[film_id] = replace(
            candidate,
            popularity_score=int(artifacts.popularity_counts[row]),
            popularity_rank=popularity_rank,
            retrieved_by_popularity=True,
        )
    candidates = tuple(merged[film_id] for film_id in ordered_ids)
    if excluded_ids.intersection(candidate.film_id for candidate in candidates):
        raise RuntimeError("fold candidate exclusion invariant was violated")
    return CandidateGenerationResult(
        user_id=user_id,
        candidates=candidates,
        nominal_budget=SVD_DEPTH + POPULARITY_DEPTH,
        svd_depth=SVD_DEPTH,
        popularity_depth=POPULARITY_DEPTH,
        svd_profile_available=query is not None,
    )


def target_source(candidate: RecommendationCandidate | None) -> str:
    """Classify the retrieved source membership of one target."""
    if candidate is None:
        return "missed"
    if candidate.retrieved_by_svd and candidate.retrieved_by_popularity:
        return "both"
    if candidate.retrieved_by_svd:
        return "svd_only"
    if candidate.retrieved_by_popularity:
        return "popularity_only"
    raise ValueError("candidate has no retrieval source")
