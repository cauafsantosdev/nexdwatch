"""Exact finalized reciprocal-rank fusion for internal candidate inventories."""

import math

from app.domain.candidates import RecommendationCandidate
from app.domain.categorized_recommendations import RankedCandidate
from app.policy.config import DEFAULT_POLICY_CONFIG, CategoryPolicyConfig


def rank_candidates_by_rrf(
    candidates: tuple[RecommendationCandidate, ...],
    popularity_rank_by_film: dict[int, int],
    popularity_film_count: int,
    *,
    config: CategoryPolicyConfig = DEFAULT_POLICY_CONFIG,
) -> tuple[RankedCandidate, ...]:
    """Apply equal-weight k=60 RRF with missing-source zero and film-ID ties."""
    if popularity_film_count <= 0:
        raise ValueError("popularity film count must be positive")
    if len({candidate.film_id for candidate in candidates}) != len(candidates):
        raise ValueError("RRF candidates must have unique film IDs")

    scored = [
        (
            candidate,
            _rrf_score(candidate, config.rrf_k),
            _popularity_stratum(
                popularity_rank_by_film.get(candidate.film_id, popularity_film_count),
                popularity_film_count,
            ),
        )
        for candidate in candidates
    ]
    scored.sort(key=lambda value: (-value[1], value[0].film_id))
    return tuple(
        RankedCandidate.from_candidate(
            candidate,
            rrf_score=score,
            rrf_rank=rank,
            popularity_stratum=stratum,
        )
        for rank, (candidate, score, stratum) in enumerate(scored, start=1)
    )


def _rrf_score(candidate: RecommendationCandidate, k: int) -> float:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    return (
        1.0 / (k + candidate.svd_rank) if candidate.svd_rank is not None else 0.0
    ) + (
        1.0 / (k + candidate.popularity_rank)
        if candidate.popularity_rank is not None
        else 0.0
    )


def _popularity_stratum(rank: int, film_count: int) -> str:
    head_stop = math.ceil(film_count * 0.10)
    mid_stop = math.ceil(film_count * 0.50)
    if rank <= head_stop:
        return "HEAD"
    if rank <= mid_stop:
        return "MID"
    return "TAIL"
