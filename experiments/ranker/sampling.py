"""Deterministic 512-row LambdaRank group sampling."""

from dataclasses import dataclass

import numpy as np

from app.domain.candidates import RecommendationCandidate
from experiments.ranker.config import GROUP_ROW_CAP


@dataclass(frozen=True, slots=True)
class SampledGroup:
    """Ordered candidate rows, labels, and sampling audit strata."""

    candidates: tuple[RecommendationCandidate, ...]
    labels: np.ndarray
    sampling_strata: tuple[str, ...]
    retrieved_positive_count: int
    requested_positive_count: int


def build_full_evaluation_group(
    candidates: tuple[RecommendationCandidate, ...],
    positive_labels: dict[int, int],
    *,
    forbidden_positive_ids: set[int] | None = None,
) -> SampledGroup:
    """Keep the complete candidate inventory while excluding alternate positives."""
    forbidden = forbidden_positive_ids or set()
    if forbidden.intersection(positive_labels):
        raise ValueError("a positive cannot also be forbidden")
    retained = tuple(
        candidate for candidate in candidates if candidate.film_id not in forbidden
    )
    labels = np.asarray(
        [positive_labels.get(candidate.film_id, 0) for candidate in retained],
        dtype=np.int8,
    )
    retrieved = int(np.count_nonzero(labels))
    return SampledGroup(
        candidates=retained,
        labels=np.ascontiguousarray(labels),
        sampling_strata=tuple(
            "positive" if label else "full_pool_candidate" for label in labels
        ),
        retrieved_positive_count=retrieved,
        requested_positive_count=len(positive_labels),
    )


def sample_ranker_group(
    candidates: tuple[RecommendationCandidate, ...],
    positive_labels: dict[int, int],
    *,
    forbidden_positive_ids: set[int] | None = None,
    seed: int,
    user_id: int,
    cap: int = GROUP_ROW_CAP,
) -> SampledGroup:
    """Include positives then sample mutually deduplicated hard-negative strata."""
    if cap <= 0:
        raise ValueError("ranker group cap must be positive")
    forbidden = forbidden_positive_ids or set()
    if forbidden.intersection(positive_labels):
        raise ValueError("a positive cannot also be forbidden")
    by_id = {candidate.film_id: candidate for candidate in candidates}
    positive_ids = [
        candidate.film_id
        for candidate in candidates
        if candidate.film_id in positive_labels
    ]
    selected: list[RecommendationCandidate] = [
        by_id[film_id] for film_id in positive_ids
    ]
    labels = [positive_labels[film_id] for film_id in positive_ids]
    strata = ["positive"] * len(selected)
    admitted = set(positive_ids) | forbidden
    negative_limit = max(0, cap - len(selected))
    negative_pool = [
        candidate for candidate in candidates if candidate.film_id not in admitted
    ]
    quotas = _negative_quotas(negative_limit)

    def take(values: list[RecommendationCandidate], quota: int, stratum: str) -> None:
        for candidate in values:
            if quota <= 0:
                break
            if candidate.film_id in admitted:
                continue
            selected.append(candidate)
            labels.append(0)
            strata.append(stratum)
            admitted.add(candidate.film_id)
            quota -= 1

    take(
        sorted(
            [candidate for candidate in negative_pool if candidate.source_count == 2],
            key=_source_rank_key,
        ),
        quotas["both_source"],
        "both_source",
    )
    take(
        sorted(
            [
                candidate
                for candidate in negative_pool
                if candidate.retrieved_by_svd and not candidate.retrieved_by_popularity
            ],
            key=lambda candidate: (candidate.svd_rank or 10**9, candidate.film_id),
        ),
        quotas["svd_only"],
        "svd_only",
    )
    take(
        sorted(
            [
                candidate
                for candidate in negative_pool
                if candidate.retrieved_by_popularity and not candidate.retrieved_by_svd
            ],
            key=lambda candidate: (
                candidate.popularity_rank or 10**9,
                candidate.film_id,
            ),
        ),
        quotas["popularity_only"],
        "popularity_only",
    )
    take(
        sorted(
            [
                candidate
                for candidate in negative_pool
                if _minimum_source_rank(candidate) <= 500
            ],
            key=_source_rank_key,
        ),
        quotas["additional_top_500"],
        "additional_top_500",
    )
    take(
        sorted(
            [
                candidate
                for candidate in negative_pool
                if 501 <= _minimum_source_rank(candidate) <= 2000
            ],
            key=_source_rank_key,
        ),
        quotas["source_tail"],
        "source_tail",
    )
    remaining = [
        candidate for candidate in negative_pool if candidate.film_id not in admitted
    ]
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(user_id) & 0xFFFFFFFF, 512])
    )
    if remaining:
        permutation = rng.permutation(len(remaining))
        shuffled = [remaining[index] for index in permutation]
    else:
        shuffled = []
    take(
        shuffled,
        quotas["uniform_remainder"],
        "uniform_remainder",
    )
    fill = sorted(
        [candidate for candidate in negative_pool if candidate.film_id not in admitted],
        key=_source_rank_key,
    )
    take(fill, cap - len(selected), "deterministic_fill")
    return SampledGroup(
        candidates=tuple(selected),
        labels=np.ascontiguousarray(labels, dtype=np.int8),
        sampling_strata=tuple(strata),
        retrieved_positive_count=len(positive_ids),
        requested_positive_count=len(positive_labels),
    )


def _negative_quotas(limit: int) -> dict[str, int]:
    fractions = (
        ("both_source", 0.25),
        ("svd_only", 0.25),
        ("popularity_only", 0.20),
        ("additional_top_500", 0.125),
        ("source_tail", 0.10),
    )
    quotas = {name: int(limit * fraction) for name, fraction in fractions}
    quotas["uniform_remainder"] = limit - sum(quotas.values())
    return quotas


def _minimum_source_rank(candidate: RecommendationCandidate) -> int:
    return min(
        candidate.svd_rank or 10**9,
        candidate.popularity_rank or 10**9,
    )


def _source_rank_key(
    candidate: RecommendationCandidate,
) -> tuple[int, int, int, int]:
    return (
        _minimum_source_rank(candidate),
        candidate.svd_rank or 10**9,
        candidate.popularity_rank or 10**9,
        candidate.film_id,
    )
