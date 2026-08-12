"""Tests for the ranking-independent hybrid candidate application boundary."""

import asyncio
import json
from unittest.mock import AsyncMock

import numpy as np

from app.ml.candidate_policy import (
    FINAL_CANDIDATE_NOMINAL_BUDGET,
    FINAL_POPULARITY_DEPTH,
    FINAL_WEIGHTED_SVD_DEPTH,
)
from app.ml.candidate_retrieval import retrieve_exact_candidates
from app.ml.faiss_index import build_faiss_index
from app.ml.popularity import (
    POPULARITY_ARTIFACT_SCHEMA,
    POPULARITY_RATING_THRESHOLD,
    POPULARITY_SOURCE,
    PopularityArtifact,
    PopularityEntry,
    write_popularity_artifact,
)
from app.ml.ratings import rating_to_bucket
from app.ml.svd_profiles import build_svd_profile
from app.repositories.interactions import RatedInteraction, RecommendationHistory
from app.services import candidate_generation_service as candidate_module
from app.services.candidate_generation_service import CandidateGenerationService


class _SessionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: object) -> None:
        return None


class _SessionFactory:
    def __call__(self) -> _SessionContext:
        return _SessionContext()


def _artifacts(tmp_path) -> tuple[np.ndarray, list[int]]:
    raw = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    film_ids = [1, 2, 3, 4, 5, 6]
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text(json.dumps(film_ids), encoding="utf-8")
    build_faiss_index(vectors, film_ids, tmp_path / "retrieval.faiss")
    popularity = PopularityArtifact(
        schema=POPULARITY_ARTIFACT_SCHEMA,
        rating_threshold=POPULARITY_RATING_THRESHOLD,
        film_count=6,
        source_description=POPULARITY_SOURCE,
        films=tuple(
            PopularityEntry(film_id, count, rank)
            for rank, (film_id, count) in enumerate(
                [(3, 60), (2, 50), (4, 40), (5, 30), (6, 20), (1, 10)],
                start=1,
            )
        ),
    )
    write_popularity_artifact(popularity, tmp_path / "candidates/popularity.json")
    return vectors, film_ids


def test_final_candidate_defaults_freeze_measured_source_allocation() -> None:
    service = CandidateGenerationService(_SessionFactory(), "/unused")

    assert FINAL_CANDIDATE_NOMINAL_BUDGET == 4000
    assert FINAL_WEIGHTED_SVD_DEPTH == 2000
    assert FINAL_POPULARITY_DEPTH == 2000
    assert service._svd_depth == FINAL_WEIGHTED_SVD_DEPTH
    assert service._popularity_depth == FINAL_POPULARITY_DEPTH
    assert service._svd_depth + service._popularity_depth == (
        FINAL_CANDIDATE_NOMINAL_BUDGET
    )


def test_candidate_generation_preserves_parity_exclusion_and_provenance(
    tmp_path, monkeypatch
) -> None:
    vectors, _ = _artifacts(tmp_path)
    watched = [1, 3]
    rated = [RatedInteraction(1, 5.0), RatedInteraction(2, 2.0)]
    monkeypatch.setattr(
        candidate_module.InteractionRepository,
        "get_recommendation_history",
        AsyncMock(return_value=RecommendationHistory(tuple(watched), tuple(rated))),
    )
    service = CandidateGenerationService(
        _SessionFactory(), tmp_path, svd_depth=3, popularity_depth=3
    )
    assert service.load_artifacts()
    assert service._svd_artifacts is not None

    query = build_svd_profile(
        vectors,
        np.array([0, 1], dtype=np.int64),
        np.array([rating_to_bucket(5.0), rating_to_bucket(2.0)], dtype=np.int64),
        "svd_positive_weighted",
    )
    assert query is not None
    offline = retrieve_exact_candidates(
        service._svd_artifacts.retrieval_index,
        query,
        excluded_film_ids=watched,
        depth=3,
    )

    first = asyncio.run(service.generate(7))
    second = asyncio.run(service.generate(7))

    assert first == second
    assert first.svd_profile_available
    assert [
        candidate.film_id
        for candidate in first.candidates
        if candidate.retrieved_by_svd
    ] == [film_id for film_id, _ in offline]
    assert set(watched).isdisjoint(candidate.film_id for candidate in first.candidates)
    assert len({candidate.film_id for candidate in first.candidates}) == len(
        first.candidates
    )
    assert first.unique_candidate_count < first.nominal_budget
    overlap = [
        candidate for candidate in first.candidates if candidate.source_count == 2
    ]
    assert overlap
    assert all(candidate.svd_rank is not None for candidate in overlap)
    assert all(candidate.popularity_rank is not None for candidate in overlap)
    assert sorted(
        candidate.popularity_rank
        for candidate in first.candidates
        if candidate.retrieved_by_popularity
    ) == [2, 3, 4]


def test_no_positive_profile_does_not_fall_back_to_mean_svd(
    tmp_path, monkeypatch
) -> None:
    _artifacts(tmp_path)
    monkeypatch.setattr(
        candidate_module.InteractionRepository,
        "get_recommendation_history",
        AsyncMock(
            return_value=RecommendationHistory((1,), (RatedInteraction(1, 3.0),))
        ),
    )
    service = CandidateGenerationService(
        _SessionFactory(), tmp_path, svd_depth=3, popularity_depth=2
    )
    assert service.load_artifacts()

    result = asyncio.run(service.generate(9))

    assert not result.svd_profile_available
    assert all(not candidate.retrieved_by_svd for candidate in result.candidates)
    assert [candidate.film_id for candidate in result.candidates] == [3, 2]
