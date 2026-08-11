"""Regression tests for exact FAISS SVD recommendation retrieval."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import faiss
import numpy as np
import pytest

from app.ml.faiss_index import build_faiss_index
from app.repositories.interactions import InteractionRepository, RatedInteraction
from app.services import recommendation_service as recommendation_module
from app.services.recommendation_service import (
    ModelUnavailableError,
    RecommendationService,
    build_recommendation_service,
)


class _SessionContext:
    def __init__(self, session: object) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


class _SessionFactory:
    def __init__(self) -> None:
        self.session = object()

    def __call__(self) -> _SessionContext:
        return _SessionContext(self.session)


def _normalized_vectors(count: int = 30, dimension: int = 8) -> np.ndarray:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(count, dimension))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _write_artifacts(
    tmp_path: object,
    vectors: np.ndarray,
    film_ids: list[int] | None = None,
) -> list[int]:
    ids = film_ids or list(range(1, len(vectors) + 1))
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text(json.dumps(ids), encoding="utf-8")
    build_faiss_index(vectors, ids, tmp_path / "retrieval.faiss")
    return ids


def _film(film_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=film_id,
        title=f"Film {film_id}",
        directors=[SimpleNamespace(name="Director")],
        year=2000 + film_id,
    )


def test_exact_retrieval_preserves_mean_scores_exclusion_order_and_limit(
    tmp_path, monkeypatch
) -> None:
    vectors = _normalized_vectors()
    film_ids = _write_artifacts(tmp_path, vectors)
    service = RecommendationService(_SessionFactory(), tmp_path)
    assert service.load_artifacts()

    rated_ids = [1, 7, 13]
    watched_ids = [1, 2, 7, 13]
    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_rated_film_ids",
        AsyncMock(return_value=rated_ids),
    )
    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_watched_film_ids",
        AsyncMock(return_value=watched_ids),
    )

    user_vector = np.mean(vectors[[0, 6, 12]], axis=0)
    numpy_scores = vectors @ user_vector
    oracle_ids = sorted(
        (film_id for film_id in film_ids if film_id not in watched_ids),
        key=lambda film_id: float(numpy_scores[film_id - 1]),
        reverse=True,
    )
    missing_id = oracle_ids[0]
    database_films = [
        _film(film_id) for film_id in reversed(oracle_ids) if film_id != missing_id
    ]
    film_lookup = AsyncMock(return_value=database_films)
    monkeypatch.setattr(
        recommendation_module.FilmRepository,
        "get_by_ids",
        film_lookup,
    )

    original_mean = np.mean
    observed: dict[str, np.ndarray] = {}

    def recording_mean(values: np.ndarray, axis: int) -> np.ndarray:
        observed["pooled"] = values.copy()
        return original_mean(values, axis=axis)

    monkeypatch.setattr(recommendation_module.np, "mean", recording_mean)
    monkeypatch.setattr(
        recommendation_module.np,
        "dot",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("full-catalog NumPy scoring must not run")
        ),
    )

    result = asyncio.run(service.recommend(42))

    np.testing.assert_array_equal(observed["pooled"], vectors[[0, 6, 12]])
    expected_ids = [film_id for film_id in oracle_ids if film_id != missing_id][:10]
    assert [item.id for item in result.recommendations] == expected_ids
    np.testing.assert_allclose(
        [item.match_score for item in result.recommendations],
        [round(float(numpy_scores[film_id - 1]), 4) for film_id in expected_ids],
        atol=1e-4,
    )
    assert len(result.recommendations) == 10
    assert set(watched_ids).isdisjoint(item.id for item in result.recommendations)
    assert result.strategy == "SVD_Mean_Pooling"
    film_lookup.assert_awaited_once()


class _SearchProxy:
    def __init__(
        self,
        index: faiss.IndexIDMap2,
        *,
        labels: np.ndarray | None = None,
        scores: np.ndarray | None = None,
    ) -> None:
        self._index = index
        self.ntotal = index.ntotal
        self.labels = labels
        self.scores = scores
        self.requested: list[int] = []

    def search(self, query: np.ndarray, requested_k: int):
        self.requested.append(requested_k)
        if self.labels is not None and self.scores is not None:
            return self.scores[:, :requested_k], self.labels[:, :requested_k]
        return self._index.search(query, requested_k)


def test_retrieval_depth_compensates_watched_and_caps_at_index_size(
    tmp_path, monkeypatch
) -> None:
    vectors = _normalized_vectors(12, 4)
    _write_artifacts(tmp_path, vectors)
    service = RecommendationService(_SessionFactory(), tmp_path)
    assert service.load_artifacts()
    assert service._artifacts is not None
    proxy = _SearchProxy(service._artifacts.retrieval_index)
    service._artifacts = replace(service._artifacts, retrieval_index=proxy)
    service._retrieval_top_k = 6

    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_rated_film_ids",
        AsyncMock(return_value=[1]),
    )
    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_watched_film_ids",
        AsyncMock(return_value=[1, 2, 3]),
    )
    monkeypatch.setattr(
        recommendation_module.FilmRepository,
        "get_by_ids",
        AsyncMock(return_value=[_film(film_id) for film_id in range(4, 13)]),
    )

    asyncio.run(service.recommend(1))

    assert proxy.requested == [9]


def test_invalid_faiss_labels_are_ignored(tmp_path, monkeypatch) -> None:
    vectors = _normalized_vectors(4, 3)
    _write_artifacts(tmp_path, vectors)
    service = RecommendationService(_SessionFactory(), tmp_path)
    assert service.load_artifacts()
    assert service._artifacts is not None
    proxy = _SearchProxy(
        service._artifacts.retrieval_index,
        labels=np.array([[1, -1, 3, 4]], dtype=np.int64),
        scores=np.array([[1.0, -np.inf, 0.8, 0.7]], dtype=np.float32),
    )
    service._artifacts = replace(service._artifacts, retrieval_index=proxy)
    service._retrieval_top_k = 3

    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_rated_film_ids",
        AsyncMock(return_value=[1]),
    )
    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_watched_film_ids",
        AsyncMock(return_value=[1]),
    )
    monkeypatch.setattr(
        recommendation_module.FilmRepository,
        "get_by_ids",
        AsyncMock(return_value=[_film(3), _film(4)]),
    )

    result = asyncio.run(service.recommend(1))

    assert [item.id for item in result.recommendations] == [3, 4]
    assert proxy.requested == [4]


def test_interaction_repository_filters_unrated_rows() -> None:
    scalar_result = SimpleNamespace(all=lambda: [7])
    execute_result = SimpleNamespace(scalars=lambda: scalar_result)
    session = SimpleNamespace(execute=AsyncMock(return_value=execute_result))

    film_ids = asyncio.run(InteractionRepository(session).get_rated_film_ids(3))

    assert film_ids == [7]
    statement = session.execute.await_args.args[0]
    assert "logs.rating IS NOT NULL" in str(statement)


def test_interaction_repository_returns_all_watched_rows() -> None:
    scalar_result = SimpleNamespace(all=lambda: [7, 8])
    execute_result = SimpleNamespace(scalars=lambda: scalar_result)
    session = SimpleNamespace(execute=AsyncMock(return_value=execute_result))

    film_ids = asyncio.run(InteractionRepository(session).get_watched_film_ids(3))

    assert film_ids == [7, 8]
    statement = session.execute.await_args.args[0]
    assert "logs.rating IS NOT NULL" not in str(statement)


def test_interaction_repository_returns_rated_film_and_rating_pairs() -> None:
    execute_result = SimpleNamespace(all=lambda: [(7, 4.5), (8, 1.0)])
    session = SimpleNamespace(execute=AsyncMock(return_value=execute_result))

    interactions = asyncio.run(InteractionRepository(session).get_rated_interactions(3))

    assert interactions == [RatedInteraction(7, 4.5), RatedInteraction(8, 1.0)]
    statement = session.execute.await_args.args[0]
    assert "logs.rating IS NOT NULL" in str(statement)


def test_watched_profile_without_usable_ratings_returns_accurate_info(
    tmp_path, monkeypatch
) -> None:
    _write_artifacts(tmp_path, np.array([[1.0, 0.0], [0.0, 1.0]]))
    service = RecommendationService(_SessionFactory(), tmp_path)
    assert service.load_artifacts()

    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_watched_film_ids",
        AsyncMock(return_value=[2]),
    )
    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_rated_film_ids",
        AsyncMock(return_value=[]),
    )

    result = asyncio.run(service.recommend(3))

    assert result.recommendations == ()
    assert result.info is not None
    assert "No rated films" in result.info
    assert "No watched films" not in result.info


def test_valid_three_artifact_set_loads(tmp_path) -> None:
    _write_artifacts(tmp_path, _normalized_vectors(3, 2), [10, 20, 30])
    service = RecommendationService(_SessionFactory(), tmp_path)

    assert service.load_artifacts()
    assert service.is_model_loaded


def test_missing_faiss_artifact_makes_model_unavailable(tmp_path) -> None:
    vectors = _normalized_vectors(2, 2)
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text("[1, 2]", encoding="utf-8")
    service = RecommendationService(_SessionFactory(), tmp_path)

    assert not service.load_artifacts()
    assert not service.is_model_loaded
    with pytest.raises(ModelUnavailableError):
        asyncio.run(service.recommend(1))


def test_corrupt_faiss_artifact_makes_model_unavailable(tmp_path) -> None:
    vectors = _normalized_vectors(2, 2)
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text("[1, 2]", encoding="utf-8")
    (tmp_path / "retrieval.faiss").write_bytes(b"not an index")
    service = RecommendationService(_SessionFactory(), tmp_path)

    assert not service.load_artifacts()
    assert not service.is_model_loaded


@pytest.mark.parametrize("inconsistency", ["dimension", "count", "ids"])
def test_inconsistent_faiss_artifact_is_rejected(tmp_path, inconsistency: str) -> None:
    vectors = _normalized_vectors(3, 2)
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text("[1, 2, 3]", encoding="utf-8")

    if inconsistency == "dimension":
        index_vectors = _normalized_vectors(3, 3)
        index_ids = [1, 2, 3]
    elif inconsistency == "count":
        index_vectors = vectors[:2]
        index_ids = [1, 2]
    else:
        index_vectors = vectors
        index_ids = [1, 2, 99]
    build_faiss_index(
        index_vectors,
        index_ids,
        tmp_path / "retrieval.faiss",
    )
    service = RecommendationService(_SessionFactory(), tmp_path)

    assert not service.load_artifacts()
    assert not service.is_model_loaded


def test_service_factory_builds_live_svd_service(tmp_path) -> None:
    service = build_recommendation_service(artifact_root=tmp_path)

    assert isinstance(service, RecommendationService)
