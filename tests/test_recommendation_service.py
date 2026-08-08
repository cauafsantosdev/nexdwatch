"""Regression tests for the SVD recommendation service."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.repositories.interactions import InteractionRepository
from app.services import recommendation_service as recommendation_module
from app.services.recommendation_service import (
    ModelUnavailableError,
    RecommendationService,
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


def _write_artifacts(tmp_path: object, vectors: np.ndarray) -> None:
    np.save(tmp_path / "item_embeddings.npy", vectors)
    (tmp_path / "film_index.json").write_text(
        json.dumps(list(range(1, len(vectors) + 1))), encoding="utf-8"
    )


def test_mean_dot_order_exclusion_and_limit(tmp_path, monkeypatch) -> None:
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0]]
        + [[value, value] for value in np.linspace(0.1, 1.0, 10)],
        dtype=float,
    )
    _write_artifacts(tmp_path, vectors)
    session_factory = _SessionFactory()
    service = RecommendationService(session_factory, tmp_path)
    assert service.load_artifacts()

    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_rated_film_ids",
        AsyncMock(return_value=[1]),
    )
    monkeypatch.setattr(
        recommendation_module.InteractionRepository,
        "get_watched_film_ids",
        AsyncMock(return_value=[1, 2]),
    )
    films = [
        SimpleNamespace(
            id=film_id,
            title=f"Film {film_id}",
            directors=[SimpleNamespace(name="Director")],
            year=2000 + film_id,
        )
        for film_id in range(1, 13)
    ]
    monkeypatch.setattr(
        recommendation_module.FilmRepository,
        "get_by_ids",
        AsyncMock(return_value=films),
    )

    original_mean = np.mean
    original_dot = np.dot
    observed: dict[str, np.ndarray] = {}

    def recording_mean(values: np.ndarray, axis: int) -> np.ndarray:
        observed["pooled"] = values.copy()
        return original_mean(values, axis=axis)

    def recording_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        observed["user_vector"] = right.copy()
        return original_dot(left, right)

    monkeypatch.setattr(recommendation_module.np, "mean", recording_mean)
    monkeypatch.setattr(recommendation_module.np, "dot", recording_dot)

    result = asyncio.run(service.recommend(42))

    np.testing.assert_array_equal(observed["pooled"], vectors[[0]])
    np.testing.assert_array_equal(
        observed["user_vector"], original_mean(vectors[[0]], axis=0)
    )
    assert len(result.recommendations) == 10
    assert {item.id for item in result.recommendations}.isdisjoint({1, 2})
    scores = [item.match_score for item in result.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert result.strategy == "SVD_Mean_Pooling"


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


def test_missing_artifacts_raise_model_unavailable(tmp_path) -> None:
    service = RecommendationService(_SessionFactory(), tmp_path)

    assert not service.load_artifacts()
    assert not service.is_model_loaded
    with pytest.raises(ModelUnavailableError):
        asyncio.run(service.recommend(1))
