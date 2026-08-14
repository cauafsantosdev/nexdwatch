import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from app.domain.candidates import CandidateGenerationResult, RecommendationCandidate
from app.ml.popularity import PopularityEntry
from app.policy.catalog import PolicyCatalog, PolicyEntity, PolicyFilm
from app.policy.engine import CategorizedPolicyEngine
from app.repositories.interactions import InteractionRepository, RecommendationHistory
from app.services.categorized_recommendation_service import (
    CategorizedRecommendationService,
)
from experiments.category_policy.serving_performance import semantic_fingerprint


class _SessionContext:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _CountingSessionFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> _SessionContext:
        self.calls += 1
        return _SessionContext()


class _CandidateService:
    def __init__(self, candidates, popularity, svd) -> None:
        self.is_loaded = True
        self.popularity_artifact = popularity
        self.svd_artifacts = svd
        self._candidates = candidates
        self.unloaded = False

    def generate_from_history(self, user_id, _history, *, profiler=None):
        del profiler
        return CandidateGenerationResult(user_id, self._candidates, 40, 20, 20, True)

    def load_artifacts(self) -> bool:
        self.is_loaded = True
        return True

    def unload_artifacts(self) -> None:
        self.is_loaded = False
        self.popularity_artifact = None
        self.svd_artifacts = None
        self.unloaded = True


def _loaded_service() -> tuple[CategorizedRecommendationService, object, object]:
    session_factory = _CountingSessionFactory()
    service = CategorizedRecommendationService(session_factory, "/unused")
    director = PolicyEntity(7, "Director")
    films = {
        film_id: PolicyFilm(
            film_id,
            f"Film {film_id}",
            2000 + film_id,
            directors=(director,),
        )
        for film_id in range(1, 21)
    }
    catalog = PolicyCatalog(films, frozenset(films))
    candidates = tuple(
        RecommendationCandidate(
            film_id,
            svd_score=1 / film_id,
            svd_rank=film_id,
            popularity_score=21 - film_id,
            popularity_rank=film_id,
            retrieved_by_svd=True,
            retrieved_by_popularity=True,
        )
        for film_id in range(1, 21)
    )
    popularity_entries = tuple(
        PopularityEntry(film_id, 21 - film_id, film_id) for film_id in range(1, 21)
    )
    popularity = SimpleNamespace(film_count=20, films=popularity_entries)
    vectors = np.eye(20, dtype=np.float32)
    svd = SimpleNamespace(
        item_vectors=vectors,
        id_to_position={film_id: film_id - 1 for film_id in films},
        film_index=np.asarray(tuple(films), dtype=np.int64),
    )
    service._candidate_service = _CandidateService(candidates, popularity, svd)
    service._catalog = catalog
    service._policy_engine = CategorizedPolicyEngine(
        catalog, vectors, svd.id_to_position, config=service._config
    )
    service._popularity_rank_by_film = {
        entry.film_id: entry.rank for entry in popularity_entries
    }
    return service, session_factory, service._candidate_service


def test_repeated_recommendations_are_exact_and_use_one_request_query(
    monkeypatch,
) -> None:
    service, session_factory, _ = _loaded_service()
    history_read = AsyncMock(return_value=RecommendationHistory((), ()))
    monkeypatch.setattr(
        InteractionRepository,
        "get_existing_user_recommendation_history",
        history_read,
    )

    first = asyncio.run(service.recommend(7))
    second = asyncio.run(service.recommend(7))

    assert first == second
    assert semantic_fingerprint(first) == semantic_fingerprint(second)
    assert (
        semantic_fingerprint(first)["sha256"]
        == "86b80f68604de2c18cc4fbefdf660862c1383d4c23084833bf9ced12301e59b7"
    )
    assert [category.key for category in first.categories] == ["top_picks"]
    assert [item.film_id for item in first.categories[0].items] == list(range(1, 21))
    assert all(
        item.reason.code.value == "GLOBAL_RRF"
        and [code.value for code in item.reason.additional_codes]
        == ["SOURCE_AGREEMENT"]
        for item in first.categories[0].items
    )
    assert history_read.await_count == 2
    assert session_factory.calls == 2
    assert all(
        value is not first and value is not second for value in vars(service).values()
    )


def test_resource_unload_drops_all_service_scoped_resources() -> None:
    service, _, candidate_service = _loaded_service()

    assert service.is_loaded
    service.unload_resources()

    assert candidate_service.unloaded
    assert not service.is_loaded
    assert service._catalog is None
    assert service._policy_engine is None
    assert service._popularity_rank_by_film is None


def test_resource_load_and_unload_lifecycle(monkeypatch) -> None:
    service, session_factory, candidate_service = _loaded_service()
    catalog = service._catalog
    service._catalog = None
    service._policy_engine = None
    service._popularity_rank_by_film = None
    catalog_loader = AsyncMock(return_value=catalog)
    monkeypatch.setattr(
        "app.services.categorized_recommendation_service.load_policy_catalog",
        catalog_loader,
    )

    assert asyncio.run(service.load_resources())
    assert service.is_loaded
    assert session_factory.calls == 1
    catalog_loader.assert_awaited_once()

    service.unload_resources()
    assert candidate_service.unloaded
    assert not service.is_loaded
