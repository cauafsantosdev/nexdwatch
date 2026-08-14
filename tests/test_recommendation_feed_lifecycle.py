"""Application ownership of categorized recommendation resources."""

import asyncio

import pytest
from fastapi import FastAPI
from starlette.requests import Request

from app import main as main_module
from app.api.routes.health import health


class _OldRecommendationService:
    def __init__(self) -> None:
        self.load_calls = 0
        self.unload_calls = 0

    def load_artifacts(self) -> bool:
        self.load_calls += 1
        return True

    def unload_artifacts(self) -> None:
        self.unload_calls += 1


class _CategorizedService:
    def __init__(self, loaded: bool = True) -> None:
        self.loaded = loaded
        self.load_calls = 0
        self.unload_calls = 0

    async def load_resources(self) -> bool:
        self.load_calls += 1
        return self.loaded

    def unload_resources(self) -> None:
        self.unload_calls += 1


class _TaskService:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def _configure(monkeypatch, *, categorized_loaded: bool = True):
    old = _OldRecommendationService()
    categorized = _CategorizedService(categorized_loaded)
    tasks = _TaskService()
    monkeypatch.setattr(main_module, "get_recommendation_service", lambda: old)
    monkeypatch.setattr(
        main_module, "build_categorized_recommendation_service", lambda: categorized
    )
    monkeypatch.setattr(main_module, "get_task_service", lambda: tasks)
    return old, categorized, tasks


def test_lifespan_loads_once_reuses_state_and_unloads_on_shutdown(monkeypatch) -> None:
    old, categorized, tasks = _configure(monkeypatch)
    application = FastAPI()

    async def exercise() -> None:
        async with main_module.lifespan(application):
            assert application.state.categorized_recommendation_service is categorized
            assert application.state.model_version == "legacy-flat"
            assert categorized.load_calls == 1
            assert old.load_calls == 1

    asyncio.run(exercise())

    assert categorized.unload_calls == 1
    assert old.unload_calls == 1
    assert tasks.close_calls == 1
    assert not hasattr(application.state, "categorized_recommendation_service")


def test_failed_categorized_load_aborts_startup_and_cleans_up(monkeypatch) -> None:
    old, categorized, tasks = _configure(monkeypatch, categorized_loaded=False)
    application = FastAPI()

    async def exercise() -> None:
        with pytest.raises(
            RuntimeError, match="categorized recommendation resources unavailable"
        ):
            async with main_module.lifespan(application):
                raise AssertionError("failed resource load must not enter application")

    asyncio.run(exercise())

    assert categorized.load_calls == 1
    assert categorized.unload_calls == 1
    assert old.unload_calls == 1
    assert tasks.close_calls == 1


def test_health_reports_currently_loaded_model_version() -> None:
    application = FastAPI()
    application.state.model_version = "20300102T010203Z-00000002"
    request = Request({"type": "http", "app": application})
    service = _OldRecommendationService()
    service.is_model_loaded = True

    response = asyncio.run(health(request, service))

    assert response.model_status == "loaded"
    assert response.model_version == "20300102T010203Z-00000002"
