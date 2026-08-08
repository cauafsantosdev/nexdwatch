"""Tests for film scraper outcomes and queue orchestration."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.db.loaders import sync_queue
from app.models.status import Status
from app.scraper import film_scraping
from app.scraper.film_scraping import FilmScrapeOutcome, FilmScrapeResult


def _film(total_logs: int = 1_500) -> SimpleNamespace:
    return SimpleNamespace(
        id=99,
        title="Title",
        original_title="Original",
        year=2020,
        runtime=100,
        director=["Director"],
        genre=["Drama"],
        country=["Brazil"],
        language=["Portuguese"],
        actors=["Actor"],
        studio=["Studio"],
        synopsis="Synopsis",
        tagline="Tagline",
        themes=["Theme"],
        avg_rating=4.0,
        total_logs=total_logs,
    )


def test_film_scraper_uses_tmdb_id_and_reports_all_outcomes(monkeypatch) -> None:
    films = {
        "success": _film(),
        "filtered": _film(total_logs=999),
    }

    def get_film(slug: str) -> SimpleNamespace:
        if slug == "failed":
            raise RuntimeError("not found")
        return films[slug]

    monkeypatch.setattr(
        film_scraping,
        "Scrapxd",
        lambda: SimpleNamespace(get_film=get_film),
    )

    results = film_scraping.scrape_film_queue(["success", "filtered", "failed"])

    assert [result.outcome for result in results] == [
        FilmScrapeOutcome.SUCCESS,
        FilmScrapeOutcome.FILTERED,
        FilmScrapeOutcome.FAILED,
    ]
    assert results[0].metadata["tmdb_id"] == 99
    assert "id" not in results[0].metadata


def test_pending_slug_query_calls_scalars_method() -> None:
    scalar_rows = SimpleNamespace(all=lambda: ["film-a", "film-b"])
    execute_result = SimpleNamespace(scalars=lambda: scalar_rows)
    session = SimpleNamespace(execute=AsyncMock(return_value=execute_result))

    class Context:
        async def __aenter__(self) -> object:
            return session

        async def __aexit__(self, *args: object) -> None:
            return None

    slugs = asyncio.run(sync_queue._get_pending_slugs(lambda: Context()))

    assert slugs == ["film-a", "film-b"]


def test_queue_routes_success_filtered_and_failure_to_terminal_states(
    monkeypatch,
) -> None:
    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(sync_queue.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(
        sync_queue,
        "_get_pending_slugs",
        AsyncMock(return_value=["success", "filtered", "failed", "missing"]),
    )
    persist_success = AsyncMock()
    mark_terminal = AsyncMock()
    monkeypatch.setattr(sync_queue, "_persist_success", persist_success)
    monkeypatch.setattr(sync_queue, "_mark_terminal", mark_terminal)

    def scraper(_: list[str]) -> list[FilmScrapeResult]:
        return [
            FilmScrapeResult("success", FilmScrapeOutcome.SUCCESS, metadata={}),
            FilmScrapeResult(
                "filtered", FilmScrapeOutcome.FILTERED, error="below threshold"
            ),
            FilmScrapeResult("failed", FilmScrapeOutcome.FAILED, error="not found"),
        ]

    asyncio.run(sync_queue.sync_film_queue(session_factory=object(), scraper=scraper))

    persist_success.assert_awaited_once()
    terminal_calls = [call.args[2:] for call in mark_terminal.await_args_list]
    assert (Status.FILTERED, "below threshold") in terminal_calls
    assert (Status.FAILED, "not found") in terminal_calls
    assert (Status.FAILED, "Scraper returned no result") in terminal_calls


@pytest.mark.parametrize("terminal_status", [Status.FILTERED, Status.FAILED])
def test_terminal_update_propagates_to_pending_logs(terminal_status) -> None:
    queue_item = SimpleNamespace(
        attempts=2,
        status=Status.PENDING,
        last_error=None,
        updated_at=None,
    )
    pending_log = SimpleNamespace(status=Status.PENDING)
    processed_log = SimpleNamespace(status=Status.PROCESSED)
    pending_rows = SimpleNamespace(all=lambda: [pending_log])
    execute_result = SimpleNamespace(scalars=lambda: pending_rows)
    session = SimpleNamespace(
        scalar=AsyncMock(return_value=queue_item),
        execute=AsyncMock(return_value=execute_result),
    )

    class Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *args: object) -> None:
            return None

    session.begin = lambda: Transaction()

    class Context:
        async def __aenter__(self) -> object:
            return session

        async def __aexit__(self, *args: object) -> None:
            return None

    asyncio.run(
        sync_queue._mark_terminal(
            lambda: Context(), "film", terminal_status, "terminal reason"
        )
    )

    assert queue_item.attempts == 3
    assert queue_item.status == terminal_status
    assert queue_item.last_error == "terminal reason"
    assert queue_item.updated_at is not None
    assert pending_log.status == terminal_status
    assert processed_log.status == Status.PROCESSED
    statement = session.execute.await_args.args[0]
    assert "logs_pending.status" in str(statement)
    assert "logs_pending.film_slug" in str(statement)


def test_success_update_marks_processed_and_clears_error(monkeypatch) -> None:
    queue_item = SimpleNamespace(
        attempts=0,
        status=Status.PENDING,
        last_error="previous error",
        updated_at=None,
    )
    film = SimpleNamespace(id=4, slug="film")
    session = SimpleNamespace(
        scalar=AsyncMock(side_effect=[queue_item, film]),
    )

    class Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, *args: object) -> None:
            return None

    session.begin = lambda: Transaction()

    class Context:
        async def __aenter__(self) -> object:
            return session

        async def __aexit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(sync_queue, "_attach_pending_logs", AsyncMock())
    result = FilmScrapeResult(
        slug="film",
        outcome=FilmScrapeOutcome.SUCCESS,
        metadata={"title": "Film"},
    )

    asyncio.run(sync_queue._persist_success(lambda: Context(), result))

    assert queue_item.attempts == 1
    assert queue_item.status == Status.PROCESSED
    assert queue_item.last_error is None
    assert queue_item.updated_at is not None
