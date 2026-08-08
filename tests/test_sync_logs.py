"""Tests for idempotent rated and unrated interaction synchronization."""

import asyncio
from types import SimpleNamespace

import pytest

from app.db.loaders.sync_logs import (
    InconsistentIngestionStateError,
    sync_user_logs,
)
from app.domain.profiles import ScrapedProfile, ScrapedWatch
from app.models.status import Status


class _Result:
    def __init__(self, rows=(), scalar=None) -> None:
        self._rows = list(rows)
        self._scalar = scalar

    def all(self):
        return self._rows

    def scalars(self):
        return self

    def scalar_one_or_none(self):
        return self._scalar


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _Session:
    def __init__(self, query_results: list[_Result]) -> None:
        self._query_results = iter(query_results)
        self.inserts: list[tuple[str, list[dict[str, object]]]] = []

    def begin(self) -> _Transaction:
        return _Transaction()

    async def execute(self, statement, parameters=None):
        if parameters is None:
            return next(self._query_results)
        self.inserts.append((statement.table.name, parameters))
        return _Result()


class _SessionContext:
    def __init__(self, session: _Session) -> None:
        self._session = session

    async def __aenter__(self) -> _Session:
        return self._session

    async def __aexit__(self, *args: object) -> None:
        return None


def test_unrated_known_and_unknown_watches_are_persisted() -> None:
    session = _Session(
        [
            _Result(scalar=SimpleNamespace(id=5, username="cinephile")),
            _Result(rows=[("known", 11)]),
            _Result(),
            _Result(),
            _Result(),
        ]
    )
    profile = ScrapedProfile(
        username="cinephile",
        watches=(
            ScrapedWatch(film_slug="known", rating=None),
            ScrapedWatch(film_slug="unknown", rating=None),
        ),
    )

    asyncio.run(
        sync_user_logs(profile, session_factory=lambda: _SessionContext(session))
    )

    inserts = {table: rows for table, rows in session.inserts}
    assert inserts["logs"] == [{"user_id": 5, "film_id": 11, "rating": None}]
    assert inserts["logs_pending"] == [
        {
            "username": "cinephile",
            "film_slug": "unknown",
            "rating": None,
            "status": Status.PENDING,
        }
    ]
    assert inserts["films_queue"] == [{"film_slug": "unknown"}]


@pytest.mark.parametrize(
    ("previous_rating", "scraped_rating"),
    [(None, 4.5), (3.0, 4.5), (4.0, None)],
)
def test_existing_log_rating_is_refreshed(previous_rating, scraped_rating) -> None:
    existing_log = SimpleNamespace(film_id=11, rating=previous_rating)
    session = _Session(
        [
            _Result(scalar=SimpleNamespace(id=5, username="cinephile")),
            _Result(rows=[("known", 11)]),
            _Result(rows=[existing_log]),
            _Result(),
            _Result(),
        ]
    )
    profile = ScrapedProfile(
        username="cinephile",
        watches=(ScrapedWatch(film_slug="known", rating=scraped_rating),),
    )

    asyncio.run(
        sync_user_logs(profile, session_factory=lambda: _SessionContext(session))
    )

    assert existing_log.rating == scraped_rating
    assert all(table != "logs" for table, _ in session.inserts)


@pytest.mark.parametrize("queue_status", [Status.FILTERED, Status.FAILED])
def test_terminal_queue_creates_terminal_pending_log(queue_status) -> None:
    queue_item = SimpleNamespace(film_slug="unknown", status=queue_status)
    session = _Session(
        [
            _Result(scalar=SimpleNamespace(id=5, username="cinephile")),
            _Result(),
            _Result(),
            _Result(rows=[queue_item]),
            _Result(),
        ]
    )
    profile = ScrapedProfile(
        username="cinephile",
        watches=(ScrapedWatch(film_slug="unknown", rating=4.0),),
    )

    asyncio.run(
        sync_user_logs(profile, session_factory=lambda: _SessionContext(session))
    )

    inserts = {table: rows for table, rows in session.inserts}
    assert "films_queue" not in inserts
    assert inserts["logs_pending"][0]["status"] == queue_status


@pytest.mark.parametrize(
    ("previous_rating", "scraped_rating"),
    [(None, 4.5), (3.0, 4.5), (4.0, None)],
)
def test_existing_terminal_pending_rating_refresh_preserves_status(
    previous_rating, scraped_rating
) -> None:
    queue_item = SimpleNamespace(film_slug="unknown", status=Status.FAILED)
    pending_log = SimpleNamespace(
        film_slug="unknown", rating=previous_rating, status=Status.FAILED
    )
    session = _Session(
        [
            _Result(scalar=SimpleNamespace(id=5, username="cinephile")),
            _Result(),
            _Result(),
            _Result(rows=[queue_item]),
            _Result(rows=[pending_log]),
        ]
    )
    profile = ScrapedProfile(
        username="cinephile",
        watches=(ScrapedWatch(film_slug="unknown", rating=scraped_rating),),
    )

    asyncio.run(
        sync_user_logs(profile, session_factory=lambda: _SessionContext(session))
    )

    assert pending_log.rating == scraped_rating
    assert pending_log.status == Status.FAILED
    assert all(table != "logs_pending" for table, _ in session.inserts)


@pytest.mark.parametrize("queue_status", [Status.FILTERED, Status.FAILED])
def test_existing_pending_log_adopts_terminal_queue_status(queue_status) -> None:
    queue_item = SimpleNamespace(film_slug="unknown", status=queue_status)
    pending_log = SimpleNamespace(
        film_slug="unknown", rating=3.0, status=Status.PENDING
    )
    session = _Session(
        [
            _Result(scalar=SimpleNamespace(id=5, username="cinephile")),
            _Result(),
            _Result(),
            _Result(rows=[queue_item]),
            _Result(rows=[pending_log]),
        ]
    )
    profile = ScrapedProfile(
        username="cinephile",
        watches=(ScrapedWatch(film_slug="unknown", rating=4.5),),
    )

    asyncio.run(
        sync_user_logs(profile, session_factory=lambda: _SessionContext(session))
    )

    assert pending_log.rating == 4.5
    assert pending_log.status == queue_status


def test_processed_queue_without_catalog_film_fails_explicitly() -> None:
    queue_item = SimpleNamespace(film_slug="unknown", status=Status.PROCESSED)
    session = _Session(
        [
            _Result(scalar=SimpleNamespace(id=5, username="cinephile")),
            _Result(),
            _Result(),
            _Result(rows=[queue_item]),
            _Result(),
        ]
    )
    profile = ScrapedProfile(
        username="cinephile",
        watches=(ScrapedWatch(film_slug="unknown", rating=None),),
    )

    with pytest.raises(InconsistentIngestionStateError):
        asyncio.run(
            sync_user_logs(profile, session_factory=lambda: _SessionContext(session))
        )
