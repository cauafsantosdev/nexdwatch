"""Policy, schedule, lock, and bounded maintenance tests."""

import asyncio
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.db.loaders import sync_queue
from app.domain.maintenance import TrainedModelStatistics, TrainingStatistics
from app.infrastructure.maintenance_lock import MaintenanceLock
from app.ml.model_lifecycle import decide_retraining
from app.scraper import FilmScrapeOutcome, FilmScrapeResult
from app.services import catalog_maintenance
from app.services.catalog_maintenance import (
    _valid_aggregate_values,
    catalog_refresh_years,
    refresh_recent_catalog,
)
from app.workers.schedules import MAINTENANCE_BEAT_SCHEDULE


@pytest.mark.parametrize(
    ("execution_date", "expected"),
    [
        (date(2026, 7, 1), (2025, 2026)),
        (date(2027, 1, 31), (2026,)),
        (date(2030, 7, 31), (2029, 2030)),
        (date(2031, 1, 1), (2030,)),
    ],
)
def test_catalog_refresh_year_policy(execution_date, expected) -> None:
    assert catalog_refresh_years(execution_date) == expected


def test_catalog_refresh_rejects_non_policy_month() -> None:
    with pytest.raises(ValueError, match="January and July"):
        catalog_refresh_years(date(2030, 6, 30))


def test_aggregate_projection_preserves_malformed_fields() -> None:
    assert _valid_aggregate_values(
        {"avg_rating": "bad", "total_logs": 123, "directors": ["ignored"]}
    ) == {"total_logs": 123}
    assert _valid_aggregate_values({"avg_rating": 4.2, "total_logs": -1}) == {
        "avg_rating": 4.2
    }
    assert _valid_aggregate_values({"avg_rating": None, "total_logs": None}) == {}


def _settings(**overrides: int) -> SimpleNamespace:
    values = {
        "NEW_ELIGIBLE_USERS_THRESHOLD": 100,
        "NEW_MODEL_FILMS_THRESHOLD": 250,
        "MAX_MODEL_AGE_DAYS": 180,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _statistics(users: int, interactions: int, films: range) -> TrainingStatistics:
    return TrainingStatistics(
        measured_at=datetime(2030, 1, 1, tzinfo=UTC),
        eligible_user_count=users,
        rated_interaction_count=interactions,
        rated_film_ids=tuple(films),
    )


def _trained(
    users: int = 1_000, films: range = range(1, 501)
) -> TrainedModelStatistics:
    return TrainedModelStatistics(
        trained_at=datetime(2029, 12, 1, tzinfo=UTC),
        eligible_user_count=users,
        rated_interaction_count=10_000,
        model_film_count=len(films),
        model_film_ids=tuple(films),
    )


@pytest.mark.parametrize(
    ("user_delta", "film_delta", "expected"),
    [
        (99, 249, set()),
        (100, 249, {"NEW_USERS_THRESHOLD"}),
        (101, 249, {"NEW_USERS_THRESHOLD"}),
        (99, 250, {"NEW_FILMS_THRESHOLD"}),
        (99, 251, {"NEW_FILMS_THRESHOLD"}),
        (100, 250, {"NEW_USERS_THRESHOLD", "NEW_FILMS_THRESHOLD"}),
    ],
)
def test_retraining_threshold_boundaries(user_delta, film_delta, expected) -> None:
    trained = _trained()
    current = _statistics(
        1_000 + user_delta,
        10_100,
        range(1, 501 + film_delta),
    )
    decision = decide_retraining(
        current,
        trained,
        now=datetime(2030, 1, 1, tzinfo=UTC),
        settings=_settings(),
    )
    actual = {reason.value for reason in decision.reasons if reason.value != "NONE"}
    assert actual == expected
    assert decision.should_retrain is bool(expected)


def test_retraining_age_force_and_rated_film_set_semantics() -> None:
    trained = _trained()
    aged = TrainedModelStatistics(
        trained_at=datetime(2029, 1, 1, tzinfo=UTC),
        eligible_user_count=trained.eligible_user_count,
        rated_interaction_count=trained.rated_interaction_count,
        model_film_count=trained.model_film_count,
        model_film_ids=trained.model_film_ids,
    )
    current = _statistics(1_000, 10_000, range(1, 502))
    age_decision = decide_retraining(
        current,
        aged,
        now=aged.trained_at + timedelta(days=180),
        settings=_settings(),
    )
    assert [reason.value for reason in age_decision.reasons] == ["MODEL_AGE_THRESHOLD"]
    assert age_decision.deltas.new_model_films == 1
    forced = decide_retraining(current, trained, force=True, settings=_settings())
    assert forced.reasons[0].value == "FORCED"


def test_legacy_flat_requires_explicit_bootstrap_then_normal_thresholds_apply() -> None:
    current = _statistics(1_000, 10_000, range(1, 501))
    legacy = TrainedModelStatistics(
        trained_at=datetime(2029, 12, 1, tzinfo=UTC),
        eligible_user_count=-1,
        rated_interaction_count=-1,
        model_film_count=500,
        model_film_ids=tuple(range(1, 501)),
    )

    bootstrap = decide_retraining(current, legacy, settings=_settings())
    assert bootstrap.should_retrain
    assert [reason.value for reason in bootstrap.reasons] == ["LEGACY_MODEL_BOOTSTRAP"]

    versioned = _trained(users=1_000, films=range(1, 501))
    normal = decide_retraining(
        current,
        versioned,
        now=datetime(2030, 1, 1, tzinfo=UTC),
        settings=_settings(),
    )
    assert not normal.should_retrain
    assert [reason.value for reason in normal.reasons] == ["NONE"]


def test_pending_query_is_bounded_and_deterministically_ordered() -> None:
    scalar_rows = SimpleNamespace(all=lambda: ["a", "b"])
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalars=lambda: scalar_rows))
    )

    class Context:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            return None

    result = asyncio.run(sync_queue._get_pending_slugs(lambda: Context(), batch_size=2))
    statement = str(session.execute.await_args.args[0])
    assert result == ["a", "b"]
    assert "ORDER BY films_queue.created_at, films_queue.id" in statement
    assert "LIMIT" in statement


def test_transient_queue_batch_failure_leaves_rows_for_retry(monkeypatch) -> None:
    monkeypatch.setattr(
        sync_queue,
        "_get_pending_slugs",
        AsyncMock(return_value=["a", "b"]),
    )
    monkeypatch.setattr(
        sync_queue.asyncio,
        "to_thread",
        AsyncMock(side_effect=ConnectionError("temporary")),
    )
    mark_terminal = AsyncMock()
    monkeypatch.setattr(sync_queue, "_mark_terminal", mark_terminal)
    with pytest.raises(ConnectionError, match="temporary"):
        asyncio.run(sync_queue.sync_film_queue(session_factory=object()))
    mark_terminal.assert_not_awaited()


def test_catalog_refresh_updates_only_valid_aggregates_and_continues_failures(
    monkeypatch,
) -> None:
    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(catalog_maintenance.asyncio, "to_thread", run_inline)
    query_session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                all=lambda: [(1, "good"), (2, "partial"), (3, "failed")]
            )
        )
    )
    update_sessions: list[SimpleNamespace] = []

    class Transaction:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *args):
            return None

    class Context:
        def __init__(self, session):
            self.session = session

        async def __aenter__(self):
            return self.session

        async def __aexit__(self, *args):
            return None

    def factory():
        if not update_sessions:
            update_sessions.append(SimpleNamespace(marker="query consumed"))
            return Context(query_session)
        session = SimpleNamespace(execute=AsyncMock())
        session.begin = lambda: Transaction()
        update_sessions.append(session)
        return Context(session)

    def scraper(slugs):
        assert slugs == ["good", "partial", "failed"]
        return [
            FilmScrapeResult(
                "good",
                FilmScrapeOutcome.SUCCESS,
                {"avg_rating": 4.1, "total_logs": 9000, "director": ["ignored"]},
            ),
            FilmScrapeResult(
                "partial",
                FilmScrapeOutcome.SUCCESS,
                {"avg_rating": "bad", "total_logs": 100},
            ),
            FilmScrapeResult("failed", FilmScrapeOutcome.FAILED, error="not found"),
        ]

    result = asyncio.run(
        refresh_recent_catalog(
            date(2026, 7, 15), session_factory=factory, scraper=scraper
        )
    )
    assert result.selected_count == 3
    assert result.updated_count == 2
    assert result.failed_count == 1
    statements = [
        str(session.execute.await_args.args[0]) for session in update_sessions[1:]
    ]
    assert all("UPDATE films SET" in statement for statement in statements)
    assert all("director" not in statement for statement in statements)


def test_catalog_refresh_dry_run_performs_no_scrape_or_write() -> None:
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [(1, "film")]))
    )

    class Context:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            return None

    scraper = pytest.fail
    result = asyncio.run(
        refresh_recent_catalog(
            date(2027, 1, 15),
            session_factory=lambda: Context(),
            scraper=scraper,
            dry_run=True,
        )
    )
    assert result.dry_run
    assert result.selected_count == 1
    assert session.execute.await_count == 1


class _FakeRedis:
    def __init__(self) -> None:
        self.value = None
        self.closed = False

    def set(self, _key, value, *, nx, ex):
        assert nx and ex > 0
        if self.value is not None:
            return False
        self.value = value
        return True

    def eval(self, _script, _numkeys, _key, token):
        if self.value == token:
            self.value = None
            return 1
        return 0

    def close(self):
        self.closed = True


def test_maintenance_lock_prevents_duplicates_and_releases_after_failure() -> None:
    client = _FakeRedis()
    first = MaintenanceLock("redis://unused", key="job", ttl_seconds=30, client=client)
    second = MaintenanceLock("redis://unused", key="job", ttl_seconds=30, client=client)
    with pytest.raises(RuntimeError), first.held() as acquired:
        assert acquired
        assert not second.acquire()
        raise RuntimeError("failure")
    assert client.value is None
    assert second.acquire()
    assert second.release()


def test_beat_schedule_registers_only_expected_utc_maintenance_jobs() -> None:
    assert set(MAINTENANCE_BEAT_SCHEDULE) == {
        "weekly-film-queue",
        "weekly-retraining-evaluation",
        "january-catalog-refresh",
        "july-catalog-refresh",
    }
    assert {entry["task"] for entry in MAINTENANCE_BEAT_SCHEDULE.values()} == {
        "app.tasks.maintenance.process_film_queue",
        "app.tasks.maintenance.evaluate_retraining",
        "app.tasks.maintenance.refresh_recent_catalog",
    }
