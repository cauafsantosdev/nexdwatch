"""Scheduled aggregate-only refreshes for recent existing catalog films."""

import asyncio
import logging
import math
import time
from collections.abc import Callable
from datetime import date
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal
from app.domain.maintenance import CatalogRefreshResult
from app.models import Film
from app.scraper import FilmScrapeOutcome, FilmScrapeResult, scrape_film_queue

logger = logging.getLogger(__name__)


def catalog_refresh_years(execution_date: date) -> tuple[int, ...]:
    """Return the frozen January/July target years for a UTC execution date.

    Raises:
        ValueError: If invoked outside the two scheduled refresh months.
    """
    if execution_date.month == 1:
        return (execution_date.year - 1,)
    if execution_date.month == 7:
        return (execution_date.year - 1, execution_date.year)
    raise ValueError("catalog refresh is only defined for January and July")


def _valid_aggregate_values(metadata: dict[str, Any]) -> dict[str, float | int]:
    """Extract only finite rating and exact non-negative log-count updates.

    Invalid fields are omitted independently, allowing a valid aggregate to refresh
    without trusting unrelated scraped metadata.
    """
    values: dict[str, float | int] = {}
    raw_rating = metadata.get("avg_rating")
    if raw_rating is not None and not isinstance(raw_rating, bool):
        try:
            rating = float(raw_rating)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if math.isfinite(rating) and 0 <= rating <= 5:
                values["avg_rating"] = rating
    raw_logs = metadata.get("total_logs")
    if raw_logs is not None and not isinstance(raw_logs, bool):
        try:
            logs = int(raw_logs)
        except (TypeError, ValueError, OverflowError):
            pass
        else:
            if logs >= 0 and raw_logs == logs:
                values["total_logs"] = logs
    return values


async def refresh_recent_catalog(
    execution_date: date,
    *,
    session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
    scraper: Callable[[list[str]], list[FilmScrapeResult]] = scrape_film_queue,
    dry_run: bool = False,
) -> CatalogRefreshResult:
    """Refresh only aggregate fields for existing films in scheduled target years.

    Selection is read once in film-ID order and the blocking scraper runs as one
    batch outside the event loop. Batch failure propagates for Celery retry; after a
    response, malformed/scrape-failed films are isolated and each valid film commits
    in its own transaction. No title, relationships, or identity fields are updated.

    Args:
        execution_date: UTC schedule date defining January/July target years.
        session_factory: Async database transaction factory.
        scraper: Blocking batch scraper executed in a worker thread.
        dry_run: Measure selected rows without scraping or writing.

    Returns:
        CatalogRefreshResult: Target years, selected/updated/failed counts, dry-run
            marker, and total duration.

    Raises:
        ValueError: If the execution month is outside the schedule policy.
        Exception: Propagates selection or whole-batch scraper failure for task retry.
    """
    # Freeze the target identity set before network work; dry runs stop after this
    # bounded database read and never invoke the scraper.
    started = time.perf_counter()
    years = catalog_refresh_years(execution_date)
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(Film.id, Film.slug).where(Film.year.in_(years)).order_by(Film.id)
            )
        ).all()
    selected_count = len(rows)
    if dry_run or not rows:
        result = CatalogRefreshResult(
            years,
            selected_count,
            0,
            0,
            dry_run,
            time.perf_counter() - started,
        )
        _log_refresh(result)
        return result

    # Treat a whole scraper exception as retryable batch failure. Returned per-film
    # outcomes are handled independently below.
    id_by_slug = {slug: film_id for film_id, slug in rows}
    scrape_results = await asyncio.to_thread(scraper, list(id_by_slug))
    results_by_slug = {result.slug: result for result in scrape_results}
    updated = 0
    failed = 0
    for slug, film_id in id_by_slug.items():
        scrape_result = results_by_slug.get(slug)
        if (
            scrape_result is None
            or scrape_result.outcome != FilmScrapeOutcome.SUCCESS
            or scrape_result.metadata is None
        ):
            failed += 1
            logger.warning("Catalog aggregate refresh failed film_slug=%s", slug)
            continue
        values = _valid_aggregate_values(scrape_result.metadata)
        if not values:
            failed += 1
            logger.warning("Catalog refresh aggregates malformed film_slug=%s", slug)
            continue
        # Commit aggregates per film so one persistence failure cannot roll back
        # successful refreshes for unrelated catalog entries.
        try:
            async with session_factory() as session, session.begin():
                await session.execute(
                    update(Film).where(Film.id == film_id).values(**values)
                )
            updated += 1
        except Exception:
            failed += 1
            logger.exception("Catalog aggregate persistence failed film_slug=%s", slug)
    result = CatalogRefreshResult(
        years,
        selected_count,
        updated,
        failed,
        False,
        time.perf_counter() - started,
    )
    _log_refresh(result)
    return result


def _log_refresh(result: CatalogRefreshResult) -> None:
    logger.info(
        "Catalog refresh target_years=%s selected=%d updated=%d failed=%d "
        "dry_run=%s duration_s=%.3f",
        result.target_years,
        result.selected_count,
        result.updated_count,
        result.failed_count,
        result.dry_run,
        result.duration_seconds,
    )
