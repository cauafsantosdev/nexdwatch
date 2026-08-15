"""Resolve queued film slugs and promote their dependent pending interactions.

The scraper runs once per bounded batch, while persistence runs in a separate
transaction per film. A batch-level scrape failure leaves every item pending for a
Celery retry; malformed or failed individual films become terminal independently.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal
from app.domain.maintenance import FilmQueueRunResult
from app.models import (
    Actor,
    Country,
    Director,
    Film,
    FilmQueue,
    Genre,
    Language,
    Log,
    LogPending,
    Status,
    Studio,
    Theme,
    User,
)
from app.scraper import (
    FilmScrapeOutcome,
    FilmScrapeResult,
    scrape_film_queue,
)

logger = logging.getLogger(__name__)

RelationModel = type[Director | Actor | Genre | Language | Country | Studio | Theme]
RELATION_MAP: tuple[tuple[RelationModel, str, str], ...] = (
    (Director, "director", "directors"),
    (Actor, "actors", "actors"),
    (Genre, "genre", "genres"),
    (Language, "language", "languages"),
    (Country, "country", "countries"),
    (Studio, "studio", "studios"),
    (Theme, "themes", "themes"),
)


def _relation_names(value: Any) -> set[str]:
    """Normalize scalar or iterable relationship metadata to unique names."""
    if value is None:
        return set()
    values: Iterable[Any] = [value] if isinstance(value, str) else value
    return {
        str(name).strip() for name in values if name is not None and str(name).strip()
    }


async def _get_pending_slugs(
    session_factory: async_sessionmaker[AsyncSession],
    batch_size: int = 100,
) -> list[str]:
    """Select the oldest pending slugs up to the bounded maintenance batch size."""
    async with session_factory() as session:
        queue_result = await session.execute(
            select(FilmQueue.film_slug)
            .where(FilmQueue.status == Status.PENDING)
            .order_by(FilmQueue.created_at, FilmQueue.id)
            .limit(batch_size)
        )
        return list(queue_result.scalars().all())


async def _get_or_create_relation(
    session: AsyncSession,
    model: RelationModel,
    name: str,
) -> Any:
    """Reuse a named relation or stage a new entity in the caller's transaction."""
    existing = await session.scalar(select(model).where(model.name == name))
    if existing is not None:
        return existing
    relation = model(name=name)
    session.add(relation)
    return relation


async def _mark_terminal(
    session_factory: async_sessionmaker[AsyncSession],
    slug: str,
    status: Status,
    reason: str | None,
) -> None:
    """Atomically finish a queue item and all unresolved dependent logs.

    Args:
        session_factory: Factory used to own the isolated terminal transaction.
        slug: Queue identity shared by the film and pending interactions.
        status: ``FILTERED`` or ``FAILED`` terminal outcome.
        reason: Operational failure/filter explanation retained on the queue row.

    Returns:
        None: Queue and pending-log state is committed directly to PostgreSQL.

    Raises:
        ValueError: If asked to apply a non-terminal status.
    """
    if status not in {Status.FILTERED, Status.FAILED}:
        raise ValueError(f"status is not terminal: {status}")

    async with session_factory() as session:  # noqa: SIM117
        async with session.begin():
            # Queue and dependent rows transition together so clients never observe
            # a terminal film with retryable pending interactions.
            queue_item = await session.scalar(
                select(FilmQueue).where(FilmQueue.film_slug == slug)
            )
            if queue_item is None:
                logger.warning("Queue entry disappeared for film_slug=%s", slug)
                return
            queue_item.attempts += 1
            queue_item.status = status
            queue_item.last_error = reason
            queue_item.updated_at = datetime.now(UTC)

            pending_result = await session.execute(
                select(LogPending).where(
                    LogPending.status == Status.PENDING,
                    LogPending.film_slug == slug,
                )
            )
            for pending_log in pending_result.scalars().all():
                pending_log.status = status


async def _attach_pending_logs(session: AsyncSession, film: Film) -> None:
    """Promote pending username/slug ratings after a film becomes available.

    Existing user/film logs are preserved, missing users make only their pending
    row fail, and every other resolvable row becomes ``PROCESSED`` in the caller's
    transaction.
    """
    pending_result = await session.execute(
        select(LogPending).where(
            LogPending.status == Status.PENDING,
            LogPending.film_slug == film.slug,
        )
    )
    pending_logs = list(pending_result.scalars().all())
    if not pending_logs:
        return

    # Resolve users and existing film interactions in sets before mutating rows;
    # this avoids duplicate Logs when multiple pending records share a username.
    usernames = {pending.username for pending in pending_logs}
    users_result = await session.execute(
        select(User.username, User.id).where(User.username.in_(usernames))
    )
    user_ids = dict(users_result.all())

    valid_user_ids = set(user_ids.values())
    existing_result = await session.execute(
        select(Log.user_id).where(
            Log.film_id == film.id,
            Log.user_id.in_(valid_user_ids),
        )
    )
    existing_user_ids = set(existing_result.scalars().all())

    for pending in pending_logs:
        user_id = user_ids.get(pending.username)
        if user_id is None:
            pending.status = Status.FAILED
            continue
        if user_id not in existing_user_ids:
            session.add(Log(user_id=user_id, film_id=film.id, rating=pending.rating))
            existing_user_ids.add(user_id)
        pending.status = Status.PROCESSED


async def _persist_success(
    session_factory: async_sessionmaker[AsyncSession],
    result: FilmScrapeResult,
) -> None:
    """Persist one successful scrape and unblock its logs atomically.

    The film and normalized relationship entities are created only if the catalog
    does not already contain the slug. Pending logs are then promoted and the queue
    item becomes ``PROCESSED`` in the same isolated transaction.

    Raises:
        ValueError: If a success outcome contains no film metadata.
        LookupError: If the selected queue entry disappeared before persistence.
    """
    metadata = result.metadata
    if metadata is None:
        raise ValueError("successful scrape result has no metadata")

    async with session_factory() as session:  # noqa: SIM117
        async with session.begin():
            # Re-read queue and catalog state inside this film's transaction so one
            # concurrent or malformed item cannot affect unrelated batch members.
            queue_item = await session.scalar(
                select(FilmQueue).where(FilmQueue.film_slug == result.slug)
            )
            if queue_item is None:
                raise LookupError(f"queue entry not found for {result.slug}")

            film = await session.scalar(select(Film).where(Film.slug == result.slug))
            if film is None:
                original_title = metadata.get("original_title") or metadata["title"]
                film = Film(
                    tmdb_id=(
                        int(metadata["tmdb_id"])
                        if metadata.get("tmdb_id") is not None
                        else None
                    ),
                    slug=result.slug.strip(),
                    title=str(metadata["title"]).strip(),
                    original_title=str(original_title).strip(),
                    year=(
                        int(metadata["year"])
                        if metadata.get("year") is not None
                        else None
                    ),
                    runtime=(
                        int(metadata["runtime"])
                        if metadata.get("runtime") is not None
                        else None
                    ),
                    synopsis=metadata.get("synopsis"),
                    tagline=metadata.get("tagline"),
                    avg_rating=(
                        float(metadata["avg_rating"])
                        if metadata.get("avg_rating") is not None
                        else None
                    ),
                    total_logs=int(metadata["total_logs"]),
                )
                session.add(film)

                for model, source_key, relationship_name in RELATION_MAP:
                    for name in _relation_names(metadata.get(source_key)):
                        relation = await _get_or_create_relation(session, model, name)
                        getattr(film, relationship_name).append(relation)

                await session.flush()

            # Only mark the queue processed after the film identity exists and all
            # resolvable pending interactions have been attached.
            await _attach_pending_logs(session, film)
            queue_item.attempts += 1
            queue_item.status = Status.PROCESSED
            queue_item.last_error = None
            queue_item.updated_at = datetime.now(UTC)

    logger.info("Processed queued film film_slug=%s", result.slug)


async def sync_film_queue(
    *,
    session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
    scraper: Callable[[list[str]], list[FilmScrapeResult]] = scrape_film_queue,
    batch_size: int = 100,
) -> FilmQueueRunResult:
    """Scrape a bounded pending batch and persist each outcome independently.

    A failure of the blocking batch scraper is propagated with all selected rows
    still pending. Once results exist, missing/filtered/failed outcomes transition
    only their own queue and pending rows; a successful result gets its own catalog
    transaction so later failures cannot roll it back.

    Args:
        session_factory: Async database session factory.
        scraper: Blocking batch film scraper, executed outside the event loop.
        batch_size: Maximum oldest pending entries selected for this run.

    Returns:
        FilmQueueRunResult: Selected and terminal outcome counts plus wall duration.

    Raises:
        ValueError: If ``batch_size`` is not positive.
        Exception: Propagates a batch scraper failure so Celery can retry without
            terminally changing any selected row.
    """
    if batch_size <= 0:
        raise ValueError("film queue batch size must be positive")
    # Freeze the oldest pending work before crossing the blocking scraper boundary.
    started = time.perf_counter()
    film_slugs = await _get_pending_slugs(session_factory, batch_size=batch_size)
    pending_count = len(film_slugs)
    if not film_slugs:
        return FilmQueueRunResult(0, 0, 0, 0, 0, time.perf_counter() - started)

    try:
        scrape_results = await asyncio.to_thread(scraper, film_slugs)
    except Exception:
        logger.exception("Film queue scrape batch failed")
        # Leave every selected row pending so the maintenance task can retry the
        # transient batch failure without turning the whole batch terminal.
        raise

    # Classify and persist in original queue order. Each helper owns its transaction,
    # preserving already completed films when another film fails persistence.
    results_by_slug = {result.slug: result for result in scrape_results}
    success_count = 0
    filtered_count = 0
    failed_count = 0
    for slug in film_slugs:
        result = results_by_slug.get(slug)
        if result is None:
            await _mark_terminal(
                session_factory,
                slug,
                Status.FAILED,
                "Scraper returned no result",
            )
            failed_count += 1
            continue
        if result.outcome == FilmScrapeOutcome.FILTERED:
            await _mark_terminal(session_factory, slug, Status.FILTERED, result.error)
            filtered_count += 1
            continue
        if result.outcome == FilmScrapeOutcome.FAILED:
            await _mark_terminal(session_factory, slug, Status.FAILED, result.error)
            failed_count += 1
            continue

        try:
            await _persist_success(session_factory, result)
            success_count += 1
        except Exception as exc:
            logger.exception("Film persistence failed for film_slug=%s", slug)
            await _mark_terminal(
                session_factory,
                slug,
                Status.FAILED,
                f"Persistence failed: {type(exc).__name__}",
            )
            failed_count += 1

    result = FilmQueueRunResult(
        pending_count=pending_count,
        processed_count=len(film_slugs),
        success_count=success_count,
        filtered_count=filtered_count,
        failed_count=failed_count,
        duration_seconds=time.perf_counter() - started,
    )
    logger.info(
        "Film queue maintenance pending=%d processed=%d success=%d filtered=%d "
        "failed=%d duration_s=%.3f",
        result.pending_count,
        result.processed_count,
        result.success_count,
        result.filtered_count,
        result.failed_count,
        result.duration_seconds,
    )
    return result


if __name__ == "__main__":
    asyncio.run(sync_film_queue())
