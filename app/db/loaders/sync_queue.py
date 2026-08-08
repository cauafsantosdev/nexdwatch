"""Process queued films with isolated, observable outcomes."""

import asyncio
import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal
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
    if value is None:
        return set()
    values: Iterable[Any] = [value] if isinstance(value, str) else value
    return {
        str(name).strip() for name in values if name is not None and str(name).strip()
    }


async def _get_pending_slugs(
    session_factory: async_sessionmaker[AsyncSession],
) -> list[str]:
    async with session_factory() as session:
        queue_result = await session.execute(
            select(FilmQueue.film_slug).where(FilmQueue.status == Status.PENDING)
        )
        return list(queue_result.scalars().all())


async def _get_or_create_relation(
    session: AsyncSession,
    model: RelationModel,
    name: str,
) -> Any:
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
    if status not in {Status.FILTERED, Status.FAILED}:
        raise ValueError(f"status is not terminal: {status}")

    async with session_factory() as session:
        async with session.begin():
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
    pending_result = await session.execute(
        select(LogPending).where(
            LogPending.status == Status.PENDING,
            LogPending.film_slug == film.slug,
        )
    )
    pending_logs = list(pending_result.scalars().all())
    if not pending_logs:
        return

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
    metadata = result.metadata
    if metadata is None:
        raise ValueError("successful scrape result has no metadata")

    async with session_factory() as session:
        async with session.begin():
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
) -> None:
    """Process each pending queue item in an isolated transaction.

    Args:
        session_factory: Async database session factory.
        scraper: Blocking batch film scraper.
    """
    film_slugs = await _get_pending_slugs(session_factory)
    if not film_slugs:
        return

    try:
        scrape_results = await asyncio.to_thread(scraper, film_slugs)
    except Exception as exc:
        logger.exception("Film queue scrape batch failed")
        scrape_results = [
            FilmScrapeResult(
                slug=slug,
                outcome=FilmScrapeOutcome.FAILED,
                error=f"Scrape batch failed: {type(exc).__name__}",
            )
            for slug in film_slugs
        ]

    results_by_slug = {result.slug: result for result in scrape_results}
    for slug in film_slugs:
        result = results_by_slug.get(slug)
        if result is None:
            await _mark_terminal(
                session_factory,
                slug,
                Status.FAILED,
                "Scraper returned no result",
            )
            continue
        if result.outcome == FilmScrapeOutcome.FILTERED:
            await _mark_terminal(session_factory, slug, Status.FILTERED, result.error)
            continue
        if result.outcome == FilmScrapeOutcome.FAILED:
            await _mark_terminal(session_factory, slug, Status.FAILED, result.error)
            continue

        try:
            await _persist_success(session_factory, result)
        except Exception as exc:
            logger.exception("Film persistence failed for film_slug=%s", slug)
            await _mark_terminal(
                session_factory,
                slug,
                Status.FAILED,
                f"Persistence failed: {type(exc).__name__}",
            )


if __name__ == "__main__":
    asyncio.run(sync_film_queue())
