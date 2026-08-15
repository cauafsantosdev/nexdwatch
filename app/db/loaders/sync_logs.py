"""Reconcile scraped profile interactions with the local film catalog.

Known films become ``Log`` rows immediately. Unknown slugs are represented by a
deduplicated ``FilmQueue`` entry and username-scoped ``LogPending`` rows so catalog
acquisition can finish asynchronously without losing ratings.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal
from app.domain.profiles import ScrapedProfile, ScrapedWatch
from app.models import Film, FilmQueue, Log, LogPending, Status, User

logger = logging.getLogger(__name__)


class InconsistentIngestionStateError(RuntimeError):
    """Raised when queue state cannot lead to a valid persisted interaction."""


def _normalize_profile(
    profile_or_logs: ScrapedProfile | Sequence[Mapping[str, Any]],
) -> ScrapedProfile | None:
    """Convert the typed or legacy scraper payload into one usable profile.

    Invalid legacy rows are discarded locally; an absent username or empty input
    yields ``None`` and makes synchronization a no-op.
    """
    if isinstance(profile_or_logs, ScrapedProfile):
        return profile_or_logs
    if not profile_or_logs:
        return None

    username = str(profile_or_logs[0].get("username", "")).strip()
    if not username:
        return None

    watches: list[ScrapedWatch] = []
    for log in profile_or_logs:
        slug = str(log.get("slug", "")).strip()
        if not slug:
            continue
        raw_rating = log.get("rating")
        try:
            rating = float(raw_rating) if raw_rating is not None else None
        except (TypeError, ValueError):
            rating = None
        watches.append(ScrapedWatch(film_slug=slug, rating=rating))

    return ScrapedProfile(username=username, watches=tuple(watches))


async def sync_user_logs(
    profile_or_logs: ScrapedProfile | Sequence[Mapping[str, Any]],
    *,
    session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
) -> None:
    """Reconcile one complete scraped profile in a single transaction.

    Existing ratings are updated in place and absent known-film ratings are
    inserted. Unknown films reuse or create queue state and preserve their rating in
    ``LogPending``. Repeated watches within one payload collapse by film identity,
    making replay idempotent with respect to the persisted user/film pair.

    Args:
        profile_or_logs: Typed scraped profile or legacy list of log mappings.
        session_factory: Async database session factory.

    Returns:
        None: User, log, queue, and pending-log changes commit atomically.

    Raises:
        InconsistentIngestionStateError: If queue state claims a film was processed
            although no catalog film exists, or contains an unsupported status.
        Exception: Propagates database failures after rolling back the profile sync.
    """
    # Normalize legacy scraper output before allocating a database transaction.
    profile = _normalize_profile(profile_or_logs)
    if profile is None:
        return

    async with session_factory() as session:
        async with session.begin():
            # Load or create the user, then read each reconciliation universe once
            # so known and unknown paths share one transaction snapshot.
            result_user = await session.execute(
                select(User).where(User.username == profile.username)
            )
            user = result_user.scalar_one_or_none()
            if user is None:
                user = User(username=profile.username)
                session.add(user)
                await session.flush()

            result_films = await session.execute(select(Film.slug, Film.id))
            film_ids_by_slug = dict(result_films.all())

            result_existing_logs = await session.execute(
                select(Log).where(Log.user_id == user.id)
            )
            existing_logs_by_film_id = {
                log.film_id: log for log in result_existing_logs.scalars().all()
            }

            result_film_queue = await session.execute(select(FilmQueue))
            queue_status_by_slug = {
                queue_item.film_slug: queue_item.status
                for queue_item in result_film_queue.scalars().all()
            }

            result_pending = await session.execute(
                select(LogPending).where(LogPending.username == profile.username)
            )
            pending_by_slug: dict[str, list[LogPending]] = {}
            for pending_log in result_pending.scalars().all():
                pending_by_slug.setdefault(pending_log.film_slug, []).append(
                    pending_log
                )

            logs_to_insert: dict[int, dict[str, Any]] = {}
            pending_to_insert: dict[str, dict[str, Any]] = {}
            films_to_queue: dict[str, dict[str, str]] = {}
            updated_logs = 0
            updated_pending = 0

            for watch in profile.watches:
                slug = watch.film_slug.strip()
                if not slug:
                    continue

                film_id = film_ids_by_slug.get(slug)
                if film_id is not None:
                    # Known catalog films are authoritative: update/insert the Log
                    # and retire any stale pending representation for the same slug.
                    existing_log = existing_logs_by_film_id.get(film_id)
                    if existing_log is not None:
                        if existing_log.rating != watch.rating:
                            existing_log.rating = watch.rating
                            updated_logs += 1
                    else:
                        logs_to_insert[film_id] = {
                            "user_id": user.id,
                            "film_id": film_id,
                            "rating": watch.rating,
                        }

                    for pending_log in pending_by_slug.get(slug, []):
                        if pending_log.rating != watch.rating:
                            pending_log.rating = watch.rating
                            updated_pending += 1

                        if pending_log.status == Status.PENDING:
                            pending_log.status = Status.PROCESSED
                    continue

                queue_status = queue_status_by_slug.get(slug)
                # Unknown films retain their interaction through queue-backed
                # pending state until catalog ingestion reaches a terminal outcome.
                if queue_status is None:
                    films_to_queue[slug] = {"film_slug": slug}
                    queue_status = Status.PENDING
                    queue_status_by_slug[slug] = queue_status
                elif queue_status == Status.PROCESSED:
                    logger.error(
                        "Processed queue entry has no catalog film film_slug=%s", slug
                    )
                    raise InconsistentIngestionStateError(
                        f"processed queue entry has no catalog film for {slug}"
                    )

                if queue_status not in {
                    Status.PENDING,
                    Status.FILTERED,
                    Status.FAILED,
                }:
                    raise InconsistentIngestionStateError(
                        f"unsupported queue status for {slug}: {queue_status}"
                    )

                existing_pending = pending_by_slug.get(slug, [])
                if existing_pending:
                    for pending_log in existing_pending:
                        if pending_log.rating != watch.rating:
                            pending_log.rating = watch.rating
                            updated_pending += 1
                        if pending_log.status == Status.PENDING and queue_status in {
                            Status.FILTERED,
                            Status.FAILED,
                        }:
                            pending_log.status = queue_status
                    continue

                pending_payload = pending_to_insert.get(slug)
                if pending_payload is None:
                    pending_to_insert[slug] = {
                        "username": profile.username,
                        "film_slug": slug,
                        "rating": watch.rating,
                        "status": queue_status,
                    }
                else:
                    pending_payload["rating"] = watch.rating

            # Bulk writes occur only after the complete payload is reconciled, so a
            # detected state inconsistency cannot leave a partial profile update.
            if films_to_queue:
                await session.execute(insert(FilmQueue), list(films_to_queue.values()))
            if pending_to_insert:
                await session.execute(
                    insert(LogPending), list(pending_to_insert.values())
                )
            if logs_to_insert:
                await session.execute(insert(Log), list(logs_to_insert.values()))

            logger.info(
                "Profile sync username=%s inserted logs=%d pending=%d queued=%d "
                "updated_logs=%d updated_pending=%d",
                profile.username,
                len(logs_to_insert),
                len(pending_to_insert),
                len(films_to_queue),
                updated_logs,
                updated_pending,
            )
