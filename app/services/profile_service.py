"""Profile scraping and persistence orchestration."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal
from app.db.loaders.sync_logs import sync_user_logs
from app.domain.profiles import ScrapedProfile
from app.repositories.users import UserRepository
from app.scraper.user_scraping import scrape_user_profile

logger = logging.getLogger(__name__)


class EmptyProfileError(Exception):
    """Raised when a profile cannot supply any valid watched films."""


@dataclass(frozen=True, slots=True)
class ProfileSyncResult:
    """Summary returned after profile synchronization."""

    user_id: int | None
    logs_count: int


class ProfileService:
    """Coordinate request-independent scraping and transactional profile ingestion.

    Instances own no external resources themselves. A caller-provided async session
    factory defines PostgreSQL transactions, while the blocking scraper is moved to
    a worker thread so the Celery worker's event loop remains responsive.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        scraper: Callable[[str], ScrapedProfile] = scrape_user_profile,
    ) -> None:
        self._session_factory = session_factory
        self._scraper = scraper

    async def sync_profile(self, username: str) -> ProfileSyncResult:
        """Scrape, validate, and transactionally reconcile a Letterboxd profile.

        The complete provider response is obtained before persistence. A valid empty
        profile is rejected so it cannot masquerade as a successful destructive
        synchronization; known and unknown films are then delegated to the shared
        ``Log``/``FilmQueue``/``LogPending`` ingestion workflow.

        Args:
            username: Public Letterboxd username.

        Returns:
            ProfileSyncResult: Persisted user identity, when resolvable, and the
                number of valid watches supplied by Letterboxd.

        Raises:
            EmptyProfileError: If no valid watched films were returned.
            ProfileScrapeError: Propagates safe scraper-boundary failures for worker
                retry classification.
            Exception: Propagates database failures after transactional rollback.
        """
        logger.info("Synchronizing profile for username=%s", username)
        # The provider client is blocking; keep it outside the persistent worker
        # event loop and do not open a database transaction during network I/O.
        profile = await asyncio.to_thread(self._scraper, username)
        if not profile.watches:
            raise EmptyProfileError

        # Persist the complete profile through the same known/unknown-film contract
        # used by other live ingestion paths.
        await sync_user_logs(profile, session_factory=self._session_factory)

        # Re-read the committed user identity for the task's public result rather
        # than depending on ORM state owned inside the ingestion transaction.
        async with self._session_factory() as session:
            user = await UserRepository(session).get_by_username(profile.username)

        return ProfileSyncResult(
            user_id=user.id if user is not None else None,
            logs_count=len(profile.watches),
        )


_profile_service = ProfileService()


def get_profile_service() -> ProfileService:
    """Return the process-wide profile service."""
    return _profile_service
