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
    """Coordinate blocking profile scraping and asynchronous persistence."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        scraper: Callable[[str], ScrapedProfile] = scrape_user_profile,
    ) -> None:
        self._session_factory = session_factory
        self._scraper = scraper

    async def sync_profile(self, username: str) -> ProfileSyncResult:
        """Scrape and persist a Letterboxd profile.

        Args:
            username: Public Letterboxd username.

        Returns:
            Identifiers and counts for the synchronized profile.

        Raises:
            EmptyProfileError: If no valid watched films were returned.
        """
        logger.info("Synchronizing profile for username=%s", username)
        profile = await asyncio.to_thread(self._scraper, username)
        if not profile.watches:
            raise EmptyProfileError

        await sync_user_logs(profile, session_factory=self._session_factory)

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
