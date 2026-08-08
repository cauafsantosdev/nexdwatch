"""User interaction persistence operations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Log


class InteractionRepository:
    """Read interactions used by the current SVD baseline."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_rated_film_ids(self, user_id: int) -> list[int]:
        """Return rated film IDs for current mean-pooling inference.

        Unrated watches are intentionally excluded because the existing SVD
        baseline was trained and inferred only from explicit ratings.

        Args:
            user_id: Database identifier for the profile.

        Returns:
            Film IDs for interactions with non-null ratings.
        """
        result = await self._session.execute(
            select(Log.film_id).where(
                Log.user_id == user_id,
                Log.rating.is_not(None),
            )
        )
        return list(result.scalars().all())

    async def get_watched_film_ids(self, user_id: int) -> list[int]:
        """Return every persisted watched film ID for a user.

        Args:
            user_id: Database identifier for the profile.

        Returns:
            Film IDs for rated and unrated interactions.
        """
        result = await self._session.execute(
            select(Log.film_id).where(Log.user_id == user_id)
        )
        return list(result.scalars().all())
