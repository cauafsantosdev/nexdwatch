"""User interaction persistence operations."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Log


@dataclass(frozen=True, slots=True)
class RatedInteraction:
    """Film and explicit rating used by inductive user encoding."""

    film_id: int
    rating: float


@dataclass(frozen=True, slots=True)
class RecommendationHistory:
    """One bounded read of watched IDs and explicit ratings for policy inference."""

    watched_film_ids: tuple[int, ...]
    rated_interactions: tuple[RatedInteraction, ...]


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

    async def get_rated_interactions(self, user_id: int) -> list[RatedInteraction]:
        """Return rated interactions used to build personalized candidate profiles."""
        result = await self._session.execute(
            select(Log.film_id, Log.rating).where(
                Log.user_id == user_id,
                Log.rating.is_not(None),
            )
        )
        return [
            RatedInteraction(film_id=int(film_id), rating=float(rating))
            for film_id, rating in result.all()
        ]

    async def get_recommendation_history(self, user_id: int) -> RecommendationHistory:
        """Load watched and rated history together without duplicate SQL reads."""
        result = await self._session.execute(
            select(Log.film_id, Log.rating).where(Log.user_id == user_id)
        )
        watched: list[int] = []
        rated: list[RatedInteraction] = []
        for film_id, rating in result.all():
            normalized_id = int(film_id)
            watched.append(normalized_id)
            if rating is not None:
                rated.append(RatedInteraction(normalized_id, float(rating)))
        return RecommendationHistory(tuple(watched), tuple(rated))
