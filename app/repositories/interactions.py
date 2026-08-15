"""User interaction persistence operations."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Log, User


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
    """Read watched and explicitly rated interaction universes for inference.

    The repository owns no transaction and never commits; callers provide a session
    whose snapshot is shared across the required recommendation orchestration.
    """

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
        """Return explicit ratings used to build personalized candidate profiles.

        Unrated watches are excluded, values are normalized to primitive numeric
        domain records, and database row order is preserved without a ranking claim.
        """
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
        """Load watched exclusion and explicit ratings from one SQL result.

        The method does not distinguish an unknown user from an existing empty user;
        callers needing that boundary use ``get_existing_user_recommendation_history``.
        """
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

    async def get_existing_user_recommendation_history(
        self, user_id: int
    ) -> RecommendationHistory | None:
        """Load one user's history while distinguishing absence from empty history.

        A user-to-log outer join returns ``None`` only when the user row does not
        exist. Existing users with zero interactions receive empty watched/rated
        tuples, allowing the API to preserve correct 404 semantics.
        """
        result = await self._session.execute(
            select(User.id, Log.film_id, Log.rating)
            .outerjoin(Log, Log.user_id == User.id)
            .where(User.id == user_id)
        )
        rows = result.all()
        if not rows:
            return None

        watched: list[int] = []
        rated: list[RatedInteraction] = []
        for _, film_id, rating in rows:
            if film_id is None:
                continue
            normalized_id = int(film_id)
            watched.append(normalized_id)
            if rating is not None:
                rated.append(RatedInteraction(normalized_id, float(rating)))
        return RecommendationHistory(tuple(watched), tuple(rated))
