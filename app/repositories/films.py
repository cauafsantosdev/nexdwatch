"""Film persistence operations."""

from collections.abc import Collection

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Film


class FilmRepository:
    """Read film metadata required by recommendation inference."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_ids(self, film_ids: Collection[int]) -> list[Film]:
        """Return films for the supplied IDs using a single batch query.

        Args:
            film_ids: Database film identifiers to retrieve.

        Returns:
            Matching film models. Database ordering is not guaranteed.
        """
        if not film_ids:
            return []

        result = await self._session.execute(
            select(Film)
            .options(selectinload(Film.directors))
            .where(Film.id.in_(film_ids))
        )
        return list(result.scalars().all())
