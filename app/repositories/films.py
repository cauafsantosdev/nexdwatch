"""Film persistence operations."""

from collections.abc import Collection
from dataclasses import dataclass

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Film


@dataclass(frozen=True, slots=True)
class CatalogFilm:
    """Minimal catalog identity needed by offline export resolution."""

    id: int
    slug: str
    title: str
    original_title: str | None
    year: int | None


class FilmRepository:
    """Read film metadata for recommendation display and export resolution.

    The repository is read-only and owns no transaction. Callers retain ordering
    responsibility because SQL ``IN`` predicates do not preserve requested identity
    order.
    """

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

    async def get_catalog_by_years(
        self, years: Collection[int | None]
    ) -> list[CatalogFilm]:
        """Return lightweight catalog candidates in one query.

        Year filtering keeps the candidate set bounded while title matching is
        completed in Python so internal whitespace can be normalized exactly.

        Args:
            years: Concrete release years and/or ``None`` for unknown-year rows.

        Returns:
            list[CatalogFilm]: Lightweight candidates in unspecified database order;
                no title disambiguation has yet been applied.
        """
        if not years:
            return []

        concrete_years = {year for year in years if year is not None}
        conditions = []
        if concrete_years:
            conditions.append(Film.year.in_(concrete_years))
        if None in years:
            conditions.append(Film.year.is_(None))

        result = await self._session.execute(
            select(
                Film.id,
                Film.slug,
                Film.title,
                Film.original_title,
                Film.year,
            ).where(or_(*conditions))
        )
        return [CatalogFilm(*row) for row in result.all()]
