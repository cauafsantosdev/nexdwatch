"""Bounded in-memory metadata snapshot for categorized recommendation policy."""

from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.categorized_recommendations import EntityFamily
from app.models import Country, Director, Film, Genre, Language
from app.models.relationships import (
    film_countries,
    film_directors,
    film_genres,
    film_languages,
)

StoredFamily = Literal["director", "genre", "country", "language"]


@dataclass(frozen=True, slots=True)
class PolicyEntity:
    id: int
    name: str


@dataclass(frozen=True, slots=True)
class PolicyFilm:
    film_id: int
    title: str
    year: int | None
    directors: tuple[PolicyEntity, ...] = ()
    genres: tuple[PolicyEntity, ...] = ()
    countries: tuple[PolicyEntity, ...] = ()
    languages: tuple[PolicyEntity, ...] = ()

    @property
    def decade(self) -> int | None:
        return self.year // 10 * 10 if self.year is not None else None

    def entities(self, family: EntityFamily) -> tuple[PolicyEntity, ...]:
        if family == "director":
            return self.directors
        if family == "genre":
            return self.genres
        if family == "country":
            return self.countries
        if family == "language":
            return self.languages
        if self.decade is None:
            return ()
        return (PolicyEntity(self.decade, f"{self.decade}s"),)


@dataclass(frozen=True, slots=True)
class PolicyCatalog:
    """Policy metadata keyed by the active recommendation artifact vocabulary."""

    films: dict[int, PolicyFilm]
    artifact_film_ids: frozenset[int]

    def film(self, film_id: int) -> PolicyFilm | None:
        return self.films.get(film_id)


async def load_policy_catalog(
    session: AsyncSession,
    artifact_film_ids: tuple[int, ...] | frozenset[int],
) -> PolicyCatalog:
    """Load scalar data and four relation families in five bounded queries."""
    allowed = frozenset(int(film_id) for film_id in artifact_film_ids)
    scalar_result = await session.execute(select(Film.id, Film.title, Film.year))
    scalars = {
        int(film_id): (str(title), int(year) if year is not None else None)
        for film_id, title, year in scalar_result
        if int(film_id) in allowed
    }
    relation_specs = {
        "director": (film_directors, Director, film_directors.c.director_id),
        "genre": (film_genres, Genre, film_genres.c.genre_id),
        "country": (film_countries, Country, film_countries.c.country_id),
        "language": (film_languages, Language, film_languages.c.language_id),
    }
    memberships: dict[str, dict[int, list[PolicyEntity]]] = {
        family: {} for family in relation_specs
    }
    entity_pools: dict[str, dict[int, PolicyEntity]] = {
        family: {} for family in relation_specs
    }
    for family, (association, model, entity_column) in relation_specs.items():
        result = await session.execute(
            select(association.c.film_id, model.id, model.name).join(
                model, entity_column == model.id
            )
        )
        grouped = memberships[family]
        for film_id, entity_id, name in result:
            normalized_film_id = int(film_id)
            if normalized_film_id in allowed:
                normalized_entity_id = int(entity_id)
                entity = entity_pools[family].get(normalized_entity_id)
                if entity is None:
                    entity = PolicyEntity(normalized_entity_id, str(name))
                    entity_pools[family][normalized_entity_id] = entity
                grouped.setdefault(normalized_film_id, []).append(entity)

    films = {
        film_id: PolicyFilm(
            film_id=film_id,
            title=title,
            year=year,
            directors=_ordered_unique(memberships["director"].get(film_id, [])),
            genres=_ordered_unique(memberships["genre"].get(film_id, [])),
            countries=_ordered_unique(memberships["country"].get(film_id, [])),
            languages=_ordered_unique(memberships["language"].get(film_id, [])),
        )
        for film_id, (title, year) in scalars.items()
    }
    return PolicyCatalog(films=films, artifact_film_ids=allowed)


def _ordered_unique(values: list[PolicyEntity]) -> tuple[PolicyEntity, ...]:
    return tuple(
        sorted(
            {value.id: value for value in values}.values(), key=lambda value: value.id
        )
    )
