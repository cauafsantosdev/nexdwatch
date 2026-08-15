"""Batched catalog metadata represented as film-aligned arrays and relations."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, text

from app.core.config import Settings, get_settings
from experiments.ranker.config import ENTITY_FAMILIES


@dataclass(frozen=True, slots=True)
class IndexedRelation:
    """CSR-style entity memberships aligned to catalog film rows."""

    indptr: NDArray[np.int64]
    indices: NDArray[np.int32]

    def entities(self, film_row: int) -> NDArray[np.int32]:
        """Return compact entity identifiers associated with one film row."""
        return self.indices[self.indptr[film_row] : self.indptr[film_row + 1]]

    def counts(self) -> NDArray[np.int32]:
        """Return the number of entity memberships for every film row."""
        return np.diff(self.indptr).astype(np.int32, copy=False)


@dataclass(frozen=True, slots=True)
class RankerCatalog:
    """Stable, leakage-independent film metadata for ranker research."""

    film_ids: NDArray[np.int64]
    id_to_row: dict[int, int]
    years: NDArray[np.float32]
    runtimes: NDArray[np.float32]
    relations: dict[str, IndexedRelation]


def load_ranker_catalog(
    film_ids: NDArray[np.int64], settings: Settings | None = None
) -> RankerCatalog:
    """Load catalog scalars and relations in bounded read-only SQL queries."""
    if not len(film_ids):
        raise ValueError("ranker catalog film vocabulary is empty")
    active = settings or get_settings()
    engine = create_engine(
        "postgresql+psycopg2://"
        f"{active.POSTGRES_USER}:{active.POSTGRES_PASSWORD}"
        f"@{active.POSTGRES_HOST}:{active.POSTGRES_PORT}/{active.POSTGRES_DB}"
    )
    id_to_row = {int(film_id): row for row, film_id in enumerate(film_ids)}
    years = np.full(len(film_ids), np.nan, dtype=np.float32)
    runtimes = np.full(len(film_ids), np.nan, dtype=np.float32)
    association_tables = {
        "genre": "film_genres",
        "director": "film_directors",
        "actor": "film_actors",
        "theme": "film_themes",
        "country": "film_countries",
        "language": "film_languages",
        "studio": "film_studios",
    }
    relation_pairs: dict[str, list[tuple[int, int]]] = {
        family: [] for family in ENTITY_FAMILIES
    }
    try:
        with engine.connect() as connection:
            for film_id, year, runtime in connection.execute(
                text("SELECT id, year, runtime FROM films")
            ):
                row = id_to_row.get(int(film_id))
                if row is None:
                    continue
                if year is not None:
                    years[row] = float(year)
                if runtime is not None:
                    runtimes[row] = float(runtime)
            for family, table in association_tables.items():
                query = text(f"SELECT film_id, {family}_id FROM {table}")
                for film_id, entity_id in connection.execute(query):
                    row = id_to_row.get(int(film_id))
                    if row is not None:
                        relation_pairs[family].append((row, int(entity_id)))
    finally:
        engine.dispose()

    decade_values = sorted(
        {int(year // 10 * 10) for year in years if np.isfinite(year)}
    )
    decade_ids = {decade: index for index, decade in enumerate(decade_values)}
    relation_pairs["decade"] = [
        (row, decade_ids[int(year // 10 * 10)])
        for row, year in enumerate(years)
        if np.isfinite(year)
    ]
    relations = {
        family: _build_indexed_relation(len(film_ids), pairs)
        for family, pairs in relation_pairs.items()
    }
    return RankerCatalog(
        film_ids=np.ascontiguousarray(film_ids, dtype=np.int64),
        id_to_row=id_to_row,
        years=years,
        runtimes=runtimes,
        relations=relations,
    )


def _build_indexed_relation(
    film_count: int, pairs: list[tuple[int, int]]
) -> IndexedRelation:
    """Compact arbitrary database IDs into deterministic zero-based IDs."""
    entity_ids = {entity_id for _, entity_id in pairs}
    compact = {entity_id: index for index, entity_id in enumerate(sorted(entity_ids))}
    grouped: list[list[int]] = [[] for _ in range(film_count)]
    for film_row, entity_id in pairs:
        grouped[film_row].append(compact[entity_id])
    indptr = np.zeros(film_count + 1, dtype=np.int64)
    indices: list[int] = []
    for row, values in enumerate(grouped):
        unique = sorted(set(values))
        indices.extend(unique)
        indptr[row + 1] = len(indices)
    return IndexedRelation(
        indptr=indptr,
        indices=np.ascontiguousarray(indices, dtype=np.int32),
    )
