"""Orchestrate offline Letterboxd export ingestion."""

from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.database import SessionLocal
from app.db.loaders.sync_logs import sync_user_logs
from app.domain.profiles import ScrapedProfile, ScrapedWatch
from app.importers.letterboxd_export import (
    LetterboxdExportEntry,
    parse_letterboxd_export,
)
from app.repositories.films import CatalogFilm, FilmRepository
from app.repositories.users import UserRepository


class NoResolvedFilmsError(ValueError):
    """Raised when an otherwise valid export has no catalog matches."""


@dataclass(frozen=True, slots=True)
class UnresolvedExportFilm:
    """An export row that could not be mapped deterministically."""

    name: str
    year: int | None
    uri: str
    reason: str


@dataclass(frozen=True, slots=True)
class LetterboxdImportResult:
    """Persistence and resolution summary for one export."""

    user_id: int | None
    watched_in_export: int
    rated_in_export: int
    imported: int
    unresolved_films: tuple[UnresolvedExportFilm, ...]

    @property
    def unresolved(self) -> int:
        """Return the number of unresolved export entries."""
        return len(self.unresolved_films)


SyncProfile = Callable[..., Awaitable[None]]


class LetterboxdImportService:
    """Resolve an uploaded export against the current film catalog."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
        syncer: SyncProfile = sync_user_logs,
    ) -> None:
        self._session_factory = session_factory
        self._syncer = syncer

    async def import_export(
        self, username: str, archive: bytes
    ) -> LetterboxdImportResult:
        """Parse, catalog-resolve, and persist an official export ZIP."""
        export = parse_letterboxd_export(archive)
        years = {entry.year for entry in export.entries}
        async with self._session_factory() as session:
            films = await FilmRepository(session).get_catalog_by_years(years)

        profile, unresolved = resolve_export_profile(
            username=username,
            entries=export.entries,
            catalog=films,
        )
        if not profile.watches:
            raise NoResolvedFilmsError(
                "No films in this export could be resolved against the catalog."
            )

        await self._syncer(profile, session_factory=self._session_factory)

        async with self._session_factory() as session:
            user = await UserRepository(session).get_by_username(profile.username)

        return LetterboxdImportResult(
            user_id=user.id if user is not None else None,
            watched_in_export=export.watched_count,
            rated_in_export=export.rated_count,
            imported=len(profile.watches),
            unresolved_films=unresolved,
        )


def resolve_export_profile(
    *,
    username: str,
    entries: Sequence[LetterboxdExportEntry],
    catalog: Sequence[CatalogFilm],
) -> tuple[ScrapedProfile, tuple[UnresolvedExportFilm, ...]]:
    """Resolve entries by normalized title/year with explicit ambiguity."""
    title_index = _build_index(catalog, original=False)
    original_title_index = _build_index(catalog, original=True)
    watches: list[ScrapedWatch] = []
    unresolved: list[UnresolvedExportFilm] = []

    for entry in entries:
        key = (_normalize_title(entry.name), entry.year)
        primary = title_index.get(key, {})
        candidates = primary or original_title_index.get(key, {})
        if len(candidates) == 1:
            film = next(iter(candidates.values()))
            watches.append(ScrapedWatch(film_slug=film.slug, rating=entry.rating))
            continue

        reason = "ambiguous" if candidates else "not_found"
        unresolved.append(
            UnresolvedExportFilm(
                name=entry.name,
                year=entry.year,
                uri=entry.uri,
                reason=reason,
            )
        )

    return (
        ScrapedProfile(username=username.strip(), watches=tuple(watches)),
        tuple(unresolved),
    )


def _build_index(
    catalog: Sequence[CatalogFilm], *, original: bool
) -> dict[tuple[str, int | None], dict[int, CatalogFilm]]:
    index: defaultdict[tuple[str, int | None], dict[int, CatalogFilm]] = defaultdict(
        dict
    )
    for film in catalog:
        title = film.original_title if original else film.title
        if title:
            index[(_normalize_title(title), film.year)][film.id] = film
    return dict(index)


def _normalize_title(title: str) -> str:
    return " ".join(title.split()).casefold()


_letterboxd_import_service = LetterboxdImportService()


def get_letterboxd_import_service() -> LetterboxdImportService:
    """Return the process-wide offline import service."""
    return _letterboxd_import_service
