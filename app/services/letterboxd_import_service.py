"""Orchestrate safe offline Letterboxd export resolution and ingestion.

Official ZIP rows are matched against the existing catalog without network access.
Only unique normalized title/year matches enter the shared profile persistence path;
ambiguous and missing films remain explicit in the import result.
"""

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
    """Coordinate request-scoped archive parsing, catalog reads, and persistence.

    The service owns no persistent resources. Its session factory scopes independent
    catalog/user reads, while the injected syncer owns the transactional known/unknown
    film reconciliation used by live profile ingestion.
    """

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
        """Parse, catalog-resolve, and persist one official export ZIP.

        Candidate catalog rows are bounded by the years present in the export. Title
        resolution completes in memory, and at least one unambiguous film is required
        before the shared profile sync writes any state.

        Returns:
            LetterboxdImportResult: Persisted user/counts plus every unresolved film.

        Raises:
            LetterboxdExportError: If archive safety or CSV validation fails.
            NoResolvedFilmsError: If no entry maps uniquely to the current catalog.
            Exception: Propagates database failures after transaction rollback.
        """
        # Parse the bounded archive first, then issue one year-filtered catalog query.
        export = parse_letterboxd_export(archive)
        years = {entry.year for entry in export.entries}
        async with self._session_factory() as session:
            films = await FilmRepository(session).get_catalog_by_years(years)

        # Resolve before persistence so ambiguity is explicit and cannot select a film
        # by incidental database ordering.
        profile, unresolved = resolve_export_profile(
            username=username,
            entries=export.entries,
            catalog=films,
        )
        if not profile.watches:
            raise NoResolvedFilmsError(
                "No films in this export could be resolved against the catalog."
            )

        # Delegate to the same idempotent Log/FilmQueue/LogPending transaction used by
        # online scraping, then re-read the committed public user identity.
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
    """Resolve export entries by normalized title/year with explicit ambiguity.

    Display titles take precedence; original titles are considered only when the
    display-title key has no candidate. Exactly one film ID is required, and URI is
    diagnostic only because short Letterboxd URIs do not encode the local slug.

    Returns:
        A typed profile of uniquely resolved watches and ordered unresolved records.
    """
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
    """Index catalog candidates by normalized title/year without hiding ambiguity."""
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
