"""Tests for catalog-only Letterboxd export ingestion."""

import asyncio
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.importers.letterboxd_export import (
    LetterboxdExportEntry,
    LetterboxdExportError,
    parse_letterboxd_export,
)
from app.repositories.films import CatalogFilm
from app.services import letterboxd_import_service as import_service_module
from app.services.letterboxd_import_service import (
    LetterboxdImportResult,
    LetterboxdImportService,
    NoResolvedFilmsError,
    UnresolvedExportFilm,
    resolve_export_profile,
)

WATCHED_HEADER = "Date,Name,Year,Letterboxd URI\n"
RATINGS_HEADER = "Date,Name,Year,Letterboxd URI,Rating\n"


def _export_zip(
    watched: str | bytes | None,
    ratings: str | bytes | None = None,
    *,
    watched_name: str = "watched.csv",
) -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as export_zip:
        if watched is not None:
            export_zip.writestr(watched_name, watched)
        if ratings is not None:
            export_zip.writestr("nested/ratings.csv", ratings)
    return archive.getvalue()


def _standard_export() -> bytes:
    watched = WATCHED_HEADER + (
        "2023-02-17,Taxi Driver,1976,https://boxd.it/2b8y\n"
        "2023-02-18,Stalker,1979,https://boxd.it/28PO\n"
    )
    ratings = RATINGS_HEADER + ("2023-02-17,Taxi Driver,1976,https://boxd.it/2b8y,5\n")
    return _export_zip(watched, ratings)


def test_parser_merges_ratings_by_uri_and_retains_unrated_watches() -> None:
    export = parse_letterboxd_export(_standard_export())

    assert export.entries == (
        LetterboxdExportEntry(
            uri="https://boxd.it/2b8y",
            name="Taxi Driver",
            year=1976,
            rating=5.0,
        ),
        LetterboxdExportEntry(
            uri="https://boxd.it/28PO",
            name="Stalker",
            year=1979,
            rating=None,
        ),
    )
    assert export.watched_count == 2
    assert export.rated_count == 1


@pytest.mark.parametrize("rating", ["0.5", "3.5", "5"])
def test_parser_accepts_exact_half_star_ratings(rating: str) -> None:
    watched = WATCHED_HEADER + "2024,Film,2000,https://boxd.it/film\n"
    ratings = RATINGS_HEADER + (f"2024,Film,2000,https://boxd.it/film,{rating}\n")

    export = parse_letterboxd_export(_export_zip(watched, ratings))

    assert export.entries[0].rating == float(rating)


def test_parser_accepts_missing_ratings_file_and_nested_bom_csv() -> None:
    watched = (
        "\ufeffDate,Name,Year,Letterboxd URI\n"
        "2024,  A   Film  ,not-a-year,https://boxd.it/film\n"
    ).encode()

    export = parse_letterboxd_export(
        _export_zip(watched, watched_name="letterboxd-export/watched.csv")
    )

    assert export.entries == (
        LetterboxdExportEntry(
            uri="https://boxd.it/film",
            name="A Film",
            year=None,
            rating=None,
        ),
    )


def test_parser_treats_empty_ratings_file_as_unrated() -> None:
    watched = WATCHED_HEADER + "2024,Film,2000,https://boxd.it/film\n"

    export = parse_letterboxd_export(_export_zip(watched, b""))

    assert export.entries[0].rating is None


def test_parser_collapses_exact_duplicate_rows_and_ratings() -> None:
    watched_row = "2024,Film,2000,https://boxd.it/film\n"
    rating_row = "2024,Film,2000,https://boxd.it/film,3.5\n"

    export = parse_letterboxd_export(
        _export_zip(
            WATCHED_HEADER + watched_row + watched_row,
            RATINGS_HEADER + rating_row + rating_row,
        )
    )

    assert len(export.entries) == 1
    assert export.entries[0].rating == 3.5


@pytest.mark.parametrize(
    ("archive", "message"),
    [
        (b"not a zip", "valid ZIP"),
        (_export_zip(None), "missing required watched.csv"),
        (
            _export_zip("Date,Name,Year\n2024,Film,2000\n"),
            "Letterboxd URI",
        ),
        (
            _export_zip(WATCHED_HEADER + "2024,,2000,https://boxd.it/film\n"),
            "blank film name",
        ),
        (
            _export_zip(WATCHED_HEADER),
            "does not contain any watched films",
        ),
        (
            _export_zip(b"\xff\xfe\x00\x00"),
            "must be UTF-8 encoded",
        ),
        (
            _export_zip(
                WATCHED_HEADER
                + "2024,Film,2000,https://boxd.it/film\n"
                + "2024,Other Film,2001,https://boxd.it/film\n"
            ),
            "conflicting rows",
        ),
        (
            _export_zip(
                WATCHED_HEADER + "2024,Film,2000,https://boxd.it/film\n",
                RATINGS_HEADER + "2024,Film,2000,https://boxd.it/film,4.25\n",
            ),
            "invalid rating",
        ),
        (
            _export_zip(
                WATCHED_HEADER + "2024,Film,2000,https://boxd.it/film\n",
                RATINGS_HEADER
                + "2024,Film,2000,https://boxd.it/film,3\n"
                + "2024,Film,2000,https://boxd.it/film,4\n",
            ),
            "conflicting ratings",
        ),
    ],
)
def test_parser_rejects_invalid_exports(archive: bytes, message: str) -> None:
    with pytest.raises(LetterboxdExportError, match=message):
        parse_letterboxd_export(archive)


def test_resolution_matches_title_then_original_title_and_reports_failures() -> None:
    entries = (
        LetterboxdExportEntry("uri-1", "  TAXI   driver ", 1976, 5.0),
        LetterboxdExportEntry("uri-2", "Stalker", 1979, None),
        LetterboxdExportEntry("uri-3", "Sen to Chihiro no kamikakushi", 2001, 4.5),
        LetterboxdExportEntry("uri-4", "Unknown", 2020, None),
        LetterboxdExportEntry("uri-5", "Shared", 1999, 3.0),
    )
    catalog = (
        CatalogFilm(1, "taxi-driver", "Taxi Driver", "Taxi Driver", 1976),
        CatalogFilm(5, "stalker", "Stalker", "Stalker", 1979),
        CatalogFilm(
            2,
            "spirited-away",
            "Spirited Away",
            "Sen to Chihiro no kamikakushi",
            2001,
        ),
        CatalogFilm(3, "shared-a", "Shared", None, 1999),
        CatalogFilm(4, "shared-b", "Shared", None, 1999),
    )

    profile, unresolved = resolve_export_profile(
        username=" cinephile ", entries=entries, catalog=catalog
    )

    assert profile.username == "cinephile"
    assert [(watch.film_slug, watch.rating) for watch in profile.watches] == [
        ("taxi-driver", 5.0),
        ("stalker", None),
        ("spirited-away", 4.5),
    ]
    assert [(item.name, item.reason) for item in unresolved] == [
        ("Unknown", "not_found"),
        ("Shared", "ambiguous"),
    ]


def test_resolution_deduplicates_same_film_candidate_by_id() -> None:
    entry = LetterboxdExportEntry("uri", "Film", 2000, None)
    duplicated_catalog_row = CatalogFilm(1, "film", "Film", "Film", 2000)

    profile, unresolved = resolve_export_profile(
        username="user",
        entries=(entry,),
        catalog=(duplicated_catalog_row, duplicated_catalog_row),
    )

    assert [watch.film_slug for watch in profile.watches] == ["film"]
    assert unresolved == ()


class _SessionContext:
    def __init__(self) -> None:
        self.session = SimpleNamespace()

    async def __aenter__(self) -> SimpleNamespace:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        return None


class _SessionFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> _SessionContext:
        self.calls += 1
        return _SessionContext()


def test_service_batches_resolution_and_reuses_profile_sync(monkeypatch) -> None:
    session_factory = _SessionFactory()
    syncer = AsyncMock()
    catalog_lookup = AsyncMock(
        return_value=[CatalogFilm(1, "taxi-driver", "Taxi Driver", None, 1976)]
    )
    user_lookup = AsyncMock(return_value=SimpleNamespace(id=42))
    monkeypatch.setattr(
        import_service_module.FilmRepository,
        "get_catalog_by_years",
        catalog_lookup,
    )
    monkeypatch.setattr(
        import_service_module.UserRepository,
        "get_by_username",
        user_lookup,
    )
    service = LetterboxdImportService(
        session_factory=session_factory,
        syncer=syncer,
    )

    result = asyncio.run(service.import_export("cinephile", _standard_export()))

    catalog_lookup.assert_awaited_once_with({1976, 1979})
    syncer.assert_awaited_once()
    profile = syncer.await_args.args[0]
    assert profile.username == "cinephile"
    assert [(watch.film_slug, watch.rating) for watch in profile.watches] == [
        ("taxi-driver", 5.0)
    ]
    assert syncer.await_args.kwargs == {"session_factory": session_factory}
    assert result.user_id == 42
    assert result.watched_in_export == 2
    assert result.rated_in_export == 1
    assert result.imported == 1
    assert result.unresolved == 1
    assert result.unresolved_films[0].name == "Stalker"
    assert session_factory.calls == 2


def test_service_rejects_zero_resolved_films_without_persisting(monkeypatch) -> None:
    session_factory = _SessionFactory()
    syncer = AsyncMock()
    catalog_lookup = AsyncMock(return_value=[])
    monkeypatch.setattr(
        import_service_module.FilmRepository,
        "get_catalog_by_years",
        catalog_lookup,
    )
    service = LetterboxdImportService(
        session_factory=session_factory,
        syncer=syncer,
    )

    with pytest.raises(NoResolvedFilmsError, match="No films"):
        asyncio.run(service.import_export("cinephile", _standard_export()))

    catalog_lookup.assert_awaited_once()
    syncer.assert_not_awaited()
    assert session_factory.calls == 1


def test_import_route_returns_bounded_unresolved_sample() -> None:
    from app.api.routes.imports import import_letterboxd_export

    unresolved = tuple(
        UnresolvedExportFilm(
            name=f"Film {index}",
            year=2000,
            uri=f"uri-{index}",
            reason="not_found",
        )
        for index in range(25)
    )
    result = LetterboxdImportResult(
        user_id=7,
        watched_in_export=26,
        rated_in_export=1,
        imported=1,
        unresolved_films=unresolved,
    )
    service = SimpleNamespace(import_export=AsyncMock(return_value=result))
    upload = SimpleNamespace(
        read=AsyncMock(return_value=b"zip bytes"),
        close=AsyncMock(),
    )

    response = asyncio.run(
        import_letterboxd_export(
            username="cinephile",
            file=upload,
            service=service,
        )
    )

    service.import_export.assert_awaited_once_with("cinephile", b"zip bytes")
    upload.close.assert_awaited_once()
    assert response.unresolved == 25
    assert len(response.unresolved_sample) == 20


def test_import_route_translates_controlled_parser_error() -> None:
    from app.api.routes.imports import import_letterboxd_export

    service = SimpleNamespace(
        import_export=AsyncMock(side_effect=LetterboxdExportError("bad export"))
    )
    upload = SimpleNamespace(
        read=AsyncMock(return_value=b"bad"),
        close=AsyncMock(),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            import_letterboxd_export(
                username="cinephile",
                file=upload,
                service=service,
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "bad export"
    upload.close.assert_awaited_once()
