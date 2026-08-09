"""Parse official Letterboxd export archives without network access."""

import csv
import io
import zipfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 1_000

_WATCHED_HEADERS = frozenset({"Name", "Year", "Letterboxd URI"})
_RATINGS_HEADERS = _WATCHED_HEADERS | {"Rating"}
_VALID_RATINGS = frozenset(Decimal(value) / 2 for value in range(1, 11))


class LetterboxdExportError(ValueError):
    """Raised when an export archive cannot be imported safely."""


@dataclass(frozen=True, slots=True)
class LetterboxdExportEntry:
    """One unique watched-film row from an official export."""

    uri: str
    name: str
    year: int | None
    rating: float | None


@dataclass(frozen=True, slots=True)
class LetterboxdExport:
    """Validated and merged contents of an official export archive."""

    entries: tuple[LetterboxdExportEntry, ...]

    @property
    def watched_count(self) -> int:
        """Return the number of unique watched entries in the archive."""
        return len(self.entries)

    @property
    def rated_count(self) -> int:
        """Return the number of watched entries with an exported rating."""
        return sum(entry.rating is not None for entry in self.entries)


@dataclass(frozen=True, slots=True)
class _WatchedRow:
    uri: str
    name: str
    year: int | None


def parse_letterboxd_export(archive: bytes) -> LetterboxdExport:
    """Parse ``watched.csv`` and optional ``ratings.csv`` from ZIP bytes.

    Members are read directly from memory. No paths are extracted and no
    Letterboxd URLs are followed.
    """
    if not archive or len(archive) > MAX_ARCHIVE_BYTES:
        raise LetterboxdExportError("Export ZIP is empty or exceeds the size limit.")

    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as export_zip:
            members = export_zip.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise LetterboxdExportError("Export ZIP contains too many files.")
            if sum(member.file_size for member in members) > MAX_UNCOMPRESSED_BYTES:
                raise LetterboxdExportError(
                    "Export ZIP exceeds the uncompressed size limit."
                )

            watched_member = _find_member(members, "watched.csv", required=True)
            ratings_member = _find_member(members, "ratings.csv", required=False)
            watched_bytes = _read_member(export_zip, watched_member)
            ratings_bytes = (
                _read_member(export_zip, ratings_member)
                if ratings_member is not None
                else None
            )
    except LetterboxdExportError:
        raise
    except (
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise LetterboxdExportError("Uploaded file is not a valid ZIP export.") from exc

    watched_rows = _parse_watched(watched_bytes)
    ratings = _parse_ratings(ratings_bytes) if ratings_bytes is not None else {}
    entries = tuple(
        LetterboxdExportEntry(
            uri=row.uri,
            name=row.name,
            year=row.year,
            rating=ratings.get(row.uri),
        )
        for row in watched_rows
    )
    return LetterboxdExport(entries=entries)


def _find_member(
    members: list[zipfile.ZipInfo], basename: str, *, required: bool
) -> zipfile.ZipInfo | None:
    matches = [
        member
        for member in members
        if not member.is_dir()
        and PurePosixPath(member.filename.replace("\\", "/")).name.casefold()
        == basename.casefold()
    ]
    if len(matches) > 1:
        raise LetterboxdExportError(f"Export ZIP contains multiple {basename} files.")
    if not matches:
        if required:
            raise LetterboxdExportError(f"Export ZIP is missing required {basename}.")
        return None
    return matches[0]


def _read_member(export_zip: zipfile.ZipFile, member: zipfile.ZipInfo | None) -> bytes:
    if member is None:
        return b""
    try:
        contents = export_zip.read(member)
    except (NotImplementedError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise LetterboxdExportError(
            f"Could not read {PurePosixPath(member.filename).name}."
        ) from exc
    if len(contents) != member.file_size:
        raise LetterboxdExportError("Export ZIP contains an invalid file size.")
    return contents


def _read_csv(
    contents: bytes, filename: str, required_headers: frozenset[str]
) -> list[dict[str, str]]:
    try:
        text = contents.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise LetterboxdExportError(f"{filename} must be UTF-8 encoded.") from exc

    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames is None:
            raise LetterboxdExportError(f"{filename} is missing a header row.")
        headers = [header.strip() for header in reader.fieldnames]
        if len(headers) != len(set(headers)):
            raise LetterboxdExportError(f"{filename} contains duplicate headers.")
        missing = required_headers.difference(headers)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise LetterboxdExportError(
                f"{filename} is missing required columns: {missing_list}."
            )

        rows: list[dict[str, str]] = []
        for raw_row in reader:
            if None in raw_row:
                raise LetterboxdExportError(f"{filename} contains a malformed row.")
            row = {
                key.strip(): (value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }
            if any(row.values()):
                rows.append(row)
        return rows
    except csv.Error as exc:
        raise LetterboxdExportError(f"{filename} contains invalid CSV data.") from exc


def _parse_watched(contents: bytes) -> tuple[_WatchedRow, ...]:
    rows = _read_csv(contents, "watched.csv", _WATCHED_HEADERS)
    watched_by_uri: dict[str, _WatchedRow] = {}
    for row in rows:
        uri = row["Letterboxd URI"]
        name = _normalize_display_name(row["Name"])
        if not uri:
            raise LetterboxdExportError("watched.csv contains a row without a URI.")
        if not name:
            raise LetterboxdExportError("watched.csv contains a blank film name.")

        watched = _WatchedRow(uri=uri, name=name, year=_parse_year(row["Year"]))
        existing = watched_by_uri.get(uri)
        if existing is None:
            watched_by_uri[uri] = watched
        elif _watched_identity(existing) != _watched_identity(watched):
            raise LetterboxdExportError(
                "watched.csv contains conflicting rows for the same URI."
            )

    if not watched_by_uri:
        raise LetterboxdExportError("watched.csv does not contain any watched films.")
    return tuple(watched_by_uri.values())


def _parse_ratings(contents: bytes) -> dict[str, float]:
    try:
        has_content = bool(contents.decode("utf-8-sig").strip())
    except UnicodeDecodeError as exc:
        raise LetterboxdExportError("ratings.csv must be UTF-8 encoded.") from exc
    if not has_content:
        return {}
    rows = _read_csv(contents, "ratings.csv", _RATINGS_HEADERS)
    ratings: dict[str, float] = {}
    for row in rows:
        uri = row["Letterboxd URI"]
        if not uri:
            raise LetterboxdExportError("ratings.csv contains a row without a URI.")
        rating = _parse_rating(row["Rating"])
        existing = ratings.get(uri)
        if existing is not None and existing != rating:
            raise LetterboxdExportError(
                "ratings.csv contains conflicting ratings for the same URI."
            )
        ratings[uri] = rating
    return ratings


def _parse_rating(value: str) -> float:
    try:
        rating = Decimal(value)
    except InvalidOperation as exc:
        raise LetterboxdExportError(
            "ratings.csv contains an invalid rating; expected 0.5 to 5.0."
        ) from exc
    if rating not in _VALID_RATINGS:
        raise LetterboxdExportError(
            "ratings.csv contains an invalid rating; expected half-star increments "
            "from 0.5 to 5.0."
        )
    return float(rating)


def _parse_year(value: str) -> int | None:
    try:
        year = int(value)
    except ValueError:
        return None
    return year if year > 0 else None


def _normalize_display_name(value: str) -> str:
    return " ".join(value.split())


def _watched_identity(row: _WatchedRow) -> tuple[str, int | None]:
    return (row.name.casefold(), row.year)
