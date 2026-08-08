"""Film metadata scraping with explicit per-slug outcomes."""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from letterboxdpy.movie import Movie

logger = logging.getLogger(__name__)
MINIMUM_TOTAL_LOGS = 1_000


class FilmScrapeOutcome(str, Enum):
    """Outcome classifications produced for each requested film."""

    SUCCESS = "success"
    FILTERED = "filtered"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FilmScrapeResult:
    """Observable scrape result for a single film slug."""

    slug: str
    outcome: FilmScrapeOutcome
    metadata: dict[str, Any] | None = None
    error: str | None = None


def _deduplicated_names(
    items: Iterable[Any] | None,
    *,
    item_type: str | None = None,
) -> list[str]:
    """Extract unique names in source order from letterboxdpy metadata."""
    names: list[str] = []
    seen: set[str] = set()
    for item in items or ():
        if isinstance(item, Mapping):
            if item_type is not None and item.get("type") != item_type:
                continue
            raw_name = item.get("name")
        else:
            if item_type is not None:
                continue
            raw_name = item

        if not isinstance(raw_name, str):
            continue
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _rating_count(movie: Movie) -> int:
    """Return JSON-LD aggregate rating count used by the existing threshold."""
    profile = getattr(getattr(movie, "pages", None), "profile", None)
    script = getattr(profile, "script", None)
    if not isinstance(script, Mapping):
        raise TypeError("movie profile JSON-LD is unavailable")

    aggregate = script.get("aggregateRating")
    if not isinstance(aggregate, Mapping):
        raise TypeError("aggregate rating metadata is unavailable")

    raw_rating_count = aggregate.get("ratingCount")
    if raw_rating_count is None or isinstance(raw_rating_count, bool):
        raise ValueError("aggregate rating count is unavailable")
    rating_count = int(raw_rating_count)
    if rating_count < 0:
        raise ValueError("aggregate rating count is invalid")
    return rating_count


def _tmdb_id(movie: Movie) -> int:
    """Normalize the external TMDB identifier for persistence."""
    raw_tmdb_id = movie.tmdb_id
    if raw_tmdb_id is None or isinstance(raw_tmdb_id, bool):
        raise ValueError("TMDB identifier is unavailable")
    return int(raw_tmdb_id)


def _metadata(movie: Movie, slug: str, total_logs: int) -> dict[str, Any]:
    """Normalize letterboxdpy movie data into the ingestion contract."""
    title = movie.title.strip() if isinstance(movie.title, str) else ""
    if not title:
        raise ValueError("movie title is unavailable")

    original_title = (
        movie.original_title.strip()
        if isinstance(movie.original_title, str) and movie.original_title.strip()
        else title
    )
    crew = movie.crew if isinstance(movie.crew, Mapping) else {}

    return {
        "tmdb_id": _tmdb_id(movie),
        "slug": slug,
        "title": title,
        "original_title": original_title,
        "year": movie.year,
        "runtime": movie.runtime,
        "director": _deduplicated_names(crew.get("director")),
        "genre": _deduplicated_names(movie.genres, item_type="genre"),
        "country": _deduplicated_names(movie.details, item_type="country"),
        "language": _deduplicated_names(movie.details, item_type="language"),
        "actors": _deduplicated_names(movie.cast),
        "studio": _deduplicated_names(movie.details, item_type="studio"),
        "synopsis": movie.description,
        "tagline": movie.tagline,
        "themes": _deduplicated_names(movie.genres, item_type="theme"),
        "avg_rating": movie.rating,
        "total_logs": total_logs,
    }


def scrape_film_queue(film_slugs: list[str]) -> list[FilmScrapeResult]:
    """Scrape metadata while returning an outcome for every requested slug.

    Args:
        film_slugs: Letterboxd film slugs to scrape.

    Returns:
        Results in the same order as the requested slugs.
    """
    results: list[FilmScrapeResult] = []

    for slug in film_slugs:
        try:
            movie = Movie(slug)
            total_logs = _rating_count(movie)
            if total_logs < MINIMUM_TOTAL_LOGS:
                results.append(
                    FilmScrapeResult(
                        slug=slug,
                        outcome=FilmScrapeOutcome.FILTERED,
                        error=(
                            f"Film has {total_logs} logs; minimum is "
                            f"{MINIMUM_TOTAL_LOGS}"
                        ),
                    )
                )
                continue

            results.append(
                FilmScrapeResult(
                    slug=slug,
                    outcome=FilmScrapeOutcome.SUCCESS,
                    metadata=_metadata(movie, slug, total_logs),
                )
            )
        except Exception as exc:
            logger.exception("Film scrape failed for film_slug=%s", slug)
            results.append(
                FilmScrapeResult(
                    slug=slug,
                    outcome=FilmScrapeOutcome.FAILED,
                    error=f"Scrape failed: {type(exc).__name__}",
                )
            )

    return results
