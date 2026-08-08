"""Film metadata scraping with explicit per-slug outcomes."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from scrapxd import Scrapxd

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


def scrape_film_queue(film_slugs: list[str]) -> list[FilmScrapeResult]:
    """Scrape metadata while returning an outcome for every requested slug.

    Args:
        film_slugs: Letterboxd film slugs to scrape.

    Returns:
        Results in the same order as the requested slugs.
    """
    client = Scrapxd()
    results: list[FilmScrapeResult] = []

    for slug in film_slugs:
        try:
            film = client.get_film(slug)
            if film is None:
                results.append(
                    FilmScrapeResult(
                        slug=slug,
                        outcome=FilmScrapeOutcome.FAILED,
                        error="Film was not found",
                    )
                )
                continue

            total_logs = int(film.total_logs)
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
                    metadata={
                        "tmdb_id": film.id,
                        "slug": slug,
                        "title": film.title,
                        "original_title": film.original_title,
                        "year": film.year,
                        "runtime": film.runtime,
                        "director": film.director,
                        "genre": film.genre,
                        "country": film.country,
                        "language": film.language,
                        "actors": film.actors,
                        "studio": film.studio,
                        "synopsis": film.synopsis,
                        "tagline": film.tagline,
                        "themes": film.themes,
                        "avg_rating": film.avg_rating,
                        "total_logs": total_logs,
                    },
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
