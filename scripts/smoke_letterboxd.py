"""Exercise NexdWatch's live Letterboxd adapters with bounded inputs."""

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from app.domain.profiles import ScrapedProfile
from app.scraper import FilmScrapeOutcome, FilmScrapeResult, scrape_user_profile
from app.scraper import scrape_film_queue as scrape_films
from app.scraper.user_scraping import ProfileScrapeError

USERNAME_VARIABLE = "LETTERBOXD_SMOKE_USERNAME"
FILM_SLUG_VARIABLE = "LETTERBOXD_SMOKE_FILM_SLUG"


class SmokeCheckError(RuntimeError):
    """Explain a smoke-check configuration or contract failure."""


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    """Public Letterboxd targets configured for the scheduled canary."""

    username: str
    film_slug: str


def load_config(environ: Mapping[str, str]) -> SmokeConfig:
    """Load required targets and report every missing repository variable."""
    values = {
        USERNAME_VARIABLE: environ.get(USERNAME_VARIABLE, "").strip(),
        FILM_SLUG_VARIABLE: environ.get(FILM_SLUG_VARIABLE, "").strip(),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        variable_list = ", ".join(missing)
        raise SmokeCheckError(
            "Configure these GitHub Actions repository variables before running "
            f"the Letterboxd smoke workflow: {variable_list}."
        )

    return SmokeConfig(
        username=values[USERNAME_VARIABLE],
        film_slug=values[FILM_SLUG_VARIABLE],
    )


def validate_profile(profile: ScrapedProfile, expected_username: str) -> None:
    """Validate stable application-level invariants for one scraped profile."""
    if profile.username != expected_username:
        raise SmokeCheckError(
            "Profile scraper returned a username different from the requested one."
        )
    if not profile.watches:
        raise SmokeCheckError("Profile scraper returned no watched films.")

    for watch in profile.watches:
        if not watch.film_slug.strip():
            raise SmokeCheckError("Profile scraper returned an empty film slug.")
        if watch.rating is None:
            continue
        if not math.isfinite(watch.rating) or not 0.5 <= watch.rating <= 5.0:
            raise SmokeCheckError(
                f"Profile scraper returned an invalid rating for {watch.film_slug}."
            )
        if not (watch.rating * 2).is_integer():
            raise SmokeCheckError(
                "Profile scraper returned a rating outside Letterboxd's half-star "
                f"scale for {watch.film_slug}."
            )


def validate_film(result: FilmScrapeResult, expected_slug: str) -> None:
    """Validate stable catalog invariants for one production film scrape."""
    if result.slug != expected_slug:
        raise SmokeCheckError(
            "Film scraper returned a slug different from the requested one."
        )
    if result.outcome is not FilmScrapeOutcome.SUCCESS:
        detail = f" ({result.error})" if result.error else ""
        raise SmokeCheckError(f"Film scraper returned {result.outcome.value}{detail}.")
    if not isinstance(result.metadata, Mapping):
        raise SmokeCheckError("Film scraper returned no metadata.")

    metadata = result.metadata
    if metadata.get("slug") != expected_slug:
        raise SmokeCheckError("Film metadata contains an unexpected canonical slug.")
    title = metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        raise SmokeCheckError("Film scraper returned an empty title.")
    tmdb_id = metadata.get("tmdb_id")
    if isinstance(tmdb_id, bool) or not isinstance(tmdb_id, int) or tmdb_id <= 0:
        raise SmokeCheckError("Film scraper returned an invalid TMDB identity.")


def run() -> None:
    """Run one profile scrape and one single-film scrape through NexdWatch."""
    config = load_config(os.environ)

    profile = scrape_user_profile(config.username)
    validate_profile(profile, config.username)
    print(
        f"Profile adapter passed for {config.username!r} "
        f"with {len(profile.watches)} watched films.",
        flush=True,
    )

    results = scrape_films([config.film_slug])
    if len(results) != 1:
        raise SmokeCheckError(
            f"Film scraper returned {len(results)} results for one requested slug."
        )
    validate_film(results[0], config.film_slug)
    print(
        f"Film adapter passed for {config.film_slug!r}.",
        flush=True,
    )


def main() -> int:
    """Return a process status with a concise failure instead of a traceback."""
    try:
        run()
    except (ProfileScrapeError, SmokeCheckError) as exc:
        print(
            f"Letterboxd scraper smoke failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
