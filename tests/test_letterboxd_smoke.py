"""Deterministic tests for the live smoke runner's result contracts."""

import pytest

from app.domain.profiles import ScrapedProfile, ScrapedWatch
from app.scraper import FilmScrapeOutcome, FilmScrapeResult
from scripts.smoke_letterboxd import (
    FILM_SLUG_VARIABLE,
    USERNAME_VARIABLE,
    SmokeCheckError,
    load_config,
    validate_film,
    validate_profile,
)


def test_load_config_names_all_missing_repository_variables() -> None:
    with pytest.raises(SmokeCheckError) as exc_info:
        load_config({})

    message = str(exc_info.value)
    assert USERNAME_VARIABLE in message
    assert FILM_SLUG_VARIABLE in message


def test_profile_validation_accepts_letterboxd_rating_semantics() -> None:
    profile = ScrapedProfile(
        username="public-user",
        watches=(
            ScrapedWatch(film_slug="rated-film", rating=4.5),
            ScrapedWatch(film_slug="unrated-film", rating=None),
        ),
    )

    validate_profile(profile, "public-user")


@pytest.mark.parametrize("rating", [0.0, 5.5, 2.25, float("nan")])
def test_profile_validation_rejects_invalid_ratings(rating: float) -> None:
    profile = ScrapedProfile(
        username="public-user",
        watches=(ScrapedWatch(film_slug="film", rating=rating),),
    )

    with pytest.raises(SmokeCheckError):
        validate_profile(profile, "public-user")


def test_film_validation_requires_successful_catalog_metadata() -> None:
    result = FilmScrapeResult(
        slug="known-film",
        outcome=FilmScrapeOutcome.SUCCESS,
        metadata={"slug": "known-film", "title": "Known Film", "tmdb_id": 42},
    )

    validate_film(result, "known-film")


def test_film_validation_rejects_non_success_outcomes() -> None:
    result = FilmScrapeResult(
        slug="known-film",
        outcome=FilmScrapeOutcome.FILTERED,
        error="below threshold",
    )

    with pytest.raises(SmokeCheckError, match="filtered"):
        validate_film(result, "known-film")
