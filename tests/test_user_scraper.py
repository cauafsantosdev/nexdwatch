"""Tests for typed profile scraping."""

from types import SimpleNamespace

import pytest
from letterboxdpy.core.exceptions import (
    InvalidResponseError,
    PageLoadError,
    ResourceNotFoundError,
)

from app.scraper import user_scraping


def _mock_user_films(monkeypatch, movies: dict) -> None:
    client = SimpleNamespace(get_films=lambda: {"movies": movies})
    monkeypatch.setattr(user_scraping, "UserFilms", lambda _: client)


def test_scraper_retains_rated_and_unrated_watches(monkeypatch) -> None:
    _mock_user_films(
        monkeypatch,
        {
            "slug-a": {"slug": "slug-a", "rating": 4.5},
            "slug-b": {"slug": "slug-b", "rating": 2.0},
            "slug-c": {"slug": "slug-c", "rating": None},
        },
    )

    profile = user_scraping.scrape_user_profile("cinephile")

    assert [watch.film_slug for watch in profile.watches] == [
        "slug-a",
        "slug-b",
        "slug-c",
    ]
    assert [watch.rating for watch in profile.watches] == [4.5, 2.0, None]


def test_scraper_safely_handles_malformed_entries(monkeypatch) -> None:
    _mock_user_films(
        monkeypatch,
        {
            "": {"slug": " ", "rating": 3.0},
            "malformed": None,
            "valid": {"slug": "valid", "rating": "invalid"},
        },
    )

    profile = user_scraping.scrape_user_profile("cinephile")

    assert len(profile.watches) == 1
    assert profile.watches[0].film_slug == "valid"
    assert profile.watches[0].rating is None


def test_legacy_logs_include_unrated_watches(monkeypatch) -> None:
    _mock_user_films(
        monkeypatch,
        {"slug-c": {"slug": "slug-c", "rating": None}},
    )

    logs = user_scraping.scrape_user_logs("cinephile")

    assert logs == [{"username": "cinephile", "slug": "slug-c", "rating": None}]


@pytest.mark.parametrize(
    "exception",
    [
        PageLoadError("https://letterboxd.com/user"),
        InvalidResponseError("rate limited", code=429),
        InvalidResponseError("upstream unavailable", code=503),
    ],
)
def test_scraper_translates_retryable_upstream_failures(
    monkeypatch, exception: Exception
) -> None:
    client = SimpleNamespace(get_films=lambda: (_ for _ in ()).throw(exception))
    monkeypatch.setattr(user_scraping, "UserFilms", lambda _: client)

    with pytest.raises(user_scraping.TransientProfileScrapeError):
        user_scraping.scrape_user_profile("cinephile")


@pytest.mark.parametrize(
    "exception",
    [
        ResourceNotFoundError("https://letterboxd.com/missing"),
        InvalidResponseError("bad request", code=400),
        RuntimeError("unexpected library failure"),
    ],
)
def test_scraper_translates_non_retryable_failures(
    monkeypatch, exception: Exception
) -> None:
    client = SimpleNamespace(get_films=lambda: (_ for _ in ()).throw(exception))
    monkeypatch.setattr(user_scraping, "UserFilms", lambda _: client)

    with pytest.raises(user_scraping.ProfileScrapeError) as exc_info:
        user_scraping.scrape_user_profile("cinephile")

    assert not isinstance(exc_info.value, user_scraping.TransientProfileScrapeError)
