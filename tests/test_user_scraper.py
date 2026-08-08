"""Tests for typed profile scraping."""

from types import SimpleNamespace

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
