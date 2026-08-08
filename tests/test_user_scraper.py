"""Tests for typed profile scraping."""

from types import SimpleNamespace

from app.scraper import user_scraping


def test_scraper_retains_rated_and_unrated_watches(monkeypatch) -> None:
    entries = [
        SimpleNamespace(film=SimpleNamespace(slug="rated-film"), rating=4.5),
        SimpleNamespace(film=SimpleNamespace(slug="unrated-film"), rating=None),
    ]
    user = SimpleNamespace(logs=SimpleNamespace(entries=entries))
    client = SimpleNamespace(get_user=lambda **_: user)
    monkeypatch.setattr(user_scraping, "Scrapxd", lambda: client)

    profile = user_scraping.scrape_user_profile("cinephile")

    assert [watch.film_slug for watch in profile.watches] == [
        "rated-film",
        "unrated-film",
    ]
    assert [watch.rating for watch in profile.watches] == [4.5, None]


def test_scraper_safely_skips_entries_without_valid_films(monkeypatch) -> None:
    entries = [
        SimpleNamespace(film=None, rating=3.0),
        SimpleNamespace(film=SimpleNamespace(slug=" "), rating=2.0),
        SimpleNamespace(film=SimpleNamespace(slug="valid"), rating="invalid"),
    ]
    user = SimpleNamespace(logs=SimpleNamespace(entries=entries))
    client = SimpleNamespace(get_user=lambda **_: user)
    monkeypatch.setattr(user_scraping, "Scrapxd", lambda: client)

    profile = user_scraping.scrape_user_profile("cinephile")

    assert len(profile.watches) == 1
    assert profile.watches[0].film_slug == "valid"
    assert profile.watches[0].rating is None
