"""Tests for typed profile scraping."""

from types import SimpleNamespace

import pytest
from bs4 import BeautifulSoup
from curl_cffi import requests
from letterboxdpy.core.exceptions import (
    InvalidResponseError,
    PageLoadError,
    ResourceNotFoundError,
)
from pydantic import SecretStr

from app.scraper import letterboxd_transport, user_scraping


def _mock_user_films(monkeypatch, movies: dict) -> None:
    client = SimpleNamespace(get_films=lambda: {"movies": movies})
    monkeypatch.setattr(user_scraping, "UserFilms", lambda _: client)


def _settings(api_key: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        ZENROWS_API_KEY=SecretStr(api_key) if api_key is not None else None
    )


def _zenrows_response(html: str, *, status_code: int = 200) -> SimpleNamespace:
    return SimpleNamespace(
        status_code=status_code,
        text=html,
        reason="OK" if status_code == 200 else "Service Unavailable",
        headers={},
    )


def test_user_films_transport_stays_direct_without_zenrows(monkeypatch) -> None:
    url = "https://letterboxd.com/cinephile/films/page/1/"
    direct_dom = BeautifulSoup("<html><title>Direct</title></html>", "lxml")
    direct_calls = []
    monkeypatch.setattr(
        letterboxd_transport, "get_settings", lambda: _settings(None)
    )
    monkeypatch.setattr(
        letterboxd_transport,
        "_direct_parse_url",
        lambda requested_url: direct_calls.append(requested_url) or direct_dom,
    )
    monkeypatch.setattr(letterboxd_transport.requests, "get", pytest.fail)

    result = letterboxd_transport.parse_user_films_page(url)

    assert result is direct_dom
    assert direct_calls == [url]


def test_user_films_transport_uses_zenrows_auto_mode(monkeypatch) -> None:
    url = "https://letterboxd.com/cinephile/films/page/1/"
    calls = []
    monkeypatch.setattr(
        letterboxd_transport, "get_settings", lambda: _settings("zenrows-secret")
    )

    def get(endpoint, *, params, timeout):
        calls.append((endpoint, params, timeout))
        return _zenrows_response("<html><title>Proxied</title></html>")

    monkeypatch.setattr(letterboxd_transport.requests, "get", get)

    dom = letterboxd_transport.parse_user_films_page(url)

    assert dom.title.string == "Proxied"
    assert dom.final_url == url
    assert calls == [
        (
            letterboxd_transport.ZENROWS_ENDPOINT,
            {"url": url, "apikey": "zenrows-secret", "mode": "auto"},
            letterboxd_transport._ZENROWS_TIMEOUT,
        )
    ]


def test_user_films_transport_keeps_unrelated_urls_direct(monkeypatch) -> None:
    url = "https://example.com/not-letterboxd"
    direct_dom = BeautifulSoup("<html><title>Direct</title></html>", "lxml")
    monkeypatch.setattr(
        letterboxd_transport,
        "get_settings",
        lambda: _settings("zenrows-secret"),
    )
    monkeypatch.setattr(
        letterboxd_transport, "_direct_parse_url", lambda requested_url: direct_dom
    )
    monkeypatch.setattr(letterboxd_transport.requests, "get", pytest.fail)

    assert letterboxd_transport.parse_user_films_page(url) is direct_dom


def test_zenrows_html_flows_through_existing_user_films_parser(monkeypatch) -> None:
    html = """
    <html><body><ul>
      <li class="griditem">
        <div class="react-component" data-film-id="7"
             data-item-slug="parsed-film" data-item-name="Parsed Film (2024)"></div>
        <p class="poster-viewingdata"><span class="rated-9"></span></p>
      </li>
    </ul></body></html>
    """
    monkeypatch.setattr(
        letterboxd_transport, "get_settings", lambda: _settings("zenrows-secret")
    )
    monkeypatch.setattr(
        letterboxd_transport.requests,
        "get",
        lambda *args, **kwargs: _zenrows_response(html),
    )

    profile = user_scraping.scrape_user_profile("cinephile")

    assert [(watch.film_slug, watch.rating) for watch in profile.watches] == [
        ("parsed-film", 4.5)
    ]


def test_zenrows_upstream_failure_becomes_existing_transient_error(monkeypatch) -> None:
    monkeypatch.setattr(
        letterboxd_transport, "get_settings", lambda: _settings("zenrows-secret")
    )
    monkeypatch.setattr(
        letterboxd_transport.requests,
        "get",
        lambda *args, **kwargs: _zenrows_response("unavailable", status_code=503),
    )

    with pytest.raises(user_scraping.TransientProfileScrapeError):
        user_scraping.scrape_user_profile("cinephile")


def test_zenrows_transport_exception_does_not_retain_api_key(monkeypatch) -> None:
    api_key = "never-expose-this-key"
    monkeypatch.setattr(
        letterboxd_transport, "get_settings", lambda: _settings(api_key)
    )

    def fail(*args, **kwargs):
        raise requests.errors.RequestsError(f"failed request apikey={api_key}")

    monkeypatch.setattr(letterboxd_transport.requests, "get", fail)

    with pytest.raises(user_scraping.TransientProfileScrapeError) as exc_info:
        user_scraping.scrape_user_profile("cinephile")

    transport_error = exc_info.value.__cause__
    assert transport_error is not None
    assert api_key not in str(exc_info.value)
    assert api_key not in str(transport_error)
    assert transport_error.__cause__ is None


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
