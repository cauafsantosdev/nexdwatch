"""Select direct or ZenRows transport for ``letterboxdpy`` user-film pages.

``letterboxdpy`` binds its page fetcher at module import time and does not expose
transport injection. This adapter replaces only that binding with a dispatcher;
all pagination and HTML extraction remain owned by the pinned library.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from curl_cffi import requests
from letterboxdpy.core.exceptions import PageLoadError
from letterboxdpy.core.scraper import Scraper
from letterboxdpy.pages import user_films as letterboxd_user_films

from app.core.config import get_settings

ZENROWS_ENDPOINT = "https://api.zenrows.com/v1/"
_ZENROWS_TIMEOUT = (10, 30)
_direct_parse_url = letterboxd_user_films.parse_url


@dataclass(frozen=True, slots=True)
class _SanitizedResponse:
    """Minimal response contract consumed by ``letterboxdpy`` without secrets."""

    status_code: int
    text: str
    url: str
    reason: str
    headers: Mapping[str, str]


def _is_letterboxd_url(url: str) -> bool:
    hostname = (urlsplit(url).hostname or "").lower()
    return hostname == "letterboxd.com" or hostname.endswith(".letterboxd.com")


def parse_user_films_page(url: str) -> BeautifulSoup:
    """Fetch one user-film page directly or through configured ZenRows transport."""
    configured_key = get_settings().ZENROWS_API_KEY
    if (
        configured_key is None
        or not (api_key := configured_key.get_secret_value())
        or not _is_letterboxd_url(url)
    ):
        return _direct_parse_url(url)

    try:
        response = requests.get(
            ZENROWS_ENDPOINT,
            params={"url": url, "apikey": api_key, "mode": "auto"},
            timeout=_ZENROWS_TIMEOUT,
        )
    except requests.errors.RequestsError:
        # Do not retain the provider exception: its request URL contains the key.
        raise PageLoadError(url, "ZenRows transport failed") from None

    safe_response = _SanitizedResponse(
        status_code=response.status_code,
        # Error bodies are not parser inputs and may echo request diagnostics.
        text=response.text if response.status_code == 200 else "",
        url=url,
        reason=str(response.reason),
        headers=response.headers,
    )
    Scraper._check_for_errors(url, safe_response)
    return Scraper._parse_html(safe_response)


# UserFilms resolves this module global for every paginated request. Other
# letterboxdpy page types retain their original direct transport binding.
letterboxd_user_films.parse_url = parse_user_films_page
UserFilms = letterboxd_user_films.UserFilms

__all__ = ["UserFilms"]
