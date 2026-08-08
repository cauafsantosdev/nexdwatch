"""Letterboxd profile scraping adapters."""

import logging
from typing import Any

from scrapxd import Scrapxd

from app.domain.profiles import ScrapedProfile, ScrapedWatch

logger = logging.getLogger(__name__)


def scrape_user_profile(username: str) -> ScrapedProfile:
    """Scrape rated and unrated watched films from a public profile.

    Args:
        username: Public Letterboxd username.

    Returns:
        A typed profile containing every entry with a valid film slug.
    """
    client = Scrapxd()
    user = client.get_user(username=username)
    entries = getattr(getattr(user, "logs", None), "entries", ()) or ()

    watches: list[ScrapedWatch] = []
    for entry in entries:
        film = getattr(entry, "film", None)
        slug = getattr(film, "slug", None)
        if not isinstance(slug, str) or not slug.strip():
            logger.debug("Skipping profile entry without a valid film slug")
            continue

        raw_rating = getattr(entry, "rating", None)
        try:
            rating = float(raw_rating) if raw_rating is not None else None
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid rating for film_slug=%s", slug)
            rating = None

        watches.append(ScrapedWatch(film_slug=slug.strip(), rating=rating))

    return ScrapedProfile(username=username, watches=tuple(watches))


def scrape_user_logs(username: str) -> list[dict[str, Any]]:
    """Return the legacy dictionary representation for compatibility."""
    profile = scrape_user_profile(username)
    return [
        {
            "username": profile.username,
            "slug": watch.film_slug,
            "rating": watch.rating,
        }
        for watch in profile.watches
    ]
