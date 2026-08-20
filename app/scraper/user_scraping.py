"""Adapt blocking ``letterboxdpy`` profile reads to typed ingestion values.

External failures are translated into safe permanent or retryable application
errors. Malformed individual watch entries are skipped, while an invalid top-level
response fails the whole profile so incomplete data is never persisted silently.
"""

import logging
from collections.abc import Mapping
from typing import Any

from letterboxdpy.core.exceptions import (
    AccessDeniedError,
    InvalidResponseError,
    PageLoadError,
    PrivateRouteError,
    ResourceNotFoundError,
)

from app.domain.profiles import ScrapedProfile, ScrapedWatch
from app.scraper.letterboxd_transport import UserFilms

logger = logging.getLogger(__name__)


class ProfileScrapeError(Exception):
    """Safe application error for a profile that cannot be scraped."""


class TransientProfileScrapeError(ProfileScrapeError):
    """Profile scrape failure that can safely be retried later."""


def scrape_user_profile(username: str) -> ScrapedProfile:
    """Scrape rated and unrated watched films from one public profile.

    Transport and server failures are explicitly retryable; inaccessible profiles
    and structurally invalid responses are permanent for the current task attempt.
    Rating conversion failures affect only that watch and preserve it as unrated.

    Args:
        username: Public Letterboxd username.

    Returns:
        ScrapedProfile: Every response entry with a valid film slug, in provider
            iteration order.

    Raises:
        TransientProfileScrapeError: If Letterboxd transport, rate limiting, or a
            server response indicates a later retry may succeed.
        ProfileScrapeError: If the profile is unavailable or the response contract
            cannot be trusted.
    """
    # Keep provider-specific exceptions at this boundary so Celery retry policy does
    # not depend on letterboxdpy implementation details.
    try:
        result = UserFilms(username).get_films()
    except PageLoadError as exc:
        logger.warning("Transient profile transport failure username=%s", username)
        raise TransientProfileScrapeError(
            "Letterboxd profile transport failed."
        ) from exc
    except InvalidResponseError as exc:
        logger.warning(
            "Invalid Letterboxd response username=%s status=%s", username, exc.code
        )
        if exc.code == 429 or (exc.code is not None and exc.code >= 500):
            raise TransientProfileScrapeError(
                "Letterboxd is temporarily unavailable."
            ) from exc
        raise ProfileScrapeError("Letterboxd profile response was invalid.") from exc
    except (ResourceNotFoundError, PrivateRouteError, AccessDeniedError) as exc:
        logger.info("Letterboxd profile unavailable username=%s", username)
        raise ProfileScrapeError("Letterboxd profile is unavailable.") from exc
    except Exception as exc:
        logger.exception("Profile scrape failed for username=%s", username)
        raise ProfileScrapeError("Letterboxd profile synchronization failed.") from exc

    # Reject an invalid response envelope rather than interpreting it as an empty
    # profile, which could incorrectly signal a successful synchronization.
    if not isinstance(result, Mapping) or not isinstance(result.get("movies"), Mapping):
        logger.error("Invalid watched-film response for username=%s", username)
        raise ProfileScrapeError("Letterboxd profile response was invalid.")
    movies = result["movies"]

    # Normalize each watch independently; a damaged row must not discard otherwise
    # usable history from the same public profile.
    watches: list[ScrapedWatch] = []
    for movie_key, movie_data in movies.items():
        if not isinstance(movie_data, Mapping):
            logger.debug("Skipping malformed profile entry for username=%s", username)
            continue

        slug = movie_data.get("slug") or movie_key
        if not isinstance(slug, str) or not slug.strip():
            logger.debug("Skipping profile entry without a valid film slug")
            continue

        raw_rating = movie_data.get("rating")
        try:
            rating = float(raw_rating) if raw_rating is not None else None
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid rating for film_slug=%s", slug)
            rating = None

        watches.append(ScrapedWatch(film_slug=slug.strip(), rating=rating))

    return ScrapedProfile(username=username, watches=tuple(watches))


def scrape_user_logs(username: str) -> list[dict[str, Any]]:
    """Project the current typed profile into the legacy dictionary contract.

    Provider failures from ``scrape_user_profile`` propagate unchanged; this helper
    exists only for callers not yet migrated to ``ScrapedProfile``.
    """
    profile = scrape_user_profile(username)
    return [
        {
            "username": profile.username,
            "slug": watch.film_slug,
            "rating": watch.rating,
        }
        for watch in profile.watches
    ]
