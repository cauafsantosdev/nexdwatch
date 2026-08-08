"""Domain types for scraped Letterboxd profiles."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScrapedWatch:
    """A watched film found on a Letterboxd profile."""

    film_slug: str
    rating: float | None


@dataclass(frozen=True, slots=True)
class ScrapedProfile:
    """A Letterboxd profile and its watched-film interactions."""

    username: str
    watches: tuple[ScrapedWatch, ...]
