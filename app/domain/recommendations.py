"""Domain types for recommendation results."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Recommendation:
    """A ranked film recommendation independent of transport concerns."""

    id: int
    title: str
    director: str | list[str]
    year: int | None
    match_score: float


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    """Recommendations and optional cold-start context for one user."""

    user_id: int
    recommendations: tuple[Recommendation, ...]
    strategy: str | None = None
    info: str | None = None
