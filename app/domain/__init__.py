"""Domain representations used across application boundaries."""

from .profiles import ScrapedProfile, ScrapedWatch
from .recommendations import Recommendation, RecommendationResult

__all__ = [
    "Recommendation",
    "RecommendationResult",
    "ScrapedProfile",
    "ScrapedWatch",
]
