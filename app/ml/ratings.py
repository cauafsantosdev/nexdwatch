"""Shared Letterboxd rating semantics without ML runtime dependencies."""

import math

RATING_BUCKET_COUNT = 10


def rating_to_bucket(rating: float) -> int:
    """Map a valid Letterboxd half-star rating to an index from zero to nine."""
    if isinstance(rating, bool) or not math.isfinite(rating):
        raise ValueError("rating must be a finite half-star value from 0.5 to 5.0")
    doubled = rating * 2
    rounded = round(doubled)
    if not math.isclose(doubled, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("rating must use half-star increments")
    if not 1 <= rounded <= RATING_BUCKET_COUNT:
        raise ValueError("rating must be between 0.5 and 5.0")
    return rounded - 1
