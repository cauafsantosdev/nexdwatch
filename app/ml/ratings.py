"""Shared Letterboxd rating semantics without ML runtime dependencies."""

import math

RATING_BUCKET_COUNT = 10


def rating_to_bucket(rating: float) -> int:
    """Map a finite Letterboxd half-star rating to a zero-based model bucket.

    Args:
        rating: Explicit rating from 0.5 through 5.0 in exact half-star increments.

    Returns:
        int: Bucket 0 through 9, preserving all ten Letterboxd rating values.

    Raises:
        ValueError: If the value is boolean, non-finite, out of range, or not a
            half-star increment.
    """
    if isinstance(rating, bool) or not math.isfinite(rating):
        raise ValueError("rating must be a finite half-star value from 0.5 to 5.0")
    doubled = rating * 2
    rounded = round(doubled)
    if not math.isclose(doubled, rounded, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError("rating must use half-star increments")
    if not 1 <= rounded <= RATING_BUCKET_COUNT:
        raise ValueError("rating must be between 0.5 and 5.0")
    return rounded - 1
