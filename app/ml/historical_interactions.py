"""Controlled historical interaction loading and leakage-safe evaluation splits."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from app.ml.ratings import rating_to_bucket


@dataclass(frozen=True, slots=True)
class TrainingDataSummary:
    """Counts reported while resolving the controlled training cohort."""

    csv_rows: int
    resolved_rows: int
    unresolved_rows: int
    duplicate_rows: int
    unique_users: int
    unique_films: int


@dataclass(frozen=True, slots=True)
class HistoricalUser:
    """Compact rated history for one curated-dataset user."""

    cohort_user_id: int
    item_rows: NDArray[np.int64]
    rating_buckets: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PreparedInteractions:
    """Resolved film vocabulary and compact per-user histories."""

    film_ids: NDArray[np.int64]
    users: tuple[HistoricalUser, ...]
    summary: TrainingDataSummary


@dataclass(frozen=True, slots=True)
class UserSplit:
    """Leakage-safe targets and context for one historical user."""

    cohort_user_id: int
    all_item_rows: NDArray[np.int64]
    all_rating_buckets: NDArray[np.int64]
    context_item_rows: NDArray[np.int64]
    context_rating_buckets: NDArray[np.int64]
    training_positive_rows: NDArray[np.int64]
    explicit_negative_rows: NDArray[np.int64]
    validation_target: int | None
    test_target: int | None


def load_historical_interactions(
    csv_path: str | Path,
    slug_to_film_id: Mapping[str, int],
    *,
    chunk_size: int = 250_000,
) -> PreparedInteractions:
    """Resolve a controlled historical CSV into compact per-user interactions.

    The file is streamed in bounded pandas chunks. Slugs resolve through the supplied
    in-memory catalog mapping, ratings become exact half-star buckets, and identical
    user/film duplicates collapse after a deterministic sort. Conflicting duplicate
    ratings fail the dataset rather than selecting an arbitrary row.

    Args:
        csv_path: Controlled semicolon-delimited interaction dataset.
        slug_to_film_id: Complete unique mapping from dataset slug to model film ID.
        chunk_size: Maximum rows parsed into memory per pandas chunk.

    Returns:
        PreparedInteractions: Sorted film vocabulary, compact user histories, and
            resolution/deduplication counts.

    Raises:
        FileNotFoundError: If the controlled dataset path is absent.
        ValueError: If mappings, schema, identities, ratings, duplicates, or the
            resolved interaction universe are invalid.
    """
    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"controlled training dataset not found: {source}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    _validate_film_mapping(slug_to_film_id)

    user_chunks: list[NDArray[np.int64]] = []
    film_chunks: list[NDArray[np.int64]] = []
    rating_chunks: list[NDArray[np.int64]] = []
    csv_rows = 0
    unresolved_rows = 0

    # Stream the source and resolve category-coded slugs vectorially; no database or
    # per-row lookup is permitted in this controlled-data boundary.
    try:
        chunks = pd.read_csv(
            source,
            sep=";",
            usecols=["user_id", "username", "slug", "rating"],
            chunksize=chunk_size,
            dtype={"username": "category", "slug": "category"},
        )
        for chunk in chunks:
            csv_rows += len(chunk)
            if chunk[["user_id", "slug", "rating"]].isnull().any().any():
                raise ValueError("training rows require user_id, slug, and rating")

            numeric_users = pd.to_numeric(chunk["user_id"], errors="coerce")
            numeric_ratings = pd.to_numeric(chunk["rating"], errors="coerce")
            if numeric_users.isnull().any() or numeric_ratings.isnull().any():
                raise ValueError("training user IDs and ratings must be numeric")

            user_values = numeric_users.to_numpy(dtype=np.float64)
            if (
                not np.isfinite(user_values).all()
                or not np.equal(user_values, np.floor(user_values)).all()
            ):
                raise ValueError("training user IDs must be finite integers")
            if (user_values < 0).any():
                raise ValueError("training user IDs must be non-negative")

            rating_values = numeric_ratings.to_numpy(dtype=np.float64)
            rating_buckets = _ratings_to_buckets(rating_values)
            slug_codes = chunk["slug"].cat.codes.to_numpy(dtype=np.int64)
            category_film_ids = np.fromiter(
                (
                    slug_to_film_id.get(str(slug), -1)
                    for slug in chunk["slug"].cat.categories
                ),
                dtype=np.int64,
                count=len(chunk["slug"].cat.categories),
            )
            mapped_films = category_film_ids[slug_codes]
            resolved_mask = mapped_films >= 0
            unresolved_rows += int((~resolved_mask).sum())
            if not resolved_mask.any():
                continue

            user_chunks.append(
                np.ascontiguousarray(user_values[resolved_mask], dtype=np.int64)
            )
            film_chunks.append(
                np.ascontiguousarray(
                    mapped_films[resolved_mask],
                    dtype=np.int64,
                )
            )
            rating_chunks.append(
                np.ascontiguousarray(rating_buckets[resolved_mask], dtype=np.int64)
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"unable to parse controlled training dataset: {exc}") from exc

    if not user_chunks:
        raise ValueError("controlled training dataset has no resolvable interactions")

    # Sort by user/film so duplicates become adjacent and user slices are stable.
    user_ids = np.concatenate(user_chunks)
    film_ids = np.concatenate(film_chunks)
    rating_buckets = np.concatenate(rating_chunks)
    resolved_rows = len(user_ids)
    order = np.lexsort((film_ids, user_ids))
    user_ids = user_ids[order]
    film_ids = film_ids[order]
    rating_buckets = rating_buckets[order]

    same_pair = (user_ids[1:] == user_ids[:-1]) & (film_ids[1:] == film_ids[:-1])
    if np.any(same_pair & (rating_buckets[1:] != rating_buckets[:-1])):
        raise ValueError("conflicting ratings found for the same user-film pair")
    keep = np.ones(len(user_ids), dtype=np.bool_)
    keep[1:] = ~same_pair
    duplicate_rows = int((~keep).sum())
    user_ids = user_ids[keep]
    film_ids = film_ids[keep]
    rating_buckets = rating_buckets[keep]

    # Replace database identities with compact vector rows while retaining the exact
    # sorted film-ID vocabulary required to translate model outputs later.
    vocabulary = np.unique(film_ids)
    item_rows = np.searchsorted(vocabulary, film_ids).astype(np.int64, copy=False)
    boundaries = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    stops = np.concatenate((boundaries, np.array([len(user_ids)])))
    users = tuple(
        HistoricalUser(
            cohort_user_id=int(user_ids[start]),
            item_rows=np.ascontiguousarray(item_rows[start:stop]),
            rating_buckets=np.ascontiguousarray(rating_buckets[start:stop]),
        )
        for start, stop in zip(starts, stops, strict=True)
    )
    summary = TrainingDataSummary(
        csv_rows=csv_rows,
        resolved_rows=resolved_rows,
        unresolved_rows=unresolved_rows,
        duplicate_rows=duplicate_rows,
        unique_users=len(users),
        unique_films=len(vocabulary),
    )
    return PreparedInteractions(
        film_ids=np.ascontiguousarray(vocabulary, dtype=np.int64),
        users=users,
        summary=summary,
    )


def build_interaction_splits(
    data: PreparedInteractions,
    *,
    positive_rating_threshold: float,
    negative_rating_threshold: float,
    seed: int,
) -> tuple[UserSplit, ...]:
    """Build leakage-safe deterministic train, validation, and test partitions.

    Users with at least three positive interactions contribute one validation and
    one test target selected by a user-specific seeded permutation. Both held-out
    positives are removed from profile context; explicit low-rating negatives remain
    available for training/evaluation semantics.

    Returns:
        tuple[UserSplit, ...]: One split per prepared user in stable cohort order.

    Raises:
        ValueError: If either rating threshold is not a valid half-star value.
    """
    positive_bucket = rating_to_bucket(positive_rating_threshold)
    negative_bucket = rating_to_bucket(negative_rating_threshold)
    splits: list[UserSplit] = []
    # Derive a separate RNG stream from global seed and stable cohort identity so
    # adding another user cannot perturb existing users' held-out targets.
    for user in data.users:
        positives = user.item_rows[user.rating_buckets >= positive_bucket]
        explicit_negatives = user.item_rows[user.rating_buckets <= negative_bucket]
        validation_target: int | None = None
        test_target: int | None = None
        training_positives = positives.copy()
        held_out: set[int] = set()
        if len(positives) >= 3:
            rng = _rng(seed, user.cohort_user_id, 0)
            shuffled = positives[rng.permutation(len(positives))]
            validation_target = int(shuffled[0])
            test_target = int(shuffled[1])
            training_positives = shuffled[2:]
            held_out = {validation_target, test_target}

        context_mask = np.array(
            [int(item_row) not in held_out for item_row in user.item_rows],
            dtype=np.bool_,
        )
        splits.append(
            UserSplit(
                cohort_user_id=user.cohort_user_id,
                all_item_rows=user.item_rows,
                all_rating_buckets=user.rating_buckets,
                context_item_rows=user.item_rows[context_mask],
                context_rating_buckets=user.rating_buckets[context_mask],
                training_positive_rows=np.ascontiguousarray(
                    training_positives,
                    dtype=np.int64,
                ),
                explicit_negative_rows=np.ascontiguousarray(
                    explicit_negatives,
                    dtype=np.int64,
                ),
                validation_target=validation_target,
                test_target=test_target,
            )
        )
    return tuple(splits)


def _ratings_to_buckets(ratings: NDArray[np.float64]) -> NDArray[np.int64]:
    """Vectorize strict half-star validation for one historical CSV chunk."""
    if not np.isfinite(ratings).all():
        raise ValueError("ratings must be finite")
    doubled = ratings * 2
    rounded = np.rint(doubled)
    if not np.allclose(doubled, rounded, rtol=0.0, atol=1e-7):
        raise ValueError("ratings must use half-star increments")
    if ((rounded < 1) | (rounded > 10)).any():
        raise ValueError("ratings must be between 0.5 and 5.0")
    return np.ascontiguousarray(rounded.astype(np.int64) - 1)


def _validate_film_mapping(slug_to_film_id: Mapping[str, int]) -> None:
    """Require a non-empty one-to-one mapping to positive integer film IDs."""
    if not slug_to_film_id:
        raise ValueError("film slug mapping is empty")
    film_ids = list(slug_to_film_id.values())
    if any(isinstance(value, bool) or not isinstance(value, int) for value in film_ids):
        raise ValueError("film mapping IDs must be integers")
    if any(value <= 0 for value in film_ids):
        raise ValueError("film mapping IDs must be positive")
    if len(set(film_ids)) != len(film_ids):
        raise ValueError("each model film ID must map to exactly one slug")


def _rng(*values: int) -> np.random.Generator:
    """Create a reproducible independent generator from stable integer components."""
    components = [int(value) & 0xFFFFFFFF for value in values]
    return np.random.default_rng(np.random.SeedSequence(components))
