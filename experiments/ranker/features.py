"""Deterministic ranker feature schema and leakage-safe feature construction."""

from dataclasses import dataclass
from math import log1p

import numpy as np
from numpy.typing import NDArray

from app.domain.candidates import RecommendationCandidate
from experiments.ranker.artifacts import FoldArtifacts
from experiments.ranker.catalog import RankerCatalog
from experiments.ranker.config import (
    AFFINITY_FEATURES,
    CATALOG_FEATURES,
    CATALOG_REFERENCE_YEAR,
    ENTITY_FAMILIES,
    ENTITY_SMOOTHING,
    FEATURE_NAMES,
    FILM_AGGREGATE_FEATURES,
    POPULARITY_DEPTH,
    PREFERENCE_FEATURES,
    SOURCE_FEATURES,
    SVD_DEPTH,
)


@dataclass(frozen=True, slots=True)
class EntityPreference:
    """Smoothed signed affinity and support keyed by compact entity ID."""

    affinity: dict[int, float]
    support: dict[int, int]


@dataclass(frozen=True, slots=True)
class UserFeatureProfile:
    """Context-only user statistics and reusable personalized preferences."""

    base_features: dict[str, float]
    entity_preferences: dict[str, EntityPreference]
    positive_year_mean: float
    positive_year_std: float
    positive_runtime_mean: float
    positive_runtime_std: float
    preferred_popularity_percentile: float
    stratum_positive_fractions: dict[str, float]


def feature_group_columns() -> dict[str, tuple[int, ...]]:
    """Return stable feature-index groups used by required ablations."""
    positions = {name: index for index, name in enumerate(FEATURE_NAMES)}

    def indexes(names: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(positions[name] for name in names)

    source = indexes(SOURCE_FEATURES)
    global_metadata = indexes((*FILM_AGGREGATE_FEATURES, *CATALOG_FEATURES))
    personalized = indexes((*AFFINITY_FEATURES, *PREFERENCE_FEATURES))
    no_svd_score_rank = {
        "svd_score",
        "svd_rank",
        "svd_reciprocal_rank",
        "svd_rank_fraction",
        "log_svd_rank",
        "source_rank_gap",
    }
    no_svd = tuple(
        index
        for index, name in enumerate(FEATURE_NAMES)
        if name not in no_svd_score_rank
    )
    no_popularity = tuple(
        index
        for index, name in enumerate(FEATURE_NAMES)
        if "popularity" not in name
        and name
        not in {
            "retrieved_by_popularity",
            "retrieved_by_both",
            "source_count",
            "source_rank_gap",
            "is_head",
            "is_mid",
            "is_tail",
            "head_positive_fraction",
            "mid_positive_fraction",
            "tail_positive_fraction",
            "candidate_stratum_preference",
        }
    )
    return {
        "source_only": source,
        "film_global_metadata_only": global_metadata,
        "personalized_affinity_only": personalized,
        "source_plus_global_metadata": tuple(sorted((*source, *global_metadata))),
        "full": tuple(range(len(FEATURE_NAMES))),
        "full_without_svd": no_svd,
        "full_without_popularity": no_popularity,
        "shuffled_personalization": tuple(range(len(FEATURE_NAMES))),
    }


def build_user_feature_profile(
    context_item_rows: NDArray[np.int64],
    context_rating_buckets: NDArray[np.int64],
    artifacts: FoldArtifacts,
    catalog: RankerCatalog,
    *,
    history_depth_thresholds: tuple[float, float, float],
) -> UserFeatureProfile:
    """Compute every dynamic feature from one user's visible context only."""
    ratings = (context_rating_buckets.astype(np.float64) + 1.0) / 2.0
    rated_count = len(ratings)
    positive_mask = ratings >= 3.5
    neutral_mask = ratings == 3.0
    negative_mask = ratings <= 2.5
    base = {
        "watched_count": np.nan,
        "rated_count": float(rated_count),
        "log_rated_count": log1p(rated_count),
        "mean_rating": float(ratings.mean()) if rated_count else np.nan,
        "rating_variance": float(ratings.var()) if rated_count else np.nan,
        "rating_std": float(ratings.std()) if rated_count else np.nan,
        "positive_count": float(positive_mask.sum()),
        "neutral_count": float(neutral_mask.sum()),
        "negative_count": float(negative_mask.sum()),
        "positive_fraction": (float(positive_mask.mean()) if rated_count else np.nan),
        "negative_fraction": (float(negative_mask.mean()) if rated_count else np.nan),
        "high_rating_fraction": (
            float((ratings >= 4.5).mean()) if rated_count else np.nan
        ),
        "rating_range": (
            float(ratings.max() - ratings.min()) if rated_count else np.nan
        ),
        "history_depth_bucket": float(
            np.searchsorted(history_depth_thresholds, rated_count, side="right")
        ),
        "unrated_watch_fraction": np.nan,
    }
    entity_preferences: dict[str, EntityPreference] = {}
    weights = ratings - 3.0
    for family in ENTITY_FAMILIES:
        sums: dict[int, float] = {}
        support: dict[int, int] = {}
        relation = catalog.relations[family]
        for film_row, weight in zip(context_item_rows, weights, strict=True):
            for raw_entity in relation.entities(int(film_row)):
                entity = int(raw_entity)
                sums[entity] = sums.get(entity, 0.0) + float(weight)
                support[entity] = support.get(entity, 0) + 1
        smoothing = ENTITY_SMOOTHING[family]
        entity_preferences[family] = EntityPreference(
            affinity={
                entity: total / (smoothing + support[entity])
                for entity, total in sums.items()
            },
            support=support,
        )

    positive_rows = context_item_rows[positive_mask]
    positive_ratings = ratings[positive_mask]
    positive_weights = np.maximum(positive_ratings - 3.0, 0.0)
    year_mean, year_std = _weighted_catalog_preference(
        catalog.years[positive_rows], positive_weights
    )
    runtime_mean, runtime_std = _weighted_catalog_preference(
        catalog.runtimes[positive_rows], positive_weights
    )
    preferred_popularity, _ = _weighted_catalog_preference(
        artifacts.popularity_percentiles[positive_rows], positive_weights
    )
    stratum_fractions = {name: np.nan for name in ("HEAD", "MID", "TAIL")}
    if len(positive_rows):
        positive_strata = artifacts.popularity_strata[positive_rows]
        stratum_fractions = {
            name: float(np.mean(positive_strata == name))
            for name in ("HEAD", "MID", "TAIL")
        }
    return UserFeatureProfile(
        base_features=base,
        entity_preferences=entity_preferences,
        positive_year_mean=year_mean,
        positive_year_std=year_std,
        positive_runtime_mean=runtime_mean,
        positive_runtime_std=runtime_std,
        preferred_popularity_percentile=preferred_popularity,
        stratum_positive_fractions=stratum_fractions,
    )


def build_feature_matrix(
    candidates: tuple[RecommendationCandidate, ...],
    profile: UserFeatureProfile,
    artifacts: FoldArtifacts,
    catalog: RankerCatalog,
    *,
    svd_profile_available: bool,
    preference_profile: UserFeatureProfile | None = None,
) -> NDArray[np.float32]:
    """Build an ordered float32 matrix without raw identity features."""
    taste = preference_profile or profile
    candidate_count = len(candidates)
    rows = np.full((candidate_count, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    if not candidate_count:
        return rows
    positions = {name: index for index, name in enumerate(FEATURE_NAMES)}
    film_rows = np.fromiter(
        (catalog.id_to_row[candidate.film_id] for candidate in candidates),
        dtype=np.int64,
        count=candidate_count,
    )
    _fill_source_columns(
        rows,
        candidates,
        positions,
        svd_profile_available=svd_profile_available,
    )
    for name, value in profile.base_features.items():
        rows[:, positions[name]] = value
    _fill_aggregate_columns(rows, film_rows, positions, artifacts)
    _fill_catalog_columns(rows, film_rows, positions, catalog)
    _fill_affinity_columns(rows, film_rows, positions, taste, catalog)
    _fill_preference_columns(rows, film_rows, positions, taste, artifacts, catalog)
    return rows


def build_personalized_feature_matrix(
    candidates: tuple[RecommendationCandidate, ...],
    profile: UserFeatureProfile,
    artifacts: FoldArtifacts,
    catalog: RankerCatalog,
) -> NDArray[np.float32]:
    """Build only affinity/preference columns for the shuffled control."""
    names = (*AFFINITY_FEATURES, *PREFERENCE_FEATURES)
    rows = np.full((len(candidates), len(names)), np.nan, dtype=np.float32)
    if not len(candidates):
        return rows
    positions = {name: index for index, name in enumerate(names)}
    film_rows = np.fromiter(
        (catalog.id_to_row[candidate.film_id] for candidate in candidates),
        dtype=np.int64,
        count=len(candidates),
    )
    _fill_affinity_columns(rows, film_rows, positions, profile, catalog)
    _fill_preference_columns(rows, film_rows, positions, profile, artifacts, catalog)
    return rows


def _fill_source_columns(
    rows: NDArray[np.float32],
    candidates: tuple[RecommendationCandidate, ...],
    positions: dict[str, int],
    *,
    svd_profile_available: bool,
) -> None:
    count = len(candidates)
    svd_scores = np.fromiter(
        (
            candidate.svd_score if candidate.svd_score is not None else np.nan
            for candidate in candidates
        ),
        dtype=np.float32,
        count=count,
    )
    svd_ranks = np.fromiter(
        (
            candidate.svd_rank if candidate.svd_rank is not None else np.nan
            for candidate in candidates
        ),
        dtype=np.float32,
        count=count,
    )
    popularity_counts = np.fromiter(
        (
            candidate.popularity_score
            if candidate.popularity_score is not None
            else np.nan
            for candidate in candidates
        ),
        dtype=np.float32,
        count=count,
    )
    popularity_ranks = np.fromiter(
        (
            candidate.popularity_rank
            if candidate.popularity_rank is not None
            else np.nan
            for candidate in candidates
        ),
        dtype=np.float32,
        count=count,
    )
    by_svd = np.fromiter(
        (candidate.retrieved_by_svd for candidate in candidates),
        dtype=np.float32,
        count=count,
    )
    by_popularity = np.fromiter(
        (candidate.retrieved_by_popularity for candidate in candidates),
        dtype=np.float32,
        count=count,
    )
    both = (by_svd == 1) & (by_popularity == 1)
    finite_svd = np.isfinite(svd_ranks)
    finite_popularity = np.isfinite(popularity_ranks)
    values = {
        "svd_score": svd_scores,
        "svd_rank": svd_ranks,
        "svd_reciprocal_rank": np.where(finite_svd, 1.0 / svd_ranks, np.nan),
        "svd_rank_fraction": np.where(finite_svd, svd_ranks / SVD_DEPTH, np.nan),
        "log_svd_rank": np.where(finite_svd, np.log1p(svd_ranks), np.nan),
        "popularity_positive_count": popularity_counts,
        "log_popularity_count": np.where(
            np.isfinite(popularity_counts), np.log1p(popularity_counts), np.nan
        ),
        "popularity_rank": popularity_ranks,
        "popularity_reciprocal_rank": np.where(
            finite_popularity, 1.0 / popularity_ranks, np.nan
        ),
        "popularity_rank_fraction": np.where(
            finite_popularity, popularity_ranks / POPULARITY_DEPTH, np.nan
        ),
        "retrieved_by_svd": by_svd,
        "retrieved_by_popularity": by_popularity,
        "retrieved_by_both": both,
        "source_count": by_svd + by_popularity,
        "source_rank_gap": np.where(
            both,
            np.abs(svd_ranks / SVD_DEPTH - popularity_ranks / POPULARITY_DEPTH),
            np.nan,
        ),
        "svd_profile_available": np.full(count, svd_profile_available),
    }
    for name, value in values.items():
        rows[:, positions[name]] = value


def _fill_aggregate_columns(
    rows: NDArray[np.float32],
    film_rows: NDArray[np.int64],
    positions: dict[str, int],
    artifacts: FoldArtifacts,
) -> None:
    values = {
        "cohort_rating_count": artifacts.rating_counts[film_rows],
        "cohort_positive_count": artifacts.positive_counts[film_rows],
        "cohort_negative_count": artifacts.negative_counts[film_rows],
        "cohort_mean_rating": artifacts.rating_means[film_rows],
        "cohort_rating_variance": artifacts.rating_variances[film_rows],
        "cohort_smoothed_rating": artifacts.smoothed_ratings[film_rows],
        "popularity_percentile": artifacts.popularity_percentiles[film_rows],
        "is_head": artifacts.popularity_strata[film_rows] == "HEAD",
        "is_mid": artifacts.popularity_strata[film_rows] == "MID",
        "is_tail": artifacts.popularity_strata[film_rows] == "TAIL",
    }
    for name, value in values.items():
        rows[:, positions[name]] = value


def _fill_catalog_columns(
    rows: NDArray[np.float32],
    film_rows: NDArray[np.int64],
    positions: dict[str, int],
    catalog: RankerCatalog,
) -> None:
    years = catalog.years[film_rows]
    runtimes = catalog.runtimes[film_rows]
    finite_year = np.isfinite(years)
    finite_runtime = np.isfinite(runtimes)
    rows[:, positions["release_year"]] = years
    rows[:, positions["release_decade"]] = np.where(
        finite_year, years // 10 * 10, np.nan
    )
    rows[:, positions["film_age"]] = np.where(
        finite_year, CATALOG_REFERENCE_YEAR - years, np.nan
    )
    rows[:, positions["runtime_minutes"]] = runtimes
    rows[:, positions["log_runtime"]] = np.where(
        finite_runtime, np.log1p(runtimes), np.nan
    )
    rows[:, positions["year_missing"]] = ~finite_year
    rows[:, positions["runtime_missing"]] = ~finite_runtime
    for family in ENTITY_FAMILIES[:-1]:
        rows[:, positions[f"{family}_count"]] = catalog.relations[family].counts()[
            film_rows
        ]


def _fill_affinity_columns(
    rows: NDArray[np.float32],
    film_rows: NDArray[np.int64],
    positions: dict[str, int],
    profile: UserFeatureProfile,
    catalog: RankerCatalog,
) -> None:
    for family in ENTITY_FAMILIES:
        relation = catalog.relations[family]
        preference = profile.entity_preferences[family]
        entity_count = int(relation.indices.max()) + 1 if len(relation.indices) else 0
        dense_affinity = np.zeros(entity_count, dtype=np.float32)
        dense_support = np.zeros(entity_count, dtype=np.int32)
        for entity, affinity in preference.affinity.items():
            dense_affinity[entity] = affinity
        for entity, support in preference.support.items():
            dense_support[entity] = support
        prefix = f"{family}_"
        counts = relation.counts()[film_rows].astype(np.int64, copy=False)
        entity_total = int(counts.sum())
        if not entity_total:
            for statistic in (
                "affinity_mean",
                "affinity_max",
                "affinity_min",
                "support_sum",
                "matched_count",
                "matched_fraction",
            ):
                rows[:, positions[f"{prefix}{statistic}"]] = 0.0
            continue
        output_rows = np.repeat(np.arange(len(film_rows)), counts)
        flat_offsets = np.repeat(relation.indptr[film_rows], counts)
        group_starts = np.repeat(np.cumsum(counts) - counts, counts)
        entity_positions = flat_offsets + np.arange(entity_total) - group_starts
        entities = relation.indices[entity_positions]
        affinities = dense_affinity[entities]
        supports = dense_support[entities]
        affinity_sums = np.bincount(
            output_rows, weights=affinities, minlength=len(film_rows)
        )
        affinity_means = np.divide(
            affinity_sums,
            counts,
            out=np.zeros(len(film_rows), dtype=np.float64),
            where=counts > 0,
        )
        affinity_max = np.full(len(film_rows), -np.inf, dtype=np.float32)
        affinity_min = np.full(len(film_rows), np.inf, dtype=np.float32)
        np.maximum.at(affinity_max, output_rows, affinities)
        np.minimum.at(affinity_min, output_rows, affinities)
        affinity_max[counts == 0] = 0.0
        affinity_min[counts == 0] = 0.0
        support_sums = np.bincount(
            output_rows, weights=supports, minlength=len(film_rows)
        )
        matched = np.bincount(
            output_rows, weights=supports > 0, minlength=len(film_rows)
        )
        rows[:, positions[f"{prefix}affinity_mean"]] = affinity_means
        rows[:, positions[f"{prefix}affinity_max"]] = affinity_max
        rows[:, positions[f"{prefix}affinity_min"]] = affinity_min
        rows[:, positions[f"{prefix}support_sum"]] = support_sums
        rows[:, positions[f"{prefix}matched_count"]] = matched
        rows[:, positions[f"{prefix}matched_fraction"]] = np.divide(
            matched,
            counts,
            out=np.zeros(len(film_rows), dtype=np.float64),
            where=counts > 0,
        )


def _fill_preference_columns(
    rows: NDArray[np.float32],
    film_rows: NDArray[np.int64],
    positions: dict[str, int],
    profile: UserFeatureProfile,
    artifacts: FoldArtifacts,
    catalog: RankerCatalog,
) -> None:
    years = catalog.years[film_rows]
    runtimes = catalog.runtimes[film_rows]
    popularity = artifacts.popularity_percentiles[film_rows]
    year_distance = (
        np.abs(years - profile.positive_year_mean)
        if np.isfinite(profile.positive_year_mean)
        else np.full(len(film_rows), np.nan)
    )
    runtime_distance = (
        np.abs(runtimes - profile.positive_runtime_mean)
        if np.isfinite(profile.positive_runtime_mean)
        else np.full(len(film_rows), np.nan)
    )
    rows[:, positions["positive_year_mean"]] = profile.positive_year_mean
    rows[:, positions["year_distance"]] = year_distance
    rows[:, positions["year_z_distance"]] = (
        year_distance / max(profile.positive_year_std, 1.0)
        if np.isfinite(profile.positive_year_std)
        else np.nan
    )
    rows[:, positions["positive_runtime_mean"]] = profile.positive_runtime_mean
    rows[:, positions["runtime_distance"]] = runtime_distance
    rows[:, positions["runtime_z_distance"]] = (
        runtime_distance / max(profile.positive_runtime_std, 1.0)
        if np.isfinite(profile.positive_runtime_std)
        else np.nan
    )
    rows[:, positions["preferred_popularity_percentile"]] = (
        profile.preferred_popularity_percentile
    )
    rows[:, positions["popularity_preference_distance"]] = (
        np.abs(popularity - profile.preferred_popularity_percentile)
        if np.isfinite(profile.preferred_popularity_percentile)
        else np.nan
    )
    for stratum, name in (
        ("HEAD", "head_positive_fraction"),
        ("MID", "mid_positive_fraction"),
        ("TAIL", "tail_positive_fraction"),
    ):
        rows[:, positions[name]] = profile.stratum_positive_fractions[stratum]
    candidate_strata = artifacts.popularity_strata[film_rows]
    rows[:, positions["candidate_stratum_preference"]] = np.asarray(
        [profile.stratum_positive_fractions[str(value)] for value in candidate_strata],
        dtype=np.float32,
    )


def _source_features(
    candidate: RecommendationCandidate, *, svd_profile_available: bool
) -> dict[str, float]:
    svd_rank = float(candidate.svd_rank) if candidate.svd_rank is not None else np.nan
    popularity_rank = (
        float(candidate.popularity_rank)
        if candidate.popularity_rank is not None
        else np.nan
    )
    both = candidate.retrieved_by_svd and candidate.retrieved_by_popularity
    return {
        "svd_score": candidate.svd_score if candidate.svd_score is not None else np.nan,
        "svd_rank": svd_rank,
        "svd_reciprocal_rank": 1.0 / svd_rank if np.isfinite(svd_rank) else np.nan,
        "svd_rank_fraction": svd_rank / SVD_DEPTH if np.isfinite(svd_rank) else np.nan,
        "log_svd_rank": log1p(svd_rank) if np.isfinite(svd_rank) else np.nan,
        "popularity_positive_count": (
            float(candidate.popularity_score)
            if candidate.popularity_score is not None
            else np.nan
        ),
        "log_popularity_count": (
            log1p(candidate.popularity_score)
            if candidate.popularity_score is not None
            else np.nan
        ),
        "popularity_rank": popularity_rank,
        "popularity_reciprocal_rank": (
            1.0 / popularity_rank if np.isfinite(popularity_rank) else np.nan
        ),
        "popularity_rank_fraction": (
            popularity_rank / POPULARITY_DEPTH
            if np.isfinite(popularity_rank)
            else np.nan
        ),
        "retrieved_by_svd": float(candidate.retrieved_by_svd),
        "retrieved_by_popularity": float(candidate.retrieved_by_popularity),
        "retrieved_by_both": float(both),
        "source_count": float(candidate.source_count),
        "source_rank_gap": (
            abs(svd_rank / SVD_DEPTH - popularity_rank / POPULARITY_DEPTH)
            if both
            else np.nan
        ),
        "svd_profile_available": float(svd_profile_available),
    }


def _film_aggregate_features(row: int, artifacts: FoldArtifacts) -> dict[str, float]:
    stratum = str(artifacts.popularity_strata[row])
    return {
        "cohort_rating_count": float(artifacts.rating_counts[row]),
        "cohort_positive_count": float(artifacts.positive_counts[row]),
        "cohort_negative_count": float(artifacts.negative_counts[row]),
        "cohort_mean_rating": float(artifacts.rating_means[row]),
        "cohort_rating_variance": float(artifacts.rating_variances[row]),
        "cohort_smoothed_rating": float(artifacts.smoothed_ratings[row]),
        "popularity_percentile": float(artifacts.popularity_percentiles[row]),
        "is_head": float(stratum == "HEAD"),
        "is_mid": float(stratum == "MID"),
        "is_tail": float(stratum == "TAIL"),
    }


def _catalog_features(row: int, catalog: RankerCatalog) -> dict[str, float]:
    year = float(catalog.years[row])
    runtime = float(catalog.runtimes[row])
    values = {
        "release_year": year,
        "release_decade": float(year // 10 * 10) if np.isfinite(year) else np.nan,
        "film_age": CATALOG_REFERENCE_YEAR - year if np.isfinite(year) else np.nan,
        "runtime_minutes": runtime,
        "log_runtime": log1p(runtime) if np.isfinite(runtime) else np.nan,
        "year_missing": float(not np.isfinite(year)),
        "runtime_missing": float(not np.isfinite(runtime)),
    }
    for family in ENTITY_FAMILIES[:-1]:
        values[f"{family}_count"] = float(len(catalog.relations[family].entities(row)))
    return {name: values[name] for name in CATALOG_FEATURES}


def _affinity_features(
    row: int, profile: UserFeatureProfile, catalog: RankerCatalog
) -> dict[str, float]:
    result: dict[str, float] = {}
    for family in ENTITY_FAMILIES:
        entities = catalog.relations[family].entities(row)
        preference = profile.entity_preferences[family]
        affinities = [preference.affinity.get(int(entity), 0.0) for entity in entities]
        supports = [preference.support.get(int(entity), 0) for entity in entities]
        matched = sum(value > 0 for value in supports)
        prefix = f"{family}_"
        result[f"{prefix}affinity_mean"] = (
            float(np.mean(affinities)) if affinities else 0.0
        )
        result[f"{prefix}affinity_max"] = max(affinities, default=0.0)
        result[f"{prefix}affinity_min"] = min(affinities, default=0.0)
        result[f"{prefix}support_sum"] = float(sum(supports))
        result[f"{prefix}matched_count"] = float(matched)
        result[f"{prefix}matched_fraction"] = (
            matched / len(entities) if len(entities) else 0.0
        )
    return result


def _preference_features(
    row: int,
    profile: UserFeatureProfile,
    artifacts: FoldArtifacts,
    catalog: RankerCatalog,
) -> dict[str, float]:
    year = float(catalog.years[row])
    runtime = float(catalog.runtimes[row])
    popularity = float(artifacts.popularity_percentiles[row])
    year_distance = (
        abs(year - profile.positive_year_mean)
        if np.isfinite(year) and np.isfinite(profile.positive_year_mean)
        else np.nan
    )
    runtime_distance = (
        abs(runtime - profile.positive_runtime_mean)
        if np.isfinite(runtime) and np.isfinite(profile.positive_runtime_mean)
        else np.nan
    )
    stratum = str(artifacts.popularity_strata[row])
    return {
        "positive_year_mean": profile.positive_year_mean,
        "year_distance": year_distance,
        "year_z_distance": _z_distance(year_distance, profile.positive_year_std),
        "positive_runtime_mean": profile.positive_runtime_mean,
        "runtime_distance": runtime_distance,
        "runtime_z_distance": _z_distance(
            runtime_distance, profile.positive_runtime_std
        ),
        "preferred_popularity_percentile": profile.preferred_popularity_percentile,
        "popularity_preference_distance": (
            abs(popularity - profile.preferred_popularity_percentile)
            if np.isfinite(profile.preferred_popularity_percentile)
            else np.nan
        ),
        "head_positive_fraction": profile.stratum_positive_fractions["HEAD"],
        "mid_positive_fraction": profile.stratum_positive_fractions["MID"],
        "tail_positive_fraction": profile.stratum_positive_fractions["TAIL"],
        "candidate_stratum_preference": profile.stratum_positive_fractions[stratum],
    }


def _weighted_catalog_preference(
    values: NDArray[np.floating], weights: NDArray[np.floating]
) -> tuple[float, float]:
    populated = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not populated.any():
        return np.nan, np.nan
    selected_values = values[populated].astype(np.float64)
    selected_weights = weights[populated].astype(np.float64)
    mean = float(np.average(selected_values, weights=selected_weights))
    variance = float(
        np.average(np.square(selected_values - mean), weights=selected_weights)
    )
    return mean, float(np.sqrt(variance))


def _z_distance(distance: float, standard_deviation: float) -> float:
    if not np.isfinite(distance) or not np.isfinite(standard_deviation):
        return np.nan
    return distance / max(standard_deviation, 1.0)
